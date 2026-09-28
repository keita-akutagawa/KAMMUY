#include "particle_push.hpp"


ParticlePush::ParticlePush(
    PICConstParameter& pICConstParameter, 
    const PICGridParameter& pICGridParameter
)
    : NX(pICGridParameter.NX), 
      NY(pICGridParameter.NY), 
      DX(pICGridParameter.DX), 
      DY(pICGridParameter.DY), 
      
      pICConstParameter(pICConstParameter), 
      pICGridParameter(pICGridParameter)
{
}


void ParticlePush::pushVelocity(
    const thrust::device_vector<MagneticField>& B, 
    const thrust::device_vector<ElectricField>& E, 
    const PICFloat DT, 
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron
)
{
    pushVelocityOfOneSpecies(
        B, E, 
        pICConstParameter.Q_ION, pICConstParameter.M_ION, 
        pICConstParameter.EXIST_NUM_ION, 
        DT,  
        particlesIon
    );
    pushVelocityOfOneSpecies(
        B, E, 
        pICConstParameter.Q_ELECTRON, pICConstParameter.M_ELECTRON, 
        pICConstParameter.EXIST_NUM_ELECTRON, 
        DT,  
        particlesElectron
    );
}


void ParticlePush::pushPosition(
    const PICFloat DT, 
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron
)
{
    pushPositionOfOneSpecies(
        pICConstParameter.EXIST_NUM_ION, 
        DT, 
        particlesIon
    );
    pushPositionOfOneSpecies(
        pICConstParameter.EXIST_NUM_ELECTRON, 
        DT, 
        particlesElectron
    );
}


//////////

__device__
ParticleField getParticleFields(
    const PICUnsignedInt NX, PICUnsignedInt NY, 
    const PICFloat DX, PICFloat DY, 
    const PICFloat XMIN, const PICFloat YMIN, 
    const MagneticField* B,
    const ElectricField* E, 
    const Particle& particle
)
{
    ParticleField particleField;

    PICFloat xOverDx = (particle.x - XMIN) / DX;
    PICFloat yOverDy = (particle.y - YMIN) / DY;

    PICUnsignedInt xIndex1 = floor(xOverDx);
    PICUnsignedInt xIndex2 = xIndex1 + 1;
    xIndex2 = (xIndex2 == NX) ? 0 : xIndex2;
    PICUnsignedInt yIndex1 = floor(yOverDy);
    PICUnsignedInt yIndex2 = yIndex1 + 1;
    yIndex2 = (yIndex2 == NY) ? 0 : yIndex2;

    if (xIndex1 >= NX) printf("x = %f, index = %u, ERROR\n", particle.x, xIndex1); 
    if (yIndex1 >= NY) printf("y = %f, index = %u, ERROR\n", particle.y, yIndex1);

    PICFloat cx1 = xOverDx - xIndex1;
    PICFloat cx2 = 1.0 - cx1;
    PICFloat cy1 = yOverDy - yIndex1;
    PICFloat cy2 = 1.0 - cy1;

    PICUnsignedLongLong index11 = getIndex<PICUnsignedLongLong>(xIndex1, yIndex1, NX, NY); 
    PICUnsignedLongLong index12 = getIndex<PICUnsignedLongLong>(xIndex1, yIndex2, NX, NY); 
    PICUnsignedLongLong index21 = getIndex<PICUnsignedLongLong>(xIndex2, yIndex1, NX, NY); 
    PICUnsignedLongLong index22 = getIndex<PICUnsignedLongLong>(xIndex2, yIndex2, NX, NY); 

    particleField.bX += B[index11].bX * cx2 * cy2;
    particleField.bX += B[index12].bX * cx2 * cy1;
    particleField.bX += B[index21].bX * cx1 * cy2;
    particleField.bX += B[index22].bX * cx1 * cy1;

    particleField.bY += B[index11].bY * cx2 * cy2;
    particleField.bY += B[index12].bY * cx2 * cy1;
    particleField.bY += B[index21].bY * cx1 * cy2;
    particleField.bY += B[index22].bY * cx1 * cy1;

    particleField.bZ += B[index11].bZ * cx2 * cy2;
    particleField.bZ += B[index12].bZ * cx2 * cy1;
    particleField.bZ += B[index21].bZ * cx1 * cy2;
    particleField.bZ += B[index22].bZ * cx1 * cy1;

    particleField.eX += E[index11].eX * cx2 * cy2;
    particleField.eX += E[index12].eX * cx2 * cy1;
    particleField.eX += E[index21].eX * cx1 * cy2;
    particleField.eX += E[index22].eX * cx1 * cy1;

    particleField.eY += E[index11].eY * cx2 * cy2;
    particleField.eY += E[index12].eY * cx2 * cy1;
    particleField.eY += E[index21].eY * cx1 * cy2;
    particleField.eY += E[index22].eY * cx1 * cy1;

    particleField.eZ += E[index11].eZ * cx2 * cy2;
    particleField.eZ += E[index12].eZ * cx2 * cy1;
    particleField.eZ += E[index21].eZ * cx1 * cy2;
    particleField.eZ += E[index22].eZ * cx1 * cy1;

    return particleField;
}


__global__ void pushVelocityOfOneSpecies_kernel(
    const PICUnsignedInt NX, const PICUnsignedInt NY, 
    const PICFloat DX, const PICFloat DY, 
    const PICFloat XMIN, const PICFloat YMIN, 
    const PICFloat C, 
    const MagneticField* B, const ElectricField* E, 
    const PICFloat Q, const PICFloat M, 
    const PICUnsignedLongLong EXIST_NUM, 
    const PICFloat DT, 
    Particle* particlesSpecies
)
{
    PICUnsignedLongLong i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < EXIST_NUM) {

        PICFloat qOverMTimesDtOver2 = Q / M * DT / 2.0;
        PICFloat tmp1OverC2 = 1.0 / (C * C);

        PICFloat& ux = particlesSpecies[i].ux;
        PICFloat& uy = particlesSpecies[i].uy;
        PICFloat& uz = particlesSpecies[i].uz;

        ParticleField particleField = getParticleFields(
            NX, NY, 
            DX, DY, 
            XMIN, YMIN, 
            B, 
            E, 
            particlesSpecies[i]
        );

        PICFloat& bx = particleField.bX;
        PICFloat& by = particleField.bY;
        PICFloat& bz = particleField.bZ; 
        PICFloat& ex = particleField.eX;
        PICFloat& ey = particleField.eY; 
        PICFloat& ez = particleField.eZ;

        PICFloat uxMinus = ux + qOverMTimesDtOver2 * ex;
        PICFloat uyMinus = uy + qOverMTimesDtOver2 * ey;
        PICFloat uzMinus = uz + qOverMTimesDtOver2 * ez;

        PICFloat gamma = sqrt(1.0 + (uxMinus * uxMinus + uyMinus * uyMinus + uzMinus * uzMinus));
        PICFloat tmpForT = qOverMTimesDtOver2 / gamma;
        PICFloat tx = tmpForT * bx;
        PICFloat ty = tmpForT * by;
        PICFloat tz = tmpForT * bz;

        PICFloat tmpForS = 2.0 / (1.0 + tx * tx + ty * ty + tz * tz);
        PICFloat sx = tmpForS * tx;
        PICFloat sy = tmpForS * ty;
        PICFloat sz = tmpForS * tz;

        PICFloat ux0 = uxMinus + (uyMinus * tz - uzMinus * ty);
        PICFloat uy0 = uyMinus + (uzMinus * tx - uxMinus * tz);
        PICFloat uz0 = uzMinus + (uxMinus * ty - uyMinus * tx);

        PICFloat uxPlus = uxMinus + (uy0 * sz - uz0 * sy);
        PICFloat uyPlus = uyMinus + (uz0 * sx - ux0 * sz);
        PICFloat uzPlus = uzMinus + (ux0 * sy - uy0 * sx);

        ux = uxPlus + qOverMTimesDtOver2 * ex;
        uy = uyPlus + qOverMTimesDtOver2 * ey;
        uz = uzPlus + qOverMTimesDtOver2 * ez;
        gamma = sqrt(1.0 + (ux * ux + uy * uy + uz * uz) * tmp1OverC2);

        particlesSpecies[i].ux = ux;
        particlesSpecies[i].uy = uy;
        particlesSpecies[i].uz = uz;
        particlesSpecies[i].gamma = gamma;
    }
}


void ParticlePush::pushVelocityOfOneSpecies(
    const thrust::device_vector<MagneticField>& B,
    const thrust::device_vector<ElectricField>& E, 
    const PICFloat Q, const PICFloat M, const PICUnsignedLongLong EXIST_NUM, 
    const PICFloat DT, 
    thrust::device_vector<Particle>& particles
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((EXIST_NUM + threadsPerBlock.x - 1) / threadsPerBlock.x);

    pushVelocityOfOneSpecies_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        NX, NY, 
        DX, DY, 
        pICGridParameter.XMIN, pICGridParameter.YMIN, 
        pICConstParameter.C, 
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(E.data()), 
        Q, M, 
        EXIST_NUM, 
        DT, 
        thrust::raw_pointer_cast(particles.data())
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed at pushVelocityOfOneSpecies_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed at pushVelocityOfOneSpecies_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
}


//////////

__global__
void pushPositionOfOneSpecies_kernel(
    const PICUnsignedLongLong EXIST_NUM, 
    const PICFloat DT, 
    Particle* particles
)
{
    PICUnsignedLongLong i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < EXIST_NUM) {
        PICFloat& ux = particles[i].ux;
        PICFloat& uy = particles[i].uy;
        PICFloat& uz = particles[i].uz;
        PICFloat& gamma = particles[i].gamma;
        PICFloat& xPast = particles[i].x;
        PICFloat& yPast = particles[i].y;
        PICFloat& zPast = particles[i].z;

        PICFloat dtOverGamma = DT / gamma;
        particles[i].x = xPast + dtOverGamma * ux;
        particles[i].y = yPast + dtOverGamma * uy;
        particles[i].z = zPast + dtOverGamma * uz;
    }
}


void ParticlePush::pushPositionOfOneSpecies(
    const PICUnsignedLongLong EXIST_NUM, 
    const PICFloat DT, 
    thrust::device_vector<Particle>& particles
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((EXIST_NUM + threadsPerBlock.x - 1) / threadsPerBlock.x);

    pushPositionOfOneSpecies_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        EXIST_NUM, 
        DT, 
        thrust::raw_pointer_cast(particles.data())
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed at pushPositionOfOneSpecies_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed at pushPositionOfOneSpecies_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
}


