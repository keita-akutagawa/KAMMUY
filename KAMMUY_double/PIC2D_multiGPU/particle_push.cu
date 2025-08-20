#include "particle_push.hpp"


ParticlePush::ParticlePush(PIC2DMPI::MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo)
{
}


void ParticlePush::pushVelocity(
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron, 
    const thrust::device_vector<MagneticField>& B, 
    const thrust::device_vector<ElectricField>& E, 
    double dt
)
{
    MPI_Barrier(MPI_COMM_WORLD);
    pushVelocityOfOneSpecies(
        particlesIon, B, E, PIC2DConst::qIon, PIC2DConst::mIon, 
        mPIInfo.existNumIonPerProcs, dt
    );
    pushVelocityOfOneSpecies(
        particlesElectron, B, E, PIC2DConst::qElectron, PIC2DConst::mElectron, 
        mPIInfo.existNumElectronPerProcs, dt
    );
    MPI_Barrier(MPI_COMM_WORLD);
}


void ParticlePush::pushPosition(
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron, 
    const double dt
)
{
    MPI_Barrier(MPI_COMM_WORLD);
    pushPositionOfOneSpecies(
        particlesIon, mPIInfo.existNumIonPerProcs, 
        dt
    );
    pushPositionOfOneSpecies(
        particlesElectron, mPIInfo.existNumElectronPerProcs, 
        dt
    );
    MPI_Barrier(MPI_COMM_WORLD);
}


//////////

__device__
ParticleField getParticleFields(
    const MagneticField* B,
    const ElectricField* E, 
    Particle& particle, 
    const int localNx, const int buffer, 
    const int localSizeX, 
    const double xminForProcs, const double xmaxForProcs
)
{
    ParticleField particleField;

    double cx1, cx2; 
    int xIndex1, xIndex2;
    double xOverDx;
    double cy1, cy2; 
    int yIndex1, yIndex2;
    double yOverDy;

    xOverDx = (particle.x - xminForProcs + buffer * PIC2DConst::device_dx) / PIC2DConst::device_dx;
    yOverDy = (particle.y - PIC2DConst::device_ymin) / PIC2DConst::device_dy;

    xIndex1 = floor(xOverDx);
    xIndex2 = xIndex1 + 1;
    xIndex2 = (xIndex2 == localSizeX) ? 0 : xIndex2;
    yIndex1 = floor(yOverDy);
    yIndex2 = yIndex1 + 1;
    yIndex2 = (yIndex2 == PIC2DConst::device_ny) ? 0 : yIndex2;

    if (xIndex1 < 0 || xIndex1 >= localSizeX) printf("x = %f, index = %d, ERROR\n", particle.x, xIndex1); 
    if (yIndex1 < 0 || yIndex1 >= PIC2DConst::device_ny) printf("y = %f, index = %d, ERROR\n", particle.y, yIndex1);

    cx1 = xOverDx - xIndex1;
    cx2 = 1.0f - cx1;
    cy1 = yOverDy - yIndex1;
    cy2 = 1.0f - cy1;

    unsigned long long index11 = yIndex1 + PIC2DConst::device_ny * xIndex1; 
    unsigned long long index12 = yIndex2 + PIC2DConst::device_ny * xIndex1; 
    unsigned long long index21 = yIndex1 + PIC2DConst::device_ny * xIndex2; 
    unsigned long long index22 = yIndex2 + PIC2DConst::device_ny * xIndex2; 

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
    Particle* particlesSpecies, const MagneticField* B, const ElectricField* E, 
    double q, double m, unsigned long long existNumSpecies, double dt, 
    const int localNx, const int buffer, 
    const int localSizeX, 
    const double xminForProcs, const double xmaxForProcs
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        double qOverMTimesDtOver2;
        double tmpForT, tmpForS, tmp1OverC2;
        double vx, vy, vz, gamma;
        double tx, ty, tz;
        double sx, sy, sz;
        double vxMinus, vyMinus, vzMinus;
        double vx0, vy0, vz0;
        double vxPlus, vyPlus, vzPlus; 
        double bx, by, bz;
        double ex, ey, ez;
        ParticleField particleField;

        qOverMTimesDtOver2 = q / m * dt / 2.0f;
        tmp1OverC2 = 1.0f / (PIC2DConst::device_c * PIC2DConst::device_c);

        vx = particlesSpecies[i].vx;
        vy = particlesSpecies[i].vy;
        vz = particlesSpecies[i].vz;
        gamma = particlesSpecies[i].gamma;

        particleField = getParticleFields(
            B, E, particlesSpecies[i], 
            localNx, buffer, 
            localSizeX, 
            xminForProcs, xmaxForProcs
        );

        bx = particleField.bX;
        by = particleField.bY;
        bz = particleField.bZ; 
        ex = particleField.eX;
        ey = particleField.eY; 
        ez = particleField.eZ;

        tmpForT = qOverMTimesDtOver2 / gamma;
        tx = tmpForT * bx;
        ty = tmpForT * by;
        tz = tmpForT * bz;

        tmpForS = 2.0f / (1.0f + tx * tx + ty * ty + tz * tz);
        sx = tmpForS * tx;
        sy = tmpForS * ty;
        sz = tmpForS * tz;

        vxMinus = vx + qOverMTimesDtOver2 * ex;
        vyMinus = vy + qOverMTimesDtOver2 * ey;
        vzMinus = vz + qOverMTimesDtOver2 * ez;

        vx0 = vxMinus + (vyMinus * tz - vzMinus * ty);
        vy0 = vyMinus + (vzMinus * tx - vxMinus * tz);
        vz0 = vzMinus + (vxMinus * ty - vyMinus * tx);

        vxPlus = vxMinus + (vy0 * sz - vz0 * sy);
        vyPlus = vyMinus + (vz0 * sx - vx0 * sz);
        vzPlus = vzMinus + (vx0 * sy - vy0 * sx);

        vx = vxPlus + qOverMTimesDtOver2 * ex;
        vy = vyPlus + qOverMTimesDtOver2 * ey;
        vz = vzPlus + qOverMTimesDtOver2 * ez;
        gamma = sqrt(1.0f + (vx * vx + vy * vy + vz * vz) * tmp1OverC2);

        particlesSpecies[i].vx = vx;
        particlesSpecies[i].vy = vy;
        particlesSpecies[i].vz = vz;
        particlesSpecies[i].gamma = gamma;
    }
}


void ParticlePush::pushVelocityOfOneSpecies(
    thrust::device_vector<Particle>& particlesSpecies, 
    const thrust::device_vector<MagneticField>& B,
    const thrust::device_vector<ElectricField>& E, 
    const double q, const double m, const unsigned long long existNumSpecies, 
    const double dt
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((existNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);

    pushVelocityOfOneSpecies_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(E.data()), 
        q, m, existNumSpecies, dt, 
        mPIInfo.localNx, mPIInfo.buffer, 
        mPIInfo.localSizeX, 
        mPIInfo.xminForProcs, mPIInfo.xmaxForProcs
    );
}


//////////

__global__
void pushPositionOfOneSpecies_kernel(
    Particle* particlesSpecies, 
    const unsigned long long existNumSpecies, 
    const double xminForProcs, const double xmaxForProcs, const int buffer, 
    const double dt
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        
        double vx, vy, vz, gamma;
        double xPast, yPast, zPast;
        double x, y, z;
        double dtOverGamma;

        vx = particlesSpecies[i].vx;
        vy = particlesSpecies[i].vy;
        vz = particlesSpecies[i].vz;
        gamma = particlesSpecies[i].gamma;
        xPast = particlesSpecies[i].x;
        yPast = particlesSpecies[i].y;
        zPast = particlesSpecies[i].z;

        dtOverGamma = dt / gamma;
        x = xPast + dtOverGamma * vx;
        y = yPast + dtOverGamma * vy;
        z = zPast + dtOverGamma * vz;

        particlesSpecies[i].x = x;
        particlesSpecies[i].y = y;
        particlesSpecies[i].z = z;

    }
}


void ParticlePush::pushPositionOfOneSpecies(
    thrust::device_vector<Particle>& particlesSpecies, 
    unsigned long long& existNumSpecies, 
    const double dt
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((existNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);

    pushPositionOfOneSpecies_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        existNumSpecies, 
        mPIInfo.xminForProcs, mPIInfo.xmaxForProcs, mPIInfo.buffer, 
        dt
    );
    cudaDeviceSynchronize();
}


