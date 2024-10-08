#include <cmath>
#include "particle_push.hpp"


using namespace PIC2DConst;

void ParticlePush::pushVelocity(
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron, 
    const thrust::device_vector<MagneticField>& B, 
    const thrust::device_vector<ElectricField>& E, 
    double dt
)
{
    pushVelocityOfOneSpecies(
        particlesIon, B, E, qIon_PIC, mIon_PIC, existNumIon_PIC, dt
    );
    pushVelocityOfOneSpecies(
        particlesElectron, B, E, qElectron_PIC, mElectron_PIC, existNumElectron_PIC, dt
    );
}


void ParticlePush::pushPosition(
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron, 
    double dt
)
{
    pushPositionOfOneSpecies(
        particlesIon, existNumIon_PIC, dt
    );
    pushPositionOfOneSpecies(
        particlesElectron, existNumElectron_PIC, dt
    );
}


//////////

__device__
ParticleField getParticleFields(
    const MagneticField* B,
    const ElectricField* E, 
    const Particle& particle
)
{
    ParticleField particleField;

    double cx1, cx2; 
    int xIndex1, xIndex2;
    double cy1, cy2; 
    int yIndex1, yIndex2;
    
    double xOverDx;
    xOverDx = particle.x / device_dx_PIC;
    double yOverDy;
    yOverDy = particle.y / device_dy_PIC;

    xIndex1 = floorf(xOverDx);
    xIndex2 = xIndex1 + 1;
    xIndex2 = (xIndex2 == device_nx_PIC) ? 0 : xIndex2;
    yIndex1 = floorf(yOverDy);
    yIndex2 = yIndex1 + 1;
    yIndex2 = (yIndex2 == device_ny_PIC) ? 0 : yIndex2;

    cx1 = xOverDx - xIndex1;
    cx2 = 1.0 - cx1;
    cy1 = yOverDy - yIndex1;
    cy2 = 1.0 - cy1;

    particleField.bX += B[yIndex1 + device_ny_PIC * xIndex1].bX * cx2 * cy2;
    particleField.bX += B[yIndex2 + device_ny_PIC * xIndex1].bX * cx2 * cy1 * min(1, yIndex2);
    particleField.bX += B[yIndex1 + device_ny_PIC * xIndex2].bX * cx1 * cy2 * min(1, yIndex2);
    particleField.bX += B[yIndex2 + device_ny_PIC * xIndex2].bX * cx1 * cy1 * min(1, yIndex2);

    particleField.bY += B[yIndex1 + device_ny_PIC * xIndex1].bY * cx2 * cy2;
    particleField.bY += B[yIndex2 + device_ny_PIC * xIndex1].bY * cx2 * cy1 * min(1, yIndex2);
    particleField.bY += B[yIndex1 + device_ny_PIC * xIndex2].bY * cx1 * cy2 * min(1, yIndex2);
    particleField.bY += B[yIndex2 + device_ny_PIC * xIndex2].bY * cx1 * cy1 * min(1, yIndex2);

    particleField.bZ += B[yIndex1 + device_ny_PIC * xIndex1].bZ * cx2 * cy2;
    particleField.bZ += B[yIndex2 + device_ny_PIC * xIndex1].bZ * cx2 * cy1 * min(1, yIndex2);
    particleField.bZ += B[yIndex1 + device_ny_PIC * xIndex2].bZ * cx1 * cy2 * min(1, yIndex2);
    particleField.bZ += B[yIndex2 + device_ny_PIC * xIndex2].bZ * cx1 * cy1 * min(1, yIndex2);

    particleField.eX += E[yIndex1 + device_ny_PIC * xIndex1].eX * cx2 * cy2;
    particleField.eX += E[yIndex2 + device_ny_PIC * xIndex1].eX * cx2 * cy1 * min(1, yIndex2);
    particleField.eX += E[yIndex1 + device_ny_PIC * xIndex2].eX * cx1 * cy2 * min(1, yIndex2);
    particleField.eX += E[yIndex2 + device_ny_PIC * xIndex2].eX * cx1 * cy1 * min(1, yIndex2);

    particleField.eY += E[yIndex1 + device_ny_PIC * xIndex1].eY * cx2 * cy2;
    particleField.eY += E[yIndex2 + device_ny_PIC * xIndex1].eY * cx2 * cy1 * min(1, yIndex2);
    particleField.eY += E[yIndex1 + device_ny_PIC * xIndex2].eY * cx1 * cy2 * min(1, yIndex2);
    particleField.eY += E[yIndex2 + device_ny_PIC * xIndex2].eY * cx1 * cy1 * min(1, yIndex2);

    particleField.eZ += E[yIndex1 + device_ny_PIC * xIndex1].eZ * cx2 * cy2;
    particleField.eZ += E[yIndex2 + device_ny_PIC * xIndex1].eZ * cx2 * cy1 * min(1, yIndex2);
    particleField.eZ += E[yIndex1 + device_ny_PIC * xIndex2].eZ * cx1 * cy2 * min(1, yIndex2);
    particleField.eZ += E[yIndex2 + device_ny_PIC * xIndex2].eZ * cx1 * cy1 * min(1, yIndex2);


    return particleField;
}


__global__
void pushVelocityOfOneSpecies_kernel(
    Particle* particlesSpecies, const MagneticField* B, const ElectricField* E, 
    double q, double m, unsigned long long existNumSpecies, double dt
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

        qOverMTimesDtOver2 = q / m * dt / 2.0;
        tmp1OverC2 = 1.0 / (device_c_PIC * device_c_PIC);


        vx = particlesSpecies[i].vx;
        vy = particlesSpecies[i].vy;
        vz = particlesSpecies[i].vz;
        gamma = particlesSpecies[i].gamma;

        particleField = getParticleFields(B, E, particlesSpecies[i]);
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

        tmpForS = 2.0 / (1.0 + tx * tx + ty * ty + tz * tz);
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
        gamma = sqrt(1.0 + (vx * vx + vy * vy + vz * vz) * tmp1OverC2);

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
    double q, double m, unsigned long long existNumSpecies, 
    double dt
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((existNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);

    pushVelocityOfOneSpecies_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(E.data()), 
        q, m, existNumSpecies, dt
    );

    cudaDeviceSynchronize();
}


//////////

__global__
void pushPositionOfOneSpecies_kernel(
    Particle* particlesSpecies, unsigned long long existNumSpecies, double dt
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        double vx, vy, vz, gamma;
        double x, y, z;
        double dtOverGamma;

        vx = particlesSpecies[i].vx;
        vy = particlesSpecies[i].vy;
        vz = particlesSpecies[i].vz;
        gamma = particlesSpecies[i].gamma;
        x = particlesSpecies[i].x;
        y = particlesSpecies[i].y;
        z = particlesSpecies[i].z;

        dtOverGamma = dt / gamma;
        x += dtOverGamma * vx;
        y += dtOverGamma * vy;
        z += dtOverGamma * vz;

        particlesSpecies[i].x = x;
        particlesSpecies[i].y = y;
        particlesSpecies[i].z = z;
    }
}


void ParticlePush::pushPositionOfOneSpecies(
    thrust::device_vector<Particle>& particlesSpecies, 
    unsigned long long existNumSpecies, 
    double dt
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((existNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);

    pushPositionOfOneSpecies_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        existNumSpecies, dt
    );

    cudaDeviceSynchronize();
}


