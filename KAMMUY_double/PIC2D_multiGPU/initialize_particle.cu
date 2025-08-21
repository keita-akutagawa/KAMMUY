#include "initialize_particle.hpp"
#include <thrust/transform.h>
#include <thrust/random.h>
#include <curand_kernel.h>
#include <cmath>
#include <random>


InitializeParticle::InitializeParticle(PIC2DMPI::MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo)
{
}


__global__ void uniformForPosition_x_kernel(
    Particle* particle, 
    const unsigned long long nStart, const unsigned long long nEnd, 
    const double xmin, const double xmax, 
    const unsigned long long seed, const unsigned long long offset
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nEnd - nStart) {
        curandState state; 
        curand_init(seed, offset + i, 0, &state);
        double x = curand_uniform(&state) * (xmax - xmin) + xmin;
        particle[i + nStart].x = x;
        particle[i + nStart].isExist = true;
    }
}

void InitializeParticle::uniformForPosition_x(
    unsigned long long nStart, 
    unsigned long long nEnd, 
    double xmin, double xmax, 
    unsigned long long seed, 
    thrust::device_vector<Particle>& particlesSpecies
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((nEnd - nStart + threadsPerBlock.x - 1) / threadsPerBlock.x);

    uniformForPosition_x_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        nStart, nEnd,
        xmin, xmax, 
        seed, (mPIInfo.rank + 1) * (nEnd - nStart)
    );
    cudaDeviceSynchronize();
}


__global__ void uniformForPosition_y_kernel(
    Particle* particle, 
    const unsigned long long nStart, const unsigned long long nEnd, 
    const double ymin, const double ymax, 
    const unsigned long long seed, const unsigned long long offset
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nEnd - nStart) {
        curandState state; 
        curand_init(seed, offset + i, 0, &state);
        double y = curand_uniform(&state) * (ymax - ymin) + ymin;
        particle[i + nStart].y = y;
        particle[i + nStart].isExist = true;
    }
}

void InitializeParticle::uniformForPosition_y(
    unsigned long long nStart, 
    unsigned long long nEnd, 
    double ymin, double ymax, 
    unsigned long long seed, 
    thrust::device_vector<Particle>& particlesSpecies
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((nEnd - nStart + threadsPerBlock.x - 1) / threadsPerBlock.x);

    uniformForPosition_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        nStart, nEnd,
        ymin, ymax, 
        seed, (mPIInfo.rank + 1) * (nEnd - nStart)
    );
    cudaDeviceSynchronize();
}

//////////

__global__ void maxwellDistributionForVelocity_kernel(
    Particle* particle, 
    const double bulkVxSpecies, const double bulkVySpecies, const double bulkVzSpecies, 
    const double vxThSpecies, const double vyThSpecies, const double vzThSpecies, 
    const unsigned long long nStart, const unsigned long long nEnd, 
    const unsigned long long seed, const unsigned long long offset
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nEnd - nStart) {
        curandState state; 
        curand_init(seed, offset + i, 0, &state);

        double vx, vy, vz, gamma;

        while (true) {
            vx = bulkVxSpecies + curand_normal(&state) * vxThSpecies;
            vy = bulkVySpecies + curand_normal(&state) * vyThSpecies;
            vz = bulkVzSpecies + curand_normal(&state) * vzThSpecies;

            if (vx * vx + vy * vy + vz * vz < PIC2DConst::device_c * PIC2DConst::device_c) break;
        }

        gamma = 1.0 / sqrt(1.0 - (vx * vx + vy * vy + vz * vz) / (PIC2DConst::device_c * PIC2DConst::device_c));

        particle[i + nStart].vx = vx * gamma;
        particle[i + nStart].vy = vy * gamma;
        particle[i + nStart].vz = vz * gamma;
        particle[i + nStart].gamma = gamma;
    }
}


void InitializeParticle::maxwellDistributionForVelocity(
    double bulkVxSpecies, 
    double bulkVySpecies, 
    double bulkVzSpecies, 
    double vxThSpecies, 
    double vyThSpecies, 
    double vzThSpecies, 
    unsigned long long nStart, 
    unsigned long long nEnd, 
    unsigned long long seed, 
    thrust::device_vector<Particle>& particlesSpecies
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((nEnd - nStart + threadsPerBlock.x - 1) / threadsPerBlock.x);

    maxwellDistributionForVelocity_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        bulkVxSpecies, bulkVySpecies, bulkVzSpecies, 
        vxThSpecies, vyThSpecies, vzThSpecies, 
        nStart, nEnd, seed, (mPIInfo.rank + 1) * (nEnd - nStart)
    );
    cudaDeviceSynchronize();
}


void InitializeParticle::uniformForPosition_xy_maxwellDistributionForVelocity_eachCell(
    double xmin, double xmax, double ymin, double ymax, 
    double bulkVxSpecies, double bulkVySpecies, double bulkVzSpecies, 
    double vxThSpecies, double vyThSpecies, double vzThSpecies, 
    unsigned long long nStart, unsigned long long nEnd, 
    unsigned long long seed, 
    thrust::device_vector<Particle>& particlesSpecies
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((nEnd - nStart + threadsPerBlock.x - 1) / threadsPerBlock.x);

    uniformForPosition_x_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        nStart, nEnd,
        xmin, xmax, 
        seed, (mPIInfo.rank + 1) * (nEnd - nStart)
    );
    cudaDeviceSynchronize();

    uniformForPosition_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        nStart, nEnd,
        ymin, ymax, 
        seed + 1000, (mPIInfo.rank + 1) * (nEnd - nStart)
    );
    cudaDeviceSynchronize();

    maxwellDistributionForVelocity_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        bulkVxSpecies, bulkVySpecies, bulkVzSpecies, 
        vxThSpecies, vyThSpecies, vzThSpecies, 
        nStart, nEnd, 
        seed + 2000, (mPIInfo.rank + 1) * (nEnd - nStart)
    );
    cudaDeviceSynchronize();
}


__global__ void harrisForPosition_y_kernel(
    Particle* particle, double sheatThickness, 
    const unsigned long long nStart, 
    const unsigned long long nEnd, 
    const unsigned long long seed, const unsigned long long offset
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nEnd - nStart) {
        curandState state; 
        curand_init(seed, offset + i, 0, &state);
        double yCenter = 0.5 * (PIC2DConst::device_ymax - PIC2DConst::device_ymin) + PIC2DConst::device_ymin;

        double randomValue;
        double y;
        while (true) {
            randomValue = curand_uniform(&state);
            y = yCenter + sheatThickness * atanh(2.0 * randomValue - 1.0);

            if (PIC2DConst::device_ymin + PIC2DConst::device_EPS < y && y < PIC2DConst::device_ymax - PIC2DConst::device_EPS) break;
        }
        
        particle[i + nStart].y = y;
        particle[i + nStart].isExist = true;
    }
}

void InitializeParticle::harrisForPosition_y(
    unsigned long long nStart, 
    unsigned long long nEnd, 
    unsigned long long seed, 
    double sheatThickness, 
    thrust::device_vector<Particle>& particlesSpecies
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((nEnd - nStart + threadsPerBlock.x - 1) / threadsPerBlock.x);

    harrisForPosition_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), sheatThickness, 
        nStart, nEnd, 
        seed, (mPIInfo.rank + 1) * (nEnd - nStart)
    );
    cudaDeviceSynchronize();
}


__global__ void harrisBackgroundForPosition_y_kernel(
    Particle* particle, double sheatThickness, 
    const unsigned long long nStart, 
    const unsigned long long nEnd, 
    const unsigned long long seed, const unsigned long long offset
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nEnd - nStart) {
        curandState state; 
        curand_init(seed, offset + i, 0, &state);
        double yCenter = 0.5 * (PIC2DConst::device_ymax - PIC2DConst::device_ymin) + PIC2DConst::device_ymin;

        double randomValue;
        double y;
        while (true) {
            randomValue = curand_uniform(&state);
            randomValue = thrust::min(thrust::max(PIC2DConst::device_EPS, randomValue), 1.0 - PIC2DConst::device_EPS);
            y = randomValue * (PIC2DConst::device_ymax - PIC2DConst::device_ymin) + PIC2DConst::device_ymin;
            if (randomValue < 1.0 - pow(cosh((y - yCenter) / sheatThickness), -2)) break;
        } 
        
        particle[i + nStart].y = y;
        particle[i + nStart].isExist = true;
    }
}

void InitializeParticle::harrisBackgroundForPosition_y(
    unsigned long long nStart, 
    unsigned long long nEnd, 
    unsigned long long seed, 
    double sheatThickness, 
    thrust::device_vector<Particle>& particlesSpecies
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((nEnd - nStart + threadsPerBlock.x - 1) / threadsPerBlock.x);

    harrisBackgroundForPosition_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), sheatThickness, 
        nStart, nEnd, 
        seed, (mPIInfo.rank + 1) * (nEnd - nStart)
    );
    cudaDeviceSynchronize();
}

