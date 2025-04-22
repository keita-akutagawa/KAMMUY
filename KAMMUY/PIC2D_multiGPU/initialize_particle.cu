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
    const float xmin, const float xmax, 
    const int seed, const unsigned long long offset
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nEnd - nStart) {
        curandState state; 
        curand_init(seed, offset + i, 0, &state);
        float x = curand_uniform(&state) * (xmax - xmin) + xmin;
        particle[i + nStart].x = x;
        particle[i + nStart].isExist = true;
    }
}

void InitializeParticle::uniformForPosition_x(
    unsigned long long nStart, 
    unsigned long long nEnd, 
    float xmin, float xmax, 
    int seed, 
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
    const float ymin, const float ymax, 
    const int seed, const unsigned long long offset
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nEnd - nStart) {
        curandState state; 
        curand_init(seed, offset + i, 0, &state);
        float y = curand_uniform(&state) * (ymax - ymin) + ymin;
        particle[i + nStart].y = y;
        particle[i + nStart].isExist = true;
    }
}

void InitializeParticle::uniformForPosition_y(
    unsigned long long nStart, 
    unsigned long long nEnd, 
    float ymin, float ymax, 
    int seed, 
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
    const float bulkVxSpecies, const float bulkVySpecies, const float bulkVzSpecies, 
    const float vxThSpecies, const float vyThSpecies, const float vzThSpecies, 
    const unsigned long long nStart, const unsigned long long nEnd, 
    const int seed, const unsigned long long offset
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nEnd - nStart) {
        curandState state; 
        curand_init(seed, offset + i, 0, &state);

        float vx, vy, vz, gamma;

        while (true) {
            vx = bulkVxSpecies + curand_normal(&state) * vxThSpecies;
            vy = bulkVySpecies + curand_normal(&state) * vyThSpecies;
            vz = bulkVzSpecies + curand_normal(&state) * vzThSpecies;

            if (vx * vx + vy * vy + vz * vz < PIC2DConst::device_c * PIC2DConst::device_c) break;
        }

        gamma = 1.0f / sqrt(1.0f - (vx * vx + vy * vy + vz * vz) / (PIC2DConst::device_c * PIC2DConst::device_c));

        particle[i + nStart].vx = vx * gamma;
        particle[i + nStart].vy = vy * gamma;
        particle[i + nStart].vz = vz * gamma;
        particle[i + nStart].gamma = gamma;
    }
}


void InitializeParticle::maxwellDistributionForVelocity(
    float bulkVxSpecies, 
    float bulkVySpecies, 
    float bulkVzSpecies, 
    float vxThSpecies, 
    float vyThSpecies, 
    float vzThSpecies, 
    unsigned long long nStart, 
    unsigned long long nEnd, 
    int seed, 
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
    float xmin, float xmax, float ymin, float ymax, 
    float bulkVxSpecies, float bulkVySpecies, float bulkVzSpecies, 
    float vxThSpecies, float vyThSpecies, float vzThSpecies, 
    unsigned long long nStart, unsigned long long nEnd, 
    int seed, 
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
    Particle* particle, float sheatThickness, 
    const unsigned long long nStart, 
    const unsigned long long nEnd, 
    const int seed, const unsigned long long offset
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nEnd - nStart) {
        curandState state; 
        curand_init(seed, offset + i, 0, &state);
        float yCenter = 0.5f * (PIC2DConst::device_ymax - PIC2DConst::device_ymin) + PIC2DConst::device_ymin;

        float randomValue;
        float y;
        while (true) {
            randomValue = curand_uniform(&state);
            y = yCenter + sheatThickness * atanh(2.0f * randomValue - 1.0f);

            if (PIC2DConst::device_ymin + PIC2DConst::device_EPS < y && y < PIC2DConst::device_ymax - PIC2DConst::device_EPS) break;
        }
        
        particle[i + nStart].y = y;
        particle[i + nStart].isExist = true;
    }
}

void InitializeParticle::harrisForPosition_y(
    unsigned long long nStart, 
    unsigned long long nEnd, 
    int seed, 
    float sheatThickness, 
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
    Particle* particle, float sheatThickness, 
    const unsigned long long nStart, 
    const unsigned long long nEnd, 
    const int seed, const unsigned long long offset
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nEnd - nStart) {
        curandState state; 
        curand_init(seed, 100 * (i + offset), 0, &state);
        float yCenter = 0.5f * (PIC2DConst::device_ymax - PIC2DConst::device_ymin) + PIC2DConst::device_ymin;

        float randomValue;
        float y;
        while (true) {
            randomValue = curand_uniform(&state);
            y = randomValue * (PIC2DConst::device_ymax - PIC2DConst::device_ymin);

            if (randomValue < (1.0f - 1.0f / cosh((y - yCenter) / sheatThickness))) break;
        } 
        
        particle[i + nStart].y = y;
        particle[i + nStart].isExist = true;
    }
}

void InitializeParticle::harrisBackgroundForPosition_y(
    unsigned long long nStart, 
    unsigned long long nEnd, 
    int seed, 
    float sheatThickness, 
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

