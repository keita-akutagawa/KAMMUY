#include "initialize_particle.hpp"
#include <thrust/transform.h>
#include <thrust/random.h>
#include <curand_kernel.h>
#include <cmath>
#include <random>


using namespace PIC2DConst;

__global__ void uniformForPositionXY_maxwellDistributionForVelocity_detail_kernel(
    Particle* particle, 
    const double xmin, const double ymin,  
    const double bulkVxSpecies, const double bulkVySpecies, const double bulkVzSpecies, 
    const double vxThSpecies, const double vyThSpecies, const double vzThSpecies, 
    const unsigned long long nStart, const unsigned long long nEnd, const int seed
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nEnd - nStart) {
        curandState stateX; 
        curandState stateY; 
        curand_init(seed,           100 * i, 0, &stateX);
        curand_init(seed + 1000000, 100 * i, 0, &stateY);

        curandState stateVx; 
        curandState stateVy; 
        curandState stateVz; 
        curand_init(seed + 2000000, 100 * i, 0, &stateVx);
        curand_init(seed + 3000000, 100 * i, 0, &stateVy);
        curand_init(seed + 4000000, 100 * i, 0, &stateVz);

        double x, y, vx, vy, vz;
        while (true) {
            x = curand_uniform_double(&stateX) + xmin;
            y = curand_uniform_double(&stateY) + ymin;
            vx = bulkVxSpecies + curand_normal_double(&stateVx) * vxThSpecies;
            vy = bulkVySpecies + curand_normal_double(&stateVy) * vyThSpecies;
            vz = bulkVzSpecies + curand_normal_double(&stateVz) * vzThSpecies;

            if (vx * vx + vy * vy + vz * vz < device_c_PIC * device_c_PIC) break;
        }

        particle[i + nStart].x = x;
        particle[i + nStart].y = y;
        particle[i + nStart].z = 0.0;
        particle[i + nStart].vx = vx;
        particle[i + nStart].vy = vy;
        particle[i + nStart].vz = vz;
        particle[i + nStart].gamma = sqrt(1.0 + (vx * vx + vy * vy + vz * vz) / (device_c_PIC * device_c_PIC));
        particle[i + nStart].isExist = true;
    }
}

void InitializeParticle::uniformForPositionXY_maxwellDistributionForVelocity_detail(
    double xmin, double ymin, 
    double bulkVxSpecies, 
    double bulkVySpecies, 
    double bulkVzSpecies, 
    double vxThSpecies, 
    double vyThSpecies, 
    double vzThSpecies, 
    unsigned long long nStart, 
    unsigned long long nEnd, 
    int seed, 
    thrust::device_vector<Particle>& particlesSpecies
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((nEnd - nStart + threadsPerBlock.x - 1) / threadsPerBlock.x);

    uniformForPositionXY_maxwellDistributionForVelocity_detail_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        xmin, ymin, 
        bulkVxSpecies, bulkVySpecies, bulkVzSpecies, 
        vxThSpecies, vyThSpecies, vzThSpecies, 
        nStart, nEnd, seed
    );

    cudaDeviceSynchronize();
}



__global__ void uniformForPositionX_kernel(
    Particle* particle, 
    const unsigned long long nStart, const unsigned long long nEnd, const int seed
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nEnd - nStart) {
        curandState state; 
        curand_init(seed, i, 0, &state);
        double x = curand_uniform_double(&state) * (device_xmax_PIC - device_xmin_PIC) + device_xmin_PIC;
        particle[i + nStart].x = x;
        particle[i + nStart].isExist = true;
    }
}

void InitializeParticle::uniformForPositionX(
    unsigned long long nStart, 
    unsigned long long nEnd, 
    int seed, 
    thrust::device_vector<Particle>& particlesSpecies
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((nEnd - nStart + threadsPerBlock.x - 1) / threadsPerBlock.x);

    uniformForPositionX_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        nStart, nEnd, seed
    );

    cudaDeviceSynchronize();
}


__global__ void uniformForPositionY_kernel(
    Particle* particle, 
    const unsigned long long nStart, const unsigned long long nEnd, const int seed
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nEnd - nStart) {
        curandState state; 
        curand_init(seed, i, 0, &state);
        double y = curand_uniform_double(&state) * (device_ymax_PIC - device_ymin_PIC) + device_ymin_PIC;
        particle[i + nStart].y = y;
        particle[i + nStart].isExist = true;
    }
}

void InitializeParticle::uniformForPositionY(
    unsigned long long nStart, 
    unsigned long long nEnd, 
    int seed, 
    thrust::device_vector<Particle>& particlesSpecies
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((nEnd - nStart + threadsPerBlock.x - 1) / threadsPerBlock.x);

    uniformForPositionY_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        nStart, nEnd, seed
    );

    cudaDeviceSynchronize();
}


__global__ void uniformForPositionYDetail_kernel(
    Particle* particle, 
    const unsigned long long nStart, const unsigned long long nEnd, const int seed, const double ymin, const double ymax
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nEnd - nStart) {
        curandState state; 
        curand_init(seed, i, 0, &state);
        double y = curand_uniform_double(&state) * (ymax - ymin) + ymin;
        particle[i + nStart].y = y;
        particle[i + nStart].isExist = true;
    }
}

void InitializeParticle::uniformForPositionY_detail(
    unsigned long long nStart, 
    unsigned long long nEnd, 
    int seed, 
    double ymin, 
    double ymax, 
    thrust::device_vector<Particle>& particlesSpecies
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((nEnd - nStart + threadsPerBlock.x - 1) / threadsPerBlock.x);

    uniformForPositionYDetail_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        nStart, nEnd, seed, ymin, ymax
    );

    cudaDeviceSynchronize();
}


__global__ void maxwellDistributionForVelocity_kernel(
    Particle* particle, 
    const double bulkVxSpecies, const double bulkVySpecies, const double bulkVzSpecies, 
    const double vxThSpecies, const double vyThSpecies, const double vzThSpecies, 
    const unsigned long long nStart, const unsigned long long nEnd, const int seed
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nEnd - nStart) {
        curandState stateVx; 
        curandState stateVy; 
        curandState stateVz; 
        curand_init(seed,           100 * i, 0, &stateVx);
        curand_init(seed + 1000000, 100 * i, 0, &stateVy);
        curand_init(seed + 2000000, 100 * i, 0, &stateVz);

        double vx, vy, vz;

        while (true) {
            vx = bulkVxSpecies + curand_normal_double(&stateVx) * vxThSpecies;
            vy = bulkVySpecies + curand_normal_double(&stateVy) * vyThSpecies;
            vz = bulkVzSpecies + curand_normal_double(&stateVz) * vzThSpecies;

            if (vx * vx + vy * vy + vz * vz < device_c_PIC * device_c_PIC) break;
        }

        particle[i + nStart].vx = vx;
        particle[i + nStart].vy = vy;
        particle[i + nStart].vz = vz;
        particle[i + nStart].gamma = sqrt(1.0 + (vx * vx + vy * vy + vz * vz) / (device_c_PIC * device_c_PIC));
        particle[i + nStart].isExist = true;
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
        nStart, nEnd, seed
    );

    cudaDeviceSynchronize();
}


__global__ void harrisForPositionY_kernel(
    Particle* particle, double sheatThickness, 
    const unsigned long long nStart, const unsigned long long nEnd, const int seed
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nEnd - nStart) {
        curandState state; 
        curand_init(seed, 10 * i, 0, &state);
        double yCenter = 0.5 * (device_ymax_PIC - device_ymin_PIC) + device_ymin_PIC;

        double randomValue;
        double y;
        while (true) {
            randomValue = curand_uniform_double(&state);
            y = yCenter + sheatThickness * atanh(2.0 * randomValue - 1.0);

            if (device_ymin_PIC < y && y < device_ymax_PIC) break;
        }
        
        particle[i + nStart].y = y;
        particle[i + nStart].isExist = true;
    }
}

void InitializeParticle::harrisForPositionY(
    unsigned long long nStart, 
    unsigned long long nEnd, 
    int seed, 
    double sheatThickness, 
    thrust::device_vector<Particle>& particlesSpecies
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((nEnd - nStart + threadsPerBlock.x - 1) / threadsPerBlock.x);

    harrisForPositionY_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), sheatThickness, 
        nStart, nEnd, seed
    );

    cudaDeviceSynchronize();
}


__global__ void harrisBackgroundForPositionY_kernel(
    Particle* particle, double sheatThickness, 
    const unsigned long long nStart, const unsigned long long nEnd, const int seed
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nEnd - nStart) {
        curandState state; 
        curand_init(seed, 10 * i, 0, &state);
        double yCenter = 0.5 * (device_ymax_PIC - device_ymin_PIC) + device_ymin_PIC;

        double randomValue;
        double y;
        while (true) {
            randomValue = curand_uniform_double(&state);
            y = randomValue * (device_ymax_PIC - device_ymin_PIC);

            if (randomValue < (1.0 - 1.0 / cosh((y - yCenter) / sheatThickness))) break;
        } 
        
        particle[i + nStart].y = y;
        particle[i + nStart].isExist = true;
    }
}

void InitializeParticle::harrisBackgroundForPositionY(
    unsigned long long nStart, 
    unsigned long long nEnd, 
    int seed, 
    double sheatThickness, 
    thrust::device_vector<Particle>& particlesSpecies
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((nEnd - nStart + threadsPerBlock.x - 1) / threadsPerBlock.x);

    harrisBackgroundForPositionY_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), sheatThickness, 
        nStart, nEnd, seed
    );

    cudaDeviceSynchronize();
}

