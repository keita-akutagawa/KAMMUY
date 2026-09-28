#include "initialize_particle.hpp"
#include <thrust/transform.h>
#include <thrust/random.h>
#include <curand_kernel.h>
#include <cmath>
#include <random>


InitializeParticle::InitializeParticle(
    PICConstParameter& pICConstParameter, 
    const PICGridParameter& pICGridParameter
)
    : pICConstParameter(pICConstParameter), 
      pICGridParameter(pICGridParameter)
{
}


__global__ void uniformPositionX_kernel(
    const PICUnsignedLongLong nStart, const PICUnsignedLongLong nEnd, 
    const PICFloat xmin, const PICFloat xmax, 
    const PICUnsignedLongLong seed, 
    Particle* particles
)
{
    PICUnsignedLongLong i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nEnd - nStart) {
        curandState state; 
        curand_init(seed, i, 0, &state);
        PICFloat x = curand_uniform_double(&state) * (xmax - xmin) + xmin;
        particles[i + nStart].x = x;
        particles[i + nStart].isExist = true;
    }
}

__global__ void uniformPositionY_kernel(
    const PICUnsignedLongLong nStart, const PICUnsignedLongLong nEnd, 
    const PICFloat ymin, const PICFloat ymax, 
    const PICUnsignedLongLong seed, 
    Particle* particles
)
{
    PICUnsignedLongLong i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nEnd - nStart) {
        curandState state; 
        curand_init(seed, i, 0, &state);
        PICFloat y = curand_uniform_double(&state) * (ymax - ymin) + ymin;
        particles[i + nStart].y = y;
        particles[i + nStart].isExist = true;
    }
}


__global__ void maxwellDistributionVelocity_kernel(
    const PICFloat bulkVx, const PICFloat bulkVy, const PICFloat bulkVz, 
    const PICFloat vxTh, const PICFloat vyTh, const PICFloat vzTh, 
    const PICFloat C, 
    const PICUnsignedLongLong nStart, const PICUnsignedLongLong nEnd, 
    const PICUnsignedLongLong seed, 
    Particle* particles
)
{
    PICUnsignedLongLong i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nEnd - nStart) {
        curandState state; 
        curand_init(seed, i, 0, &state);

        PICFloat vx, vy, vz, gamma;

        while (true) {
            vx = bulkVx + curand_normal_double(&state) * vxTh;
            vy = bulkVy + curand_normal_double(&state) * vyTh;
            vz = bulkVz + curand_normal_double(&state) * vzTh;

            if (vx * vx + vy * vy + vz * vz < C * C) break;
        }

        gamma = 1.0 / sqrt(1.0 - (vx * vx + vy * vy + vz * vz) / (C * C));

        particles[i + nStart].ux = vx * gamma;
        particles[i + nStart].uy = vy * gamma;
        particles[i + nStart].uz = vz * gamma;
        particles[i + nStart].gamma = gamma;
    }
}



void InitializeParticle::uniformPosition_maxwellDistributionVelocity_eachCell(
    const PICFloat xmin, const PICFloat xmax, const PICFloat ymin, const PICFloat ymax, 
    const PICFloat bulkVx, const PICFloat bulkVy, const PICFloat bulkVz, 
    const PICFloat vxTh, const PICFloat vyTh, const PICFloat vzTh, 
    const PICUnsignedLongLong nStart, const PICUnsignedLongLong nEnd, 
    const PICUnsignedLongLong seed, 
    thrust::device_vector<Particle>& particles
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((nEnd - nStart + threadsPerBlock.x - 1) / threadsPerBlock.x);

    uniformPositionX_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        nStart, nEnd,
        xmin, xmax, 
        seed, 
        thrust::raw_pointer_cast(particles.data())
    );
    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess) {
        printf("Kernel launch failed at uniformPositionX_kernel: %s\n", cudaGetErrorString(err1));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err1 = cudaDeviceSynchronize();
    if (err1 != cudaSuccess) {
        printf("Kernel execution failed at uniformPositionX_kernel: %s\n", cudaGetErrorString(err1));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }

    uniformPositionY_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        nStart, nEnd,
        ymin, ymax, 
        seed + 10000000000, 
        thrust::raw_pointer_cast(particles.data())
    );
    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess) {
        printf("Kernel launch failed at uniformPositionY_kernel: %s\n", cudaGetErrorString(err2));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err2 = cudaDeviceSynchronize();
    if (err2 != cudaSuccess) {
        printf("Kernel execution failed at uniformPositionY_kernel: %s\n", cudaGetErrorString(err2));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }

    maxwellDistributionVelocity_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        bulkVx, bulkVy, bulkVz, 
        vxTh, vyTh, vzTh, 
        pICConstParameter.C, 
        nStart, nEnd, 
        seed + 20000000000, 
        thrust::raw_pointer_cast(particles.data())
    );
    cudaError_t err3 = cudaGetLastError();
    if (err3 != cudaSuccess) {
        printf("Kernel launch failed at maxwellDistributionVelocity_kernel: %s\n", cudaGetErrorString(err3));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err3 = cudaDeviceSynchronize();
    if (err3 != cudaSuccess) {
        printf("Kernel execution failed at maxwellDistributionVelocity_kernel: %s\n", cudaGetErrorString(err3));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
}
