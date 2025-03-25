#include "boundary.hpp"



void BoundaryPIC::freeBoundaryParticle_y(
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron
)
{   
    MPI_Barrier(MPI_COMM_WORLD); 
    freeBoundaryParticleOfOneSpecies_y(
        particlesIon,
        mPIInfo.existNumIonPerProcs
    );
    freeBoundaryParticleOfOneSpecies_y(
        particlesElectron, 
        mPIInfo.existNumElectronPerProcs
    );
    MPI_Barrier(MPI_COMM_WORLD);
}


__global__ void freeBoundaryParticle_y_kernel(
    Particle* particlesSpecies, 
    const unsigned long long existNumSpecies, 
    const int buffer
)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        
        float boundaryDown  = PIC2DConst::device_ymin + PIC2DConst::device_EPS; 
        float boundaryUp    = PIC2DConst::device_ymax - PIC2DConst::device_EPS;
        
        if (x <= boundaryLeft) {
            particlesSpecies[i].isExist = false;
            return;
        }
        if (x >= boundaryRight) {
            particlesSpecies[i].isExist = false;
            return;
        }
    }
}


void BoundaryPIC::freeBoundaryParticleOfOneSpecies_y(
    thrust::device_vector<Particle>& particlesSpecies, 
    unsigned long long& existNumSpecies
)
{   
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((existNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);

    freeBoundaryParticle_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        existNumSpecies, 
        mPIInfo.buffer
    );
    cudaDeviceSynchronize();

    auto partitionEnd = thrust::partition(
        particlesSpecies.begin(), particlesSpecies.end(), 
        [] __device__ (const Particle& p) { return p.isExist; }
    );
    existNumSpecies = thrust::distance(particlesSpecies.begin(), partitionEnd);
}


