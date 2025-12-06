#include "boundary.hpp"


void BoundaryPIC::periodicBoundaryParticle_x(
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron
)
{   
    periodicBoundaryParticleOfOneSpecies_x(
        particlesIon,
        PIC2DConst::existNumIon
    );
    periodicBoundaryParticleOfOneSpecies_x(
        particlesElectron, 
        PIC2DConst::existNumElectron
    );
}


__global__ void periodicBoundaryParticle_x_kernel(
    Particle* particlesSpecies, 
    const unsigned long long existNumSpecies
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        double x = particlesSpecies[i].x; 

        if (x < PIC2DConst::device_xmin) {
            particlesSpecies[i].x += PIC2DConst::device_xmax;
            return;
        }
        if (x > PIC2DConst::device_xmax) {
            particlesSpecies[i].x -= PIC2DConst::device_xmax;
            return;
        }
    }
}

void BoundaryPIC::periodicBoundaryParticleOfOneSpecies_x(
    thrust::device_vector<Particle>& particlesSpecies, 
    unsigned long long& existNumSpecies
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((existNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);

    periodicBoundaryParticle_x_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        existNumSpecies
    );
    cudaDeviceSynchronize();
}


