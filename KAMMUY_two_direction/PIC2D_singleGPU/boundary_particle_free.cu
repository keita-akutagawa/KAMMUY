#include "boundary.hpp"



void BoundaryPIC::freeBoundaryParticle_y(
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron
)
{   
    freeBoundaryParticleOfOneSpecies_y(
        particlesIon,
        PIC2DConst::existNumIon
    );
    freeBoundaryParticleOfOneSpecies_y(
        particlesElectron, 
        PIC2DConst::existNumElectron
    );

    if (PIC2DConst::existNumIon > PIC2DConst::totalNumIon) std::cout << "BROKEN" << std::endl;
    if (PIC2DConst::existNumElectron > PIC2DConst::totalNumElectron) std::cout << "BROKEN" << std::endl;
}


__global__ void freeBoundaryParticle_y_kernel(
    Particle* particlesSpecies, 
    Particle* sendParticlesSpeciesYDown, 
    Particle* sendParticlesSpeciesYUp, 
    unsigned long long* countForFreeBoundaryParticlesSpeciesYDown, 
    unsigned long long* countForFreeBoundaryParticlesSpeciesYUp, 
    const unsigned long long existNumSpecies
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        double y = particlesSpecies[i].y; 
        
        double boundaryDown  = PIC2DConst::device_ymin; 
        double boundaryUp    = PIC2DConst::device_ymax;
        
        if (y <= boundaryDown + PIC2DConst::device_dy) {
            particlesSpecies[i].isExist = false; 
        }
        if (y >= boundaryUp - PIC2DConst::device_dy) {
            particlesSpecies[i].isExist = false; 
        }
        
        if (y > boundaryDown + PIC2DConst::device_dy && y <= boundaryDown + 2 * PIC2DConst::device_dy) {
            unsigned long long particleIndex = atomicAdd(&(countForFreeBoundaryParticlesSpeciesYDown[0]), 1);
            Particle sendParticle = particlesSpecies[i];
            sendParticle.y = sendParticle.y - PIC2DConst::device_dy + PIC2DConst::device_EPS; 
            sendParticlesSpeciesYDown[particleIndex] = sendParticle;
        }

        if (y < boundaryUp - PIC2DConst::device_dy && y >= boundaryUp - 2 * PIC2DConst::device_dy) {
            unsigned long long particleIndex = atomicAdd(&(countForFreeBoundaryParticlesSpeciesYUp[0]), 1);
            Particle sendParticle = particlesSpecies[i];
            sendParticle.y = sendParticle.y + PIC2DConst::device_dy - PIC2DConst::device_EPS; 
            sendParticlesSpeciesYUp[particleIndex] = sendParticle;
        }
    }
}

void BoundaryPIC::freeBoundaryParticleOfOneSpecies_y(
    thrust::device_vector<Particle>& particlesSpecies, 
    unsigned long long& existNumSpecies
)
{   
    thrust::device_vector<unsigned long long> countForFreeBoundaryParticlesSpeciesYDown(1, 0); 
    thrust::device_vector<unsigned long long> countForFreeBoundaryParticlesSpeciesYUp(1, 0); 

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((existNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);

    freeBoundaryParticle_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        thrust::raw_pointer_cast(sendParticlesSpeciesYDown.data()), 
        thrust::raw_pointer_cast(sendParticlesSpeciesYUp.data()), 
        thrust::raw_pointer_cast(countForFreeBoundaryParticlesSpeciesYDown.data()), 
        thrust::raw_pointer_cast(countForFreeBoundaryParticlesSpeciesYUp.data()), 
        existNumSpecies
    );
    cudaDeviceSynchronize();

    auto partitionEnd = thrust::partition(
        particlesSpecies.begin(), particlesSpecies.begin() + existNumSpecies, 
        [] __device__ (const Particle& p) { return p.isExist; }
    );
    existNumSpecies = static_cast<unsigned long long>(thrust::distance(particlesSpecies.begin(), partitionEnd));

    //sendはしないので注意。再利用しているだけ。

    thrust::copy(
        sendParticlesSpeciesYDown.begin(), 
        sendParticlesSpeciesYDown.begin() + countForFreeBoundaryParticlesSpeciesYDown[0],
        particlesSpecies.begin() + existNumSpecies
    );
    existNumSpecies += countForFreeBoundaryParticlesSpeciesYDown[0];
    thrust::copy(
        sendParticlesSpeciesYUp.begin(), 
        sendParticlesSpeciesYUp.begin() + countForFreeBoundaryParticlesSpeciesYUp[0],
        particlesSpecies.begin() + existNumSpecies
    );
    existNumSpecies += countForFreeBoundaryParticlesSpeciesYUp[0];
}


