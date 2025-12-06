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

    if (PIC2DConst::existNumIon > PIC2DConst::totalNumIon) std::cout << "BROKEN" << std::endl;
    if (PIC2DConst::existNumElectron > PIC2DConst::totalNumElectron) std::cout << "BROKEN" << std::endl;
}


__global__ void periodicBoundaryParticle_x_kernel(
    Particle* particlesSpecies, 
    Particle* sendParticlesSpeciesXLeft, 
    Particle* sendParticlesSpeciesXRight, 
    unsigned long long* countForFreeBoundaryParticlesSpeciesXLeft, 
    unsigned long long* countForFreeBoundaryParticlesSpeciesXRight, 
    const unsigned long long existNumSpecies
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        double x = particlesSpecies[i].x; 
        
        double boundaryLeft  = PIC2DConst::device_xmin; 
        double boundaryRight = PIC2DConst::device_xmax;
        
        if (x <= boundaryLeft + PIC2DConst::device_dx) {
            particlesSpecies[i].isExist = false; 
        }
        if (x >= boundaryRight - PIC2DConst::device_dx) {
            particlesSpecies[i].isExist = false; 
        }
        
        if (x > boundaryLeft + PIC2DConst::device_dx && x <= boundaryLeft + 2 * PIC2DConst::device_dx) {
            unsigned long long particleIndex = atomicAdd(&(countForFreeBoundaryParticlesSpeciesXLeft[0]), 1);
            Particle sendParticle = particlesSpecies[i];
            sendParticle.x = sendParticle.x + PIC2DConst::device_xmax - 2 * PIC2DConst::device_dx; 
            sendParticlesSpeciesXLeft[particleIndex] = sendParticle;
        }

        if (x < boundaryRight - PIC2DConst::device_dx && x >= boundaryRight - 2 * PIC2DConst::device_dx) {
            unsigned long long particleIndex = atomicAdd(&(countForFreeBoundaryParticlesSpeciesXRight[0]), 1);
            Particle sendParticle = particlesSpecies[i];
            sendParticle.x = sendParticle.x - PIC2DConst::device_xmax + 2 * PIC2DConst::device_dy; 
            sendParticlesSpeciesXRight[particleIndex] = sendParticle;
        }
    }
}

// メモリ節約のため、YDown, YUpの配列を使う
void BoundaryPIC::periodicBoundaryParticleOfOneSpecies_x(
    thrust::device_vector<Particle>& particlesSpecies, 
    unsigned long long& existNumSpecies
)
{   
    auto& sendParticlesSpeciesXLeft = sendParticlesSpeciesYDown; 
    auto& sendParticlesSpeciesXRight = sendParticlesSpeciesYUp;

    thrust::device_vector<unsigned long long> countForPeriodicBoundaryParticlesSpeciesXLeft(1, 0); 
    thrust::device_vector<unsigned long long> countForPeriodicBoundaryParticlesSpeciesXRight(1, 0); 

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((existNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);

    periodicBoundaryParticle_x_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        thrust::raw_pointer_cast(sendParticlesSpeciesXLeft.data()), 
        thrust::raw_pointer_cast(sendParticlesSpeciesXRight.data()), 
        thrust::raw_pointer_cast(countForPeriodicBoundaryParticlesSpeciesXLeft.data()), 
        thrust::raw_pointer_cast(countForPeriodicBoundaryParticlesSpeciesXRight.data()), 
        existNumSpecies
    );
    cudaDeviceSynchronize();

    auto partitionEnd = thrust::partition(
        particlesSpecies.begin(), particlesSpecies.begin() + existNumSpecies, 
        [] __device__ (const Particle& p) { return p.isExist; }
    );
    existNumSpecies = static_cast<unsigned long long>(thrust::distance(particlesSpecies.begin(), partitionEnd));

    thrust::copy(
        sendParticlesSpeciesXLeft.begin(), 
        sendParticlesSpeciesXLeft.begin() + countForPeriodicBoundaryParticlesSpeciesXLeft[0],
        particlesSpecies.begin() + existNumSpecies
    );
    existNumSpecies += countForPeriodicBoundaryParticlesSpeciesXLeft[0];
    thrust::copy(
        sendParticlesSpeciesXRight.begin(), 
        sendParticlesSpeciesXRight.begin() + countForPeriodicBoundaryParticlesSpeciesXRight[0],
        particlesSpecies.begin() + existNumSpecies
    );
    existNumSpecies += countForPeriodicBoundaryParticlesSpeciesXRight[0];
}


