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


//send用のベクトルを再利用する
//Left->Down, Right->Upでメモリ節約
__global__ void freeBoundaryParticle_y_kernel(
    Particle* particlesSpecies, 
    Particle* sendParticlesSpeciesLeft, 
    Particle* sendParticlesSpeciesRight, 
    unsigned int* countForFreeBoundaryParticlesSpeciesLeft, 
    unsigned int* countForFreeBoundaryParticlesSpeciesRight, 
    const unsigned long long existNumSpecies, 
    const int buffer
)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        float y = particlesSpecies[i].y; 
        
        float boundaryDown  = PIC2DConst::device_ymin + PIC2DConst::device_EPS; 
        float boundaryUp    = PIC2DConst::device_ymax - PIC2DConst::device_EPS;
        
        if (y < boundaryDown + PIC2DConst::device_dy) {
            particlesSpecies[i].isExist = false; 
        }
        if (y > boundaryUp - PIC2DConst::device_dy) {
            particlesSpecies[i].isExist = false; 
        }
        
        if (y > boundaryDown + PIC2DConst::device_dy && y < boundaryDown + 2 * PIC2DConst::device_dy) {
            unsigned int particleIndex = atomicAdd(&(countForFreeBoundaryParticlesSpeciesLeft[0]), 1);
            Particle sendParticle = particlesSpecies[i];
            sendParticle.y = sendParticle.y - PIC2DConst::device_dy; 
            sendParticlesSpeciesLeft[particleIndex] = sendParticle;
        }

        if (y < boundaryUp - PIC2DConst::device_dy && y > boundaryUp - 2 * PIC2DConst::device_dy) {
            unsigned int particleIndex = atomicAdd(&(countForFreeBoundaryParticlesSpeciesRight[0]), 1);
            Particle sendParticle = particlesSpecies[i];
            sendParticle.y = sendParticle.y + PIC2DConst::device_dy; 
            sendParticlesSpeciesRight[particleIndex] = sendParticle;
        }
    }
}


//send用のベクトルを再利用する
//Left->Down, Right->Upでメモリ節約
void BoundaryPIC::freeBoundaryParticleOfOneSpecies_y(
    thrust::device_vector<Particle>& particlesSpecies, 
    unsigned long long& existNumSpecies
)
{   
    thrust::device_vector<unsigned int> countForFreeBoundaryParticlesSpeciesLeft(1, 0); 
    thrust::device_vector<unsigned int> countForFreeBoundaryParticlesSpeciesRight(1, 0); 

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((existNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);

    freeBoundaryParticle_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        thrust::raw_pointer_cast(sendParticlesSpeciesLeft.data()), 
        thrust::raw_pointer_cast(sendParticlesSpeciesRight.data()), 
        thrust::raw_pointer_cast(countForFreeBoundaryParticlesSpeciesLeft.data()), 
        thrust::raw_pointer_cast(countForFreeBoundaryParticlesSpeciesRight.data()), 
        existNumSpecies, 
        mPIInfo.buffer
    );
    cudaDeviceSynchronize();

    auto partitionEnd = thrust::partition(
        particlesSpecies.begin(), particlesSpecies.end(), 
        [] __device__ (const Particle& p) { return p.isExist; }
    );
    existNumSpecies = thrust::distance(particlesSpecies.begin(), partitionEnd);

    //sendはしないので注意。再利用しているだけ。

    thrust::copy(
        sendParticlesSpeciesLeft.begin(), 
        sendParticlesSpeciesLeft.begin() + countForFreeBoundaryParticlesSpeciesLeft[0],
        particlesSpecies.begin() + existNumSpecies
    );
    existNumSpecies += countForFreeBoundaryParticlesSpeciesLeft[0];
    thrust::copy(
        sendParticlesSpeciesRight.begin(), 
        sendParticlesSpeciesRight.begin() + countForFreeBoundaryParticlesSpeciesRight[0],
        particlesSpecies.begin() + existNumSpecies
    );
    existNumSpecies += countForFreeBoundaryParticlesSpeciesRight[0];
}


