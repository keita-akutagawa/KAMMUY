#include "boundary.hpp"



void BoundaryPIC::freeBoundaryParticle_y(
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron
)
{   
    MPI_Barrier(MPI_COMM_WORLD); 
    freeBoundaryParticleOfOneSpecies_y(
        particlesIon,
        mPIInfo.existNumIonPerProcs, 
        mPIInfo.numForSendParticlesIonDown, 
        mPIInfo.numForSendParticlesIonUp, 
        mPIInfo.numForRecvParticlesIonDown, 
        mPIInfo.numForRecvParticlesIonUp
    );
    MPI_Barrier(MPI_COMM_WORLD); 
    freeBoundaryParticleOfOneSpecies_y(
        particlesElectron, 
        mPIInfo.existNumElectronPerProcs, 
        mPIInfo.numForSendParticlesElectronDown, 
        mPIInfo.numForSendParticlesElectronUp, 
        mPIInfo.numForRecvParticlesElectronDown, 
        mPIInfo.numForRecvParticlesElectronUp
    );
    MPI_Barrier(MPI_COMM_WORLD);
}


__global__ void freeBoundaryParticle_y_kernel(
    Particle* particlesSpecies, 
    Particle* sendParticlesSpeciesDown, 
    Particle* sendParticlesSpeciesUp, 
    unsigned int* countForSendParticlesSpeciesDown, 
    unsigned int* countForSendParticlesSpeciesUp, 
    const unsigned long long existNumSpecies, 
    const float yminForProcs, const float ymaxForProcs, 
    const int buffer
)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        if (particlesSpecies[i].isMPISendUp) {
            unsigned int particleIndex = atomicAdd(&(countForSendParticlesSpeciesUp[0]), 1);
            particlesSpecies[i].isMPISendUp = false;
            Particle sendParticle = particlesSpecies[i];
            if (sendParticle.y > PIC2DConst::device_ymax - buffer * PIC2DConst::device_dy) {
                sendParticle.isExist = false;
            }
            sendParticlesSpeciesUp[particleIndex] = sendParticle;
        }

        if (particlesSpecies[i].isMPISendDown) {
            unsigned int particleIndex = atomicAdd(&(countForSendParticlesSpeciesDown[0]), 1);
            particlesSpecies[i].isMPISendDown = false;
            Particle sendParticle = particlesSpecies[i];
            if (sendParticle.y < PIC2DConst::device_ymin + buffer * PIC2DConst::device_dy) {
                sendParticle.isExist = false; 
            }
            sendParticlesSpeciesDown[particleIndex] = sendParticle;
        }
    }
}


void BoundaryPIC::freeBoundaryParticleOfOneSpecies_y(
    thrust::device_vector<Particle>& particlesSpecies, 
    unsigned long long& existNumSpecies, 
    unsigned int& numForSendParticlesSpeciesDown, 
    unsigned int& numForSendParticlesSpeciesUp, 
    unsigned int& numForRecvParticlesSpeciesDown, 
    unsigned int& numForRecvParticlesSpeciesUp
)
{   
    thrust::device_vector<unsigned int> countForSendParticlesSpeciesDown(1, 0); 
    thrust::device_vector<unsigned int> countForSendParticlesSpeciesUp(1, 0); 

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((existNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);

    freeBoundaryParticle_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        thrust::raw_pointer_cast(sendParticlesSpeciesDown.data()), 
        thrust::raw_pointer_cast(sendParticlesSpeciesUp.data()), 
        thrust::raw_pointer_cast(countForSendParticlesSpeciesDown.data()), 
        thrust::raw_pointer_cast(countForSendParticlesSpeciesUp.data()), 
        existNumSpecies, 
        mPIInfo.yminForProcs, mPIInfo.ymaxForProcs, 
        mPIInfo.buffer
    );
    cudaDeviceSynchronize();

    PIC2DMPI::sendrecv_numParticle_y(
        numForSendParticlesSpeciesDown, 
        numForSendParticlesSpeciesUp, 
        numForRecvParticlesSpeciesDown, 
        numForRecvParticlesSpeciesUp, 
        mPIInfo
    );

    PIC2DMPI::sendrecv_particle_y(
        sendParticlesSpeciesDown, 
        sendParticlesSpeciesUp,  
        recvParticlesSpeciesDown, 
        recvParticlesSpeciesUp,  
        numForSendParticlesSpeciesDown, 
        numForSendParticlesSpeciesUp, 
        numForRecvParticlesSpeciesDown, 
        numForRecvParticlesSpeciesUp, 
        mPIInfo
    );

    thrust::copy(
        recvParticlesSpeciesDown.begin(), 
        recvParticlesSpeciesDown.begin() + numForRecvParticlesSpeciesDown,
        particlesSpecies.begin() + existNumSpecies
    );
    existNumSpecies += numForRecvParticlesSpeciesDown;
    thrust::copy(
        recvParticlesSpeciesUp.begin(), 
        recvParticlesSpeciesUp.begin() + numForRecvParticlesSpeciesUp,
        particlesSpecies.begin() + existNumSpecies
    );
    existNumSpecies += numForRecvParticlesSpeciesUp;
}


