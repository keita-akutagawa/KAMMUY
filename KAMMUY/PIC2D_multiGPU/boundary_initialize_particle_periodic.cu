#include "boundary.hpp"


void BoundaryPIC::periodicBoundaryForInitializeParticle_x(
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron
)
{
    periodicBoundaryForInitializeParticleOfOneSpecies_x(
        particlesIon, 
        mPIInfo.existNumIonPerProcs, 
        mPIInfo.numForSendParticlesIonLeft, 
        mPIInfo.numForSendParticlesIonRight, 
        mPIInfo.numForRecvParticlesIonLeft, 
        mPIInfo.numForRecvParticlesIonRight
    ); 
    periodicBoundaryForInitializeParticleOfOneSpecies_x(
        particlesElectron, 
        mPIInfo.existNumElectronPerProcs,  
        mPIInfo.numForSendParticlesElectronLeft, 
        mPIInfo.numForSendParticlesElectronRight, 
        mPIInfo.numForRecvParticlesElectronLeft, 
        mPIInfo.numForRecvParticlesElectronRight
    ); 
    MPI_Barrier(MPI_COMM_WORLD);
}


__global__ void periodicBoundaryForInitialize_x_count_kernel(
    Particle* particlesSpecies, 
    unsigned int* countForSendParticlesSpeciesLeft, 
    unsigned int* countForSendParticlesSpeciesRight, 
    const unsigned long long existNumSpecies, 
    const float xminForProcs, const float xmaxForProcs, 
    const int buffer
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        if (xmaxForProcs - buffer * PIC2DConst::device_dx < particlesSpecies[i].x && particlesSpecies[i].x < xmaxForProcs) {
            atomicAdd(&(countForSendParticlesSpeciesRight[0]), 1);
        }

        if (xminForProcs < particlesSpecies[i].x && particlesSpecies[i].x < xminForProcs + buffer * PIC2DConst::device_dx) {
            atomicAdd(&(countForSendParticlesSpeciesLeft[0]), 1);
        }
    }
}


__global__ void periodicBoundaryForInitialize_x_kernel(
    Particle* particlesSpecies, 
    Particle* sendParticlesSpeciesLeft, 
    Particle* sendParticlesSpeciesRight, 
    unsigned int* countForSendParticlesSpeciesLeft, 
    unsigned int* countForSendParticlesSpeciesRight, 
    const unsigned long long existNumSpecies, 
    const float xminForProcs, const float xmaxForProcs, 
    const int buffer
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        if (xminForProcs < particlesSpecies[i].x && particlesSpecies[i].x < xminForProcs + buffer * PIC2DConst::device_dx) {
            unsigned long long particleIndex = atomicAdd(&(countForSendParticlesSpeciesLeft[0]), 1);
            Particle sendParticle = particlesSpecies[i];
            if (sendParticle.x < PIC2DConst::device_xmin + buffer * PIC2DConst::device_dx) {
                sendParticle.x = sendParticle.x + PIC2DConst::device_xmax - PIC2DConst::device_EPS;
            }
            sendParticlesSpeciesLeft[particleIndex] = sendParticle;
        }

        if (xmaxForProcs - buffer * PIC2DConst::device_dx < particlesSpecies[i].x && particlesSpecies[i].x < xmaxForProcs) {
            unsigned long long particleIndex = atomicAdd(&(countForSendParticlesSpeciesRight[0]), 1);
            Particle sendParticle = particlesSpecies[i];
            if (PIC2DConst::device_xmax - buffer * PIC2DConst::device_dx < sendParticle.x) {
                sendParticle.x = sendParticle.x - PIC2DConst::device_xmax + PIC2DConst::device_EPS;
            }
            sendParticlesSpeciesRight[particleIndex] = sendParticle;
        }
    }
}

void BoundaryPIC::periodicBoundaryForInitializeParticleOfOneSpecies_x(
    thrust::device_vector<Particle>& particlesSpecies, 
    unsigned long long& existNumSpecies, 
    unsigned int& numForSendParticlesSpeciesLeft, 
    unsigned int& numForSendParticlesSpeciesRight, 
    unsigned int& numForRecvParticlesSpeciesLeft, 
    unsigned int& numForRecvParticlesSpeciesRight
)
{
    thrust::device_vector<unsigned int> countForSendParticlesSpeciesLeft(1, 0); 
    thrust::device_vector<unsigned int> countForSendParticlesSpeciesRight(1, 0); 

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((existNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);

    periodicBoundaryForInitialize_x_count_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        thrust::raw_pointer_cast(countForSendParticlesSpeciesLeft.data()), 
        thrust::raw_pointer_cast(countForSendParticlesSpeciesRight.data()), 
        existNumSpecies, 
        mPIInfo.xminForProcs, mPIInfo.xmaxForProcs, 
        mPIInfo.buffer
    );
    cudaDeviceSynchronize();

    numForSendParticlesSpeciesLeft = countForSendParticlesSpeciesLeft[0];
    numForSendParticlesSpeciesRight = countForSendParticlesSpeciesRight[0];

    countForSendParticlesSpeciesLeft[0] = 0;
    countForSendParticlesSpeciesRight[0] = 0;

    periodicBoundaryForInitialize_x_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        thrust::raw_pointer_cast(sendParticlesSpeciesLeft.data()), 
        thrust::raw_pointer_cast(sendParticlesSpeciesRight.data()), 
        thrust::raw_pointer_cast(countForSendParticlesSpeciesLeft.data()), 
        thrust::raw_pointer_cast(countForSendParticlesSpeciesRight.data()), 
        existNumSpecies, 
        mPIInfo.xminForProcs, mPIInfo.xmaxForProcs, 
        mPIInfo.buffer
    );
    cudaDeviceSynchronize();

    PIC2DMPI::sendrecv_numParticle_x(
        numForSendParticlesSpeciesLeft, 
        numForSendParticlesSpeciesRight, 
        numForRecvParticlesSpeciesLeft, 
        numForRecvParticlesSpeciesRight, 
        mPIInfo
    );

    PIC2DMPI::sendrecv_particle_x(
        sendParticlesSpeciesLeft, 
        sendParticlesSpeciesRight,  
        recvParticlesSpeciesLeft, 
        recvParticlesSpeciesRight,  
        numForSendParticlesSpeciesLeft, 
        numForSendParticlesSpeciesRight, 
        numForRecvParticlesSpeciesLeft, 
        numForRecvParticlesSpeciesRight, 
        mPIInfo
    );

    thrust::copy(
        recvParticlesSpeciesLeft.begin(), 
        recvParticlesSpeciesLeft.begin() + numForRecvParticlesSpeciesLeft,
        particlesSpecies.begin() + existNumSpecies
    );
    existNumSpecies += numForRecvParticlesSpeciesLeft;
    thrust::copy(
        recvParticlesSpeciesRight.begin(), 
        recvParticlesSpeciesRight.begin() + numForRecvParticlesSpeciesRight,
        particlesSpecies.begin() + existNumSpecies
    );
    existNumSpecies += numForRecvParticlesSpeciesRight;
}

