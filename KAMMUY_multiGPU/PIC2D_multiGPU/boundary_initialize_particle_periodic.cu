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


void BoundaryPIC::periodicBoundaryForInitializeParticle_y(
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron
)
{
    periodicBoundaryForInitializeParticleOfOneSpecies_y(
        particlesIon, 
        mPIInfo.existNumIonPerProcs, 
        mPIInfo.numForSendParticlesIonDown, 
        mPIInfo.numForSendParticlesIonUp, 
        mPIInfo.numForRecvParticlesIonDown, 
        mPIInfo.numForRecvParticlesIonUp
    ); 
    periodicBoundaryForInitializeParticleOfOneSpecies_y(
        particlesElectron, 
        mPIInfo.existNumElectronPerProcs,  
        mPIInfo.numForSendParticlesElectronDown, 
        mPIInfo.numForSendParticlesElectronUp, 
        mPIInfo.numForRecvParticlesElectronDown, 
        mPIInfo.numForRecvParticlesElectronUp
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

    thrust::device_vector<Particle> sendParticlesSpeciesLeft(numForSendParticlesSpeciesLeft);
    thrust::device_vector<Particle> sendParticlesSpeciesRight(numForSendParticlesSpeciesRight);
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

    thrust::host_vector<Particle> host_sendParticlesSpeciesLeft(numForSendParticlesSpeciesLeft);
    thrust::host_vector<Particle> host_sendParticlesSpeciesRight(numForSendParticlesSpeciesRight);
    host_sendParticlesSpeciesLeft = sendParticlesSpeciesLeft;
    host_sendParticlesSpeciesRight = sendParticlesSpeciesRight;

    sendrecv_numParticle_x(
        numForSendParticlesSpeciesLeft, 
        numForSendParticlesSpeciesRight, 
        numForRecvParticlesSpeciesLeft, 
        numForRecvParticlesSpeciesRight, 
        mPIInfo
    );

    thrust::host_vector<Particle> host_recvParticlesSpeciesLeft(numForRecvParticlesSpeciesLeft);
    thrust::host_vector<Particle> host_recvParticlesSpeciesRight(numForRecvParticlesSpeciesRight);

    sendrecv_particle_x(
        host_sendParticlesSpeciesLeft, 
        host_sendParticlesSpeciesRight,  
        host_recvParticlesSpeciesLeft, 
        host_recvParticlesSpeciesRight,  
        mPIInfo
    );

    for (unsigned int i = 0; i < numForRecvParticlesSpeciesLeft; i++) {
        particlesSpecies[existNumSpecies + i] = host_recvParticlesSpeciesLeft[i];
    }
    existNumSpecies += numForRecvParticlesSpeciesLeft;
    for (unsigned int i = 0; i < numForRecvParticlesSpeciesRight; i++) {
        particlesSpecies[existNumSpecies + i] = host_recvParticlesSpeciesRight[i];
    }
    existNumSpecies += numForRecvParticlesSpeciesRight;
}


__global__ void periodicBoundaryForInitialize_y_count_kernel(
    Particle* particlesSpecies, 
    unsigned int* countForSendParticlesSpeciesDown, 
    unsigned int* countForSendParticlesSpeciesUp, 
    const unsigned long long existNumSpecies, 
    const float yminForProcs, const float ymaxForProcs, 
    const int buffer
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        if (ymaxForProcs - buffer * PIC2DConst::device_dy < particlesSpecies[i].y && particlesSpecies[i].y < ymaxForProcs) {
            atomicAdd(&(countForSendParticlesSpeciesUp[0]), 1);
        }

        if (yminForProcs < particlesSpecies[i].y && particlesSpecies[i].y < yminForProcs + buffer * PIC2DConst::device_dy) {
            atomicAdd(&(countForSendParticlesSpeciesDown[0]), 1);
        }
    }
}


__global__ void periodicBoundaryForInitialize_y_kernel(
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
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        if (yminForProcs < particlesSpecies[i].y && particlesSpecies[i].y < yminForProcs + buffer * PIC2DConst::device_dy) {
            unsigned long long particleIndex = atomicAdd(&(countForSendParticlesSpeciesDown[0]), 1);
            Particle sendParticle = particlesSpecies[i];
            if (sendParticle.y < PIC2DConst::device_ymin + buffer * PIC2DConst::device_dy) {
                sendParticle.y = sendParticle.y + PIC2DConst::device_ymax - PIC2DConst::device_EPS;
            }
            sendParticlesSpeciesDown[particleIndex] = sendParticle;
        }

        if (ymaxForProcs - buffer * PIC2DConst::device_dy < particlesSpecies[i].y && particlesSpecies[i].y < ymaxForProcs) {
            unsigned long long particleIndex = atomicAdd(&(countForSendParticlesSpeciesUp[0]), 1);
            Particle sendParticle = particlesSpecies[i];
            if (PIC2DConst::device_ymax - buffer * PIC2DConst::device_dy < sendParticle.y) {
                sendParticle.y = sendParticle.y - PIC2DConst::device_ymax + PIC2DConst::device_EPS;
            }
            sendParticlesSpeciesUp[particleIndex] = sendParticle;
        }
    }
}

void BoundaryPIC::periodicBoundaryForInitializeParticleOfOneSpecies_y(
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

    periodicBoundaryForInitialize_y_count_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        thrust::raw_pointer_cast(countForSendParticlesSpeciesDown.data()), 
        thrust::raw_pointer_cast(countForSendParticlesSpeciesUp.data()), 
        existNumSpecies, 
        mPIInfo.yminForProcs, mPIInfo.ymaxForProcs, 
        mPIInfo.buffer
    );
    cudaDeviceSynchronize();

    numForSendParticlesSpeciesDown = countForSendParticlesSpeciesDown[0];
    numForSendParticlesSpeciesUp = countForSendParticlesSpeciesUp[0];

    thrust::device_vector<Particle> sendParticlesSpeciesDown(numForSendParticlesSpeciesDown);
    thrust::device_vector<Particle> sendParticlesSpeciesUp(numForSendParticlesSpeciesUp);
    countForSendParticlesSpeciesDown[0] = 0;
    countForSendParticlesSpeciesUp[0] = 0;

    periodicBoundaryForInitialize_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(
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

    thrust::host_vector<Particle> host_sendParticlesSpeciesDown(numForSendParticlesSpeciesDown);
    thrust::host_vector<Particle> host_sendParticlesSpeciesUp(numForSendParticlesSpeciesUp);
    host_sendParticlesSpeciesDown = sendParticlesSpeciesDown;
    host_sendParticlesSpeciesUp = sendParticlesSpeciesUp;

    sendrecv_numParticle_y(
        numForSendParticlesSpeciesDown, 
        numForSendParticlesSpeciesUp, 
        numForRecvParticlesSpeciesDown, 
        numForRecvParticlesSpeciesUp, 
        mPIInfo
    );

    thrust::host_vector<Particle> host_recvParticlesSpeciesDown(numForRecvParticlesSpeciesDown);
    thrust::host_vector<Particle> host_recvParticlesSpeciesUp(numForRecvParticlesSpeciesUp);
    sendrecv_particle_y(
        host_sendParticlesSpeciesDown, 
        host_sendParticlesSpeciesUp,  
        host_recvParticlesSpeciesDown, 
        host_recvParticlesSpeciesUp,  
        mPIInfo
    );

    for (unsigned int i = 0; i < numForRecvParticlesSpeciesDown; i++) {
        particlesSpecies[existNumSpecies + i] = host_recvParticlesSpeciesDown[i];
    }
    existNumSpecies += numForRecvParticlesSpeciesDown;
    for (unsigned int i = 0; i < numForRecvParticlesSpeciesUp; i++) {
        particlesSpecies[existNumSpecies + i] = host_recvParticlesSpeciesUp[i];
    }
    existNumSpecies += numForRecvParticlesSpeciesUp;
}


