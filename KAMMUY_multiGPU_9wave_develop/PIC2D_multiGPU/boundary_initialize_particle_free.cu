#include "boundary.hpp"


void BoundaryPIC::freeBoundaryForInitializeParticle_x(
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron
)
{
    std::cout << "Not writtern yet. Finish your calculation now!" << std::endl;
}


void BoundaryPIC::freeBoundaryForInitializeParticle_y(
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron
)
{
    //periodic boundary function is used except for boundary
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

    //for boundary 
    freeBoundaryForInitializeParticleOfOneSpecies_y(
        particlesIon, 
        mPIInfo.existNumIonPerProcs
    ); 
    freeBoundaryForInitializeParticleOfOneSpecies_y(
        particlesElectron, 
        mPIInfo.existNumElectronPerProcs
    );
    MPI_Barrier(MPI_COMM_WORLD);
}



void BoundaryPIC::freeBoundaryForInitializeParticleOfOneSpecies_x(
    thrust::device_vector<Particle>& particlesSpecies, 
    unsigned long long& existNumSpecies
)
{
    std::cout << "Not writtern yet. Finish your calculation now!" << std::endl;
}


// particles in [ymin, ymin + dy] are copied to buffer region 
__global__ void freeBoundaryForInitialize_y_count_kernel(
    Particle* particlesSpecies, 
    unsigned int* countForAddParticlesSpeciesDown, 
    unsigned int* countForAddParticlesSpeciesUp, 
    const unsigned long long existNumSpecies, 
    const int buffer
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        if (PIC2DConst::device_ymin < particlesSpecies[i].y && particlesSpecies[i].y < PIC2DConst::device_ymin + PIC2DConst::device_dy) {
            atomicAdd(&(countForAddParticlesSpeciesDown[0]), buffer);
        }

        if (PIC2DConst::device_ymax - PIC2DConst::device_dy < particlesSpecies[i].y && particlesSpecies[i].y < PIC2DConst::device_ymax) {
            atomicAdd(&(countForAddParticlesSpeciesUp[0]), buffer);
        }
    }
}


__global__ void freeBoundaryForInitialize_y_kernel(
    Particle* particlesSpecies, 
    Particle* addParticlesSpeciesDown, 
    Particle* addParticlesSpeciesUp, 
    unsigned int* countForAddParticlesSpeciesDown, 
    unsigned int* countForAddParticlesSpeciesUp, 
    const unsigned long long existNumSpecies, 
    const int buffer
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        //delete particles which are added by periodicBoundaryForInitialize_y()
        if (particlesSpecies[i].y <= PIC2DConst::device_ymin) {
            particlesSpecies[i].isExist = false;
            return;
        }
        if (particlesSpecies[i].y >= PIC2DConst::device_ymax) {
            particlesSpecies[i].isExist = false;
            return;
        }

        if (PIC2DConst::device_ymin < particlesSpecies[i].y && particlesSpecies[i].y < PIC2DConst::device_ymin + PIC2DConst::device_dy) {
            unsigned long long particleIndex = atomicAdd(&(countForAddParticlesSpeciesDown[0]), buffer);
            Particle addParticle = particlesSpecies[i];
            for (int j = 0; j < buffer; j++) {
                addParticle.y = addParticle.y - PIC2DConst::device_dy;
                addParticlesSpeciesDown[particleIndex + j] = addParticle;
            }
        }

        if (PIC2DConst::device_ymax - PIC2DConst::device_dy < particlesSpecies[i].y && particlesSpecies[i].y < PIC2DConst::device_ymax) {
            unsigned long long particleIndex = atomicAdd(&(countForAddParticlesSpeciesUp[0]), buffer);
            Particle addParticle = particlesSpecies[i];
            for (int j = 0; j < buffer; j++) {
                addParticle.y = addParticle.y + PIC2DConst::device_dy;
                addParticlesSpeciesUp[particleIndex + j] = addParticle;
            }
        }
    }
}

void BoundaryPIC::freeBoundaryForInitializeParticleOfOneSpecies_y(
    thrust::device_vector<Particle>& particlesSpecies, 
    unsigned long long& existNumSpecies
)
{
    thrust::device_vector<unsigned int> countForAddParticlesSpeciesDown(1, 0); 
    thrust::device_vector<unsigned int> countForAddParticlesSpeciesUp(1, 0); 

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((existNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);

    freeBoundaryForInitialize_y_count_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        thrust::raw_pointer_cast(countForAddParticlesSpeciesDown.data()), 
        thrust::raw_pointer_cast(countForAddParticlesSpeciesUp.data()), 
        existNumSpecies, 
        mPIInfo.buffer
    );
    cudaDeviceSynchronize();

    unsigned int numForAddParticlesSpeciesDown = countForAddParticlesSpeciesDown[0];
    unsigned int numForAddParticlesSpeciesUp   = countForAddParticlesSpeciesUp[0];

    thrust::device_vector<Particle> addParticlesSpeciesDown(numForAddParticlesSpeciesDown);
    thrust::device_vector<Particle> addParticlesSpeciesUp(numForAddParticlesSpeciesUp);
    countForAddParticlesSpeciesDown[0] = 0;
    countForAddParticlesSpeciesUp[0] = 0;

    freeBoundaryForInitialize_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        thrust::raw_pointer_cast(addParticlesSpeciesDown.data()), 
        thrust::raw_pointer_cast(addParticlesSpeciesUp.data()), 
        thrust::raw_pointer_cast(countForAddParticlesSpeciesDown.data()), 
        thrust::raw_pointer_cast(countForAddParticlesSpeciesUp.data()), 
        existNumSpecies, 
        mPIInfo.buffer
    );
    cudaDeviceSynchronize();

    for (unsigned int i = 0; i < numForAddParticlesSpeciesDown; i++) {
        particlesSpecies[existNumSpecies + i] = addParticlesSpeciesDown[i];
    }
    existNumSpecies += numForAddParticlesSpeciesDown;
    for (unsigned int i = 0; i < numForAddParticlesSpeciesUp; i++) {
        particlesSpecies[existNumSpecies + i] = addParticlesSpeciesUp[i];
    }
    existNumSpecies += numForAddParticlesSpeciesUp;

    //Remove dead particles
    auto partitionEnd = thrust::partition(
        particlesSpecies.begin(), particlesSpecies.end(), 
        [] __device__ (const Particle& p) { return p.isExist; }
    );

    existNumSpecies = thrust::distance(particlesSpecies.begin(), partitionEnd);
}


