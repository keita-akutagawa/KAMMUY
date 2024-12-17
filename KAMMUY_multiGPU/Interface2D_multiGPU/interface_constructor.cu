#include "interface.hpp"


__global__ void initializeReloadParticlesSource_kernel(
    Particle* reloadParticlesSourceSpecies, 
    unsigned long long reloadParticlesNumSpecies, 
    int seed
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < reloadParticlesNumSpecies) {
        curandState stateX; 
        curandState stateY;
        curandState stateVx; 
        curandState stateVy; 
        curandState stateVz;  
        curand_init(seed + 0, 100 * i, 0, &stateX);
        curand_init(seed + 1, 100 * i, 0, &stateY);
        curand_init(seed + 2, 100 * i, 0, &stateVx);
        curand_init(seed + 3, 100 * i, 0, &stateVy);
        curand_init(seed + 4, 100 * i, 0, &stateVz);

        float x, y, z, vx, vy, vz;
        float EPS = 0.001f;
        while (true) {
            x  = curand_uniform(&stateX);
            y  = curand_uniform(&stateY);
            z  = 0.0f;

            if (EPS < x && x < 1.0f - EPS && EPS < y && y < 1.0f - EPS) break;
        }
        
        vx = curand_normal(&stateVx);
        vy = curand_normal(&stateVy);
        vz = curand_normal(&stateVz);

        reloadParticlesSourceSpecies[i].x  = x;
        reloadParticlesSourceSpecies[i].y  = y;
        reloadParticlesSourceSpecies[i].z  = z;
        reloadParticlesSourceSpecies[i].vx = vx;
        reloadParticlesSourceSpecies[i].vy = vy;
        reloadParticlesSourceSpecies[i].vz = vz;
    }
}

Interface2D::Interface2D(
    IdealMHD2DMPI::MPIInfo& mPIInfoMHD, 
    PIC2DMPI::MPIInfo& mPIInfoPIC, 
    Interface2DMPI::MPIInfo& mPIInfoInterface, 
    int indexOfInterfaceStartMHD, 
    int indexOfInterfaceStartPIC, 
    thrust::host_vector<double>& host_interlockingFunctionY, 
    thrust::host_vector<double>& host_interlockingFunctionYHalf, 
    InterfaceNoiseRemover2D& interfaceNoiseRemover2D
)
    : mPIInfoMHD(mPIInfoMHD), 
      mPIInfoPIC(mPIInfoPIC), 
      mPIInfoInterface(mPIInfoInterface), 

      indexOfInterfaceStartInMHD(indexOfInterfaceStartMHD), 
      indexOfInterfaceStartInPIC(indexOfInterfaceStartPIC),

      localSizeXPIC(mPIInfoPIC.localSizeX), 
      localSizeYPIC(mPIInfoPIC.localSizeY), 
      localSizeXMHD(mPIInfoMHD.localSizeX), 
      localSizeYMHD(mPIInfoMHD.localSizeY), 
      localSizeXInterface(mPIInfoInterface.localSizeX), 
      localSizeYInterface(mPIInfoInterface.localSizeY), 

      interlockingFunctionY    (localSizeYInterface, 0.0), 
      interlockingFunctionYHalf(localSizeYInterface, 0.0),

      zerothMomentIon     (localSizeXPIC * localSizeYPIC), 
      zerothMomentElectron(localSizeXPIC * localSizeYPIC), 
      firstMomentIon      (localSizeXPIC * localSizeYPIC), 
      firstMomentElectron (localSizeXPIC * localSizeYPIC),

      restartParticlesIndexIon(0), 
      restartParticlesIndexElectron(0), 

      reloadParticlesSourceIon     (Interface2DConst::reloadParticlesTotalNum), 
      reloadParticlesSourceElectron(Interface2DConst::reloadParticlesTotalNum), 

      reloadParticlesDataIon     (localSizeXInterface * localSizeYInterface), 
      reloadParticlesDataElectron(localSizeXInterface * localSizeYInterface), 
      
      B_timeAve                   (localSizeXPIC * localSizeYPIC), 
      zerothMomentIon_timeAve     (localSizeXPIC * localSizeYPIC), 
      zerothMomentElectron_timeAve(localSizeXPIC * localSizeYPIC), 
      firstMomentIon_timeAve      (localSizeXPIC * localSizeYPIC), 
      firstMomentElectron_timeAve (localSizeXPIC * localSizeYPIC), 

      USub (localSizeXMHD * localSizeYMHD), 
      UHalf(localSizeXMHD * localSizeYMHD), 

      momentCalculater(mPIInfoPIC), 
      interfaceNoiseRemover2D(interfaceNoiseRemover2D)
{

    cudaMalloc(&device_mPIInfoMHD, sizeof(IdealMHD2DMPI::MPIInfo));
    cudaMemcpy(device_mPIInfoMHD, &mPIInfoMHD, sizeof(IdealMHD2DMPI::MPIInfo), cudaMemcpyHostToDevice);
    cudaMalloc(&device_mPIInfoPIC, sizeof(PIC2DMPI::MPIInfo));
    cudaMemcpy(device_mPIInfoPIC, &mPIInfoPIC, sizeof(PIC2DMPI::MPIInfo), cudaMemcpyHostToDevice);
    cudaMalloc(&device_mPIInfoInterface, sizeof(Interface2DMPI::MPIInfo));
    cudaMemcpy(device_mPIInfoInterface, &mPIInfoInterface, sizeof(Interface2DMPI::MPIInfo), cudaMemcpyHostToDevice);

    interlockingFunctionY = host_interlockingFunctionY;
    interlockingFunctionYHalf = host_interlockingFunctionYHalf;

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((Interface2DConst::reloadParticlesTotalNum + threadsPerBlock.x - 1) / threadsPerBlock.x);

    initializeReloadParticlesSource_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(reloadParticlesSourceIon.data()),
        Interface2DConst::reloadParticlesTotalNum, 
        10000000 + 100 * mPIInfoPIC.rank
    );
    cudaDeviceSynchronize();

    initializeReloadParticlesSource_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(reloadParticlesSourceElectron.data()),
        Interface2DConst::reloadParticlesTotalNum, 
        20000000 + 100 * mPIInfoPIC.rank
    );
    cudaDeviceSynchronize();
}

