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
        curand_init(seed + 0, i, 0, &stateX);
        curand_init(seed + 1, i, 0, &stateY);
        curand_init(seed + 2, i, 0, &stateVx);
        curand_init(seed + 3, i, 0, &stateVy);
        curand_init(seed + 4, i, 0, &stateVz);

        reloadParticlesSourceSpecies[i].x  = curand_uniform_double(&stateX);
        reloadParticlesSourceSpecies[i].y  = curand_uniform_double(&stateY);
        reloadParticlesSourceSpecies[i].z  = 0.0;
        reloadParticlesSourceSpecies[i].vx = curand_normal_double(&stateVx);
        reloadParticlesSourceSpecies[i].vy = curand_normal_double(&stateVy);
        reloadParticlesSourceSpecies[i].vz = curand_normal_double(&stateVz);
        reloadParticlesSourceSpecies[i].gamma = 0.0;
        reloadParticlesSourceSpecies[i].isExist = false;
    }
}

Interface2D::Interface2D(
    PIC2DMPI::MPIInfo& mPIInfoPIC, 
    IdealMHD2DMPI::MPIInfo& mPIInfoMHD, 
    int indexStartMHD, 
    int indexStartPIC, 
    int length, 
    thrust::host_vector<double>& host_interlockingFunctionY, 
    thrust::host_vector<double>& host_interlockingFunctionYHalf, 
    InterfaceNoiseRemover2D& interfaceNoiseRemover2D_Lower, 
    InterfaceNoiseRemover2D& interfaceNoiseRemover2D_Upper
)
    : mPIInfoPIC(mPIInfoPIC), 
      mPIInfoMHD(mPIInfoMHD), 

      indexOfInterfaceStartInMHD(indexStartMHD), 
      indexOfInterfaceStartInPIC(indexStartPIC), 
      interfaceLength(length), 
      indexOfInterfaceEndInMHD(indexStartMHD + length), 
      indexOfInterfaceEndInPIC(indexStartPIC + length), 

      interlockingFunctionY    (interfaceLength, 0.0), 
      interlockingFunctionYHalf(interfaceLength, 0.0),

      zerothMomentIon     (mPIInfoPIC.localSizeX * mPIInfoPIC.localSizeY), 
      zerothMomentElectron(mPIInfoPIC.localSizeX * mPIInfoPIC.localSizeY), 
      firstMomentIon      (mPIInfoPIC.localSizeX * mPIInfoPIC.localSizeY), 
      firstMomentElectron (mPIInfoPIC.localSizeX * mPIInfoPIC.localSizeY),

      restartParticlesIndexIon(0), 
      restartParticlesIndexElectron(0), 

      reloadParticlesSourceIon     (Interface2DConst::reloadParticlesTotalNumIon), 
      reloadParticlesSourceElectron(Interface2DConst::reloadParticlesTotalNumElectron), 

      reloadParticlesDataIon            (mPIInfoPIC.localSizeX * interfaceLength + 1), 
      reloadParticlesDataElectron       (mPIInfoPIC.localSizeX * interfaceLength + 1), 
      host_reloadParticlesDataIon       (mPIInfoPIC.localSizeX * interfaceLength + 1), 
      host_reloadParticlesDataElectron  (mPIInfoPIC.localSizeX * interfaceLength + 1), 

      B_timeAve                   (mPIInfoPIC.localSizeX * mPIInfoPIC.localSizeY), 
      zerothMomentIon_timeAve     (mPIInfoPIC.localSizeX * mPIInfoPIC.localSizeY), 
      zerothMomentElectron_timeAve(mPIInfoPIC.localSizeX * mPIInfoPIC.localSizeY), 
      firstMomentIon_timeAve      (mPIInfoPIC.localSizeX * mPIInfoPIC.localSizeY), 
      firstMomentElectron_timeAve (mPIInfoPIC.localSizeX * mPIInfoPIC.localSizeY), 

      USub (mPIInfoMHD.localSizeX * mPIInfoMHD.localSizeY), 
      UHalf(mPIInfoMHD.localSizeX * mPIInfoMHD.localSizeY), 

      interfaceNoiseRemover2D_Lower(interfaceNoiseRemover2D_Lower), 
      interfaceNoiseRemover2D_Upper(interfaceNoiseRemover2D_Upper)
{

    cudaMalloc(&device_mPIInfoPIC, sizeof(PIC2DMPI::MPIInfo));
    cudaMemcpy(device_mPIInfoPIC, &mPIInfoPIC, sizeof(PIC2DMPI::MPIInfo), cudaMemcpyHostToDevice);
    cudaMalloc(&device_mPIInfoMHD, sizeof(IdealMHD2DMPI::MPIInfo));
    cudaMemcpy(device_mPIInfoMHD, &mPIInfoMHD, sizeof(IdealMHD2DMPI::MPIInfo), cudaMemcpyHostToDevice);

    interlockingFunctionY = host_interlockingFunctionY;
    interlockingFunctionYHalf = host_interlockingFunctionYHalf;

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((Interface2DConst::reloadParticlesTotalNum + threadsPerBlockForIon.x - 1) / threadsPerBlockForIon.x);

    initializeReloadParticlesSource_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(reloadParticlesSourceIon.data()),
        Interface2DConst::reloadParticlesTotalNumIon, 
        100000
    );
    cudaDeviceSynchronize();

    initializeReloadParticlesSource_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(reloadParticlesSourceElectron.data()),
        Interface2DConst::reloadParticlesTotalNumElectron, 
        200000
    );
    cudaDeviceSynchronize();
}

