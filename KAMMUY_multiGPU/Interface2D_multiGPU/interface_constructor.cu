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

        float x, y, z, vx, vy, vz, gamma;
        x = curand_uniform_double(&stateX);
        y = curand_uniform_double(&stateY);
        z = 0.0f;
        vx = curand_uniform_double(&stateVx);
        vy = curand_uniform_double(&stateVy);
        vz = curand_uniform_double(&stateVz);
        gamma = 1.0f / sqrt(1.0f - (vx * vx + vy * vy + vz * vz) / pow(PIC2DConst::device_c, 2));

        reloadParticlesSourceSpecies[i].x       = x;
        reloadParticlesSourceSpecies[i].y       = y;
        reloadParticlesSourceSpecies[i].z       = z;
        reloadParticlesSourceSpecies[i].vx      = vx * gamma;
        reloadParticlesSourceSpecies[i].vy      = vy * gamma;
        reloadParticlesSourceSpecies[i].vz      = vz * gamma;
        reloadParticlesSourceSpecies[i].gamma   = gamma;
        reloadParticlesSourceSpecies[i].isExist = false;
    }
}

Interface2D::Interface2D(
    IdealMHD2DMPI::MPIInfo& mPIInfoMHD, 
    PIC2DMPI::MPIInfo& mPIInfoPIC, 
    bool isLower, bool isUpper, 
    int indexStartMHD, 
    int indexStartPIC, 
    int length, 
    thrust::host_vector<double>& host_interlockingFunctionY, 
    thrust::host_vector<double>& host_interlockingFunctionYHalf, 
    InterfaceNoiseRemover2D& interfaceNoiseRemover2D
)
    : mPIInfoMHD(mPIInfoMHD), 
      mPIInfoPIC(mPIInfoPIC), 

      isLower(isLower), 
      isUpper(isUpper), 

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

      reloadParticlesSourceIon     (Interface2DConst::reloadParticlesTotalNum), 
      reloadParticlesSourceElectron(Interface2DConst::reloadParticlesTotalNum), 

      reloadParticlesDataIon            (mPIInfoPIC.localNx * interfaceLength + 1), 
      reloadParticlesDataElectron       (mPIInfoPIC.localNx * interfaceLength + 1), 
      host_reloadParticlesDataIon       (mPIInfoPIC.localNx * interfaceLength + 1), 
      host_reloadParticlesDataElectron  (mPIInfoPIC.localNx * interfaceLength + 1), 

      B_timeAve                   (mPIInfoPIC.localSizeX * mPIInfoPIC.localSizeY), 
      zerothMomentIon_timeAve     (mPIInfoPIC.localSizeX * mPIInfoPIC.localSizeY), 
      zerothMomentElectron_timeAve(mPIInfoPIC.localSizeX * mPIInfoPIC.localSizeY), 
      firstMomentIon_timeAve      (mPIInfoPIC.localSizeX * mPIInfoPIC.localSizeY), 
      firstMomentElectron_timeAve (mPIInfoPIC.localSizeX * mPIInfoPIC.localSizeY), 

      USub (mPIInfoMHD.localSizeX * mPIInfoMHD.localSizeY), 
      UHalf(mPIInfoMHD.localSizeX * mPIInfoMHD.localSizeY), 

      momentCalculater(mPIInfoPIC), 
      interfaceNoiseRemover2D(interfaceNoiseRemover2D)
{

    cudaMalloc(&device_mPIInfoMHD, sizeof(IdealMHD2DMPI::MPIInfo));
    cudaMemcpy(device_mPIInfoMHD, &mPIInfoMHD, sizeof(IdealMHD2DMPI::MPIInfo), cudaMemcpyHostToDevice);
    cudaMalloc(&device_mPIInfoPIC, sizeof(PIC2DMPI::MPIInfo));
    cudaMemcpy(device_mPIInfoPIC, &mPIInfoPIC, sizeof(PIC2DMPI::MPIInfo), cudaMemcpyHostToDevice);

    interlockingFunctionY = host_interlockingFunctionY;
    interlockingFunctionYHalf = host_interlockingFunctionYHalf;

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((Interface2DConst::reloadParticlesTotalNum + threadsPerBlock.x - 1) / threadsPerBlock.x);

    initializeReloadParticlesSource_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(reloadParticlesSourceIon.data()),
        Interface2DConst::reloadParticlesTotalNum, 
        100000
    );
    cudaDeviceSynchronize();

    initializeReloadParticlesSource_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(reloadParticlesSourceElectron.data()),
        Interface2DConst::reloadParticlesTotalNum, 
        200000
    );
    cudaDeviceSynchronize();
}

