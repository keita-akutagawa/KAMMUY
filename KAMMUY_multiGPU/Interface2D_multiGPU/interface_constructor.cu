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
        float EPS = 0.00001f;
        while (true) {
            x  = curand_uniform(&stateX);
            y  = curand_uniform(&stateY);
            z  = 0.0f;

            if (EPS < x && x < 1.0f - EPS && EPS < y && y < 1.0f - EPS) break;
        }
        
        while (true) {
            vx = curand_normal(&stateVx);
            vy = curand_normal(&stateVy);
            vz = curand_normal(&stateVz);

            if (vx * vx + vy * vy + vz * vz < 1.0f * 1.0f) break;
        }

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
    bool isLower, bool isUpper, 
    int indexOfInterfaceStartMHD, 
    int indexOfInterfaceStartPIC, 
    int interfaceLength, 
    thrust::host_vector<double>& host_interlockingFunctionY, 
    thrust::host_vector<double>& host_interlockingFunctionYHalf, 
    InterfaceNoiseRemover2D& interfaceNoiseRemover2D
)
    : mPIInfoMHD(mPIInfoMHD), 
      mPIInfoPIC(mPIInfoPIC), 
      mPIInfoInterface(mPIInfoInterface), 

      isLower(isLower), 
      isUpper(isUpper), 

      indexOfInterfaceStartInMHD(indexOfInterfaceStartMHD), 
      indexOfInterfaceStartInPIC(indexOfInterfaceStartPIC), 
      interfaceLength(interfaceLength), 
      indexOfInterfaceEndInMHD(indexOfInterfaceStartMHD + interfaceLength), 
      indexOfInterfaceEndInPIC(indexOfInterfaceStartPIC + interfaceLength), 

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

      reloadParticlesDataIon          (mPIInfoInterface.localSizeX * mPIInfoInterface.localSizeY + 1), 
      reloadParticlesDataElectron     (mPIInfoInterface.localSizeX * mPIInfoInterface.localSizeY + 1), 
      host_reloadParticlesDataIon     (mPIInfoInterface.localSizeX * mPIInfoInterface.localSizeY + 1), 
      host_reloadParticlesDataElectron(mPIInfoInterface.localSizeX * mPIInfoInterface.localSizeY + 1), 
      
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

