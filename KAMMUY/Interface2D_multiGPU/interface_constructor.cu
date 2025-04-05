#include "interface.hpp"


__global__ void initializeReloadParticlesSource_kernel(
    Particle* reloadParticlesSourceSpecies, 
    unsigned long long reloadParticlesNumSpecies, 
    int seed
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < reloadParticlesNumSpecies) {
        curandState state;
        curand_init(seed, i, 0, &state);

        float x, y, z, vx, vy, vz;
        
        x = curand_uniform(&state);
        y = curand_uniform(&state);
        z = 0.0f;
        vx = curand_normal(&state);
        vy = curand_normal(&state);
        vz = curand_normal(&state);

        x = min(max(x, PIC2DConst::device_EPS), 1.0f - PIC2DConst::device_EPS); 
        y = min(max(y, PIC2DConst::device_EPS), 1.0f - PIC2DConst::device_EPS); 

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
    thrust::host_vector<double>& host_interlockingFunctionY, 
    InterfaceNoiseRemover2D& interfaceNoiseRemover2D
)
    : mPIInfoMHD(mPIInfoMHD), 
      mPIInfoPIC(mPIInfoPIC), 
      mPIInfoInterface(mPIInfoInterface), 

      indexOfInterfaceStartInMHD(indexOfInterfaceStartMHD), 

      interlockingFunctionY(mPIInfoPIC.localSizeX * PIC2DConst::ny, 0.0), 

      B_timeAve                   (mPIInfoPIC.localSizeX * PIC2DConst::ny), 
      zerothMomentIon_timeAve     (mPIInfoPIC.localSizeX * PIC2DConst::ny), 
      zerothMomentElectron_timeAve(mPIInfoPIC.localSizeX * PIC2DConst::ny), 
      firstMomentIon_timeAve      (mPIInfoPIC.localSizeX * PIC2DConst::ny), 
      firstMomentElectron_timeAve (mPIInfoPIC.localSizeX * PIC2DConst::ny),

      restartParticlesIndexIon(0), 
      restartParticlesIndexElectron(0), 

      reloadParticlesSourceIon     (Interface2DConst::reloadParticlesTotalNum), 
      reloadParticlesSourceElectron(Interface2DConst::reloadParticlesTotalNum), 

      reloadParticlesDataIon     (mPIInfoPIC.localNx * PIC2DConst::ny), 
      reloadParticlesDataElectron(mPIInfoPIC.localNx * PIC2DConst::ny), 
      
      B_PICtoMHD                   (mPIInfoMHD.localNx * (PIC2DConst::ny / Interface2DConst::gridSizeRatio)), 
      zerothMomentIon_PICtoMHD     (mPIInfoMHD.localNx * (PIC2DConst::ny / Interface2DConst::gridSizeRatio)), 
      zerothMomentElectron_PICtoMHD(mPIInfoMHD.localNx * (PIC2DConst::ny / Interface2DConst::gridSizeRatio)), 
      firstMomentIon_PICtoMHD      (mPIInfoMHD.localNx * (PIC2DConst::ny / Interface2DConst::gridSizeRatio)), 
      firstMomentElectron_PICtoMHD (mPIInfoMHD.localNx * (PIC2DConst::ny / Interface2DConst::gridSizeRatio)), 

      USub (mPIInfoMHD.localSizeX * IdealMHD2DConst::ny), 
      UHalf(mPIInfoMHD.localSizeX * IdealMHD2DConst::ny), 

      momentCalculator(mPIInfoPIC), 
      boundaryPIC(mPIInfoPIC), 
      interfaceNoiseRemover2D(interfaceNoiseRemover2D)
{

    cudaMalloc(&device_mPIInfoMHD, sizeof(IdealMHD2DMPI::MPIInfo));
    cudaMemcpy(device_mPIInfoMHD, &mPIInfoMHD, sizeof(IdealMHD2DMPI::MPIInfo), cudaMemcpyHostToDevice);
    cudaMalloc(&device_mPIInfoPIC, sizeof(PIC2DMPI::MPIInfo));
    cudaMemcpy(device_mPIInfoPIC, &mPIInfoPIC, sizeof(PIC2DMPI::MPIInfo), cudaMemcpyHostToDevice);
    cudaMalloc(&device_mPIInfoInterface, sizeof(Interface2DMPI::MPIInfo));
    cudaMemcpy(device_mPIInfoInterface, &mPIInfoInterface, sizeof(Interface2DMPI::MPIInfo), cudaMemcpyHostToDevice);

    interlockingFunctionY = host_interlockingFunctionY;
    

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

