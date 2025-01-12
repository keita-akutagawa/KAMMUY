#include "boundary.hpp"


BoundaryPIC::BoundaryPIC(PIC2DMPI::MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo), 
      
      sendParticlesSpeciesLeft(mPIInfo.mpiBufNumParticles), 
      sendParticlesSpeciesRight(mPIInfo.mpiBufNumParticles), 
      sendParticlesSpeciesDown(mPIInfo.mpiBufNumParticles), 
      sendParticlesSpeciesUp(mPIInfo.mpiBufNumParticles), 
      recvParticlesSpeciesLeft(mPIInfo.mpiBufNumParticles), 
      recvParticlesSpeciesRight(mPIInfo.mpiBufNumParticles), 
      recvParticlesSpeciesDown(mPIInfo.mpiBufNumParticles), 
      recvParticlesSpeciesUp(mPIInfo.mpiBufNumParticles), 

      sendMagneticFieldLeft(mPIInfo.localNy * mPIInfo.buffer), 
      sendMagneticFieldRight(mPIInfo.localNy * mPIInfo.buffer), 
      recvMagneticFieldLeft(mPIInfo.localNy * mPIInfo.buffer), 
      recvMagneticFieldRight(mPIInfo.localNy * mPIInfo.buffer), 

      sendMagneticFieldDown(mPIInfo.localSizeX * mPIInfo.buffer), 
      sendMagneticFieldUp(mPIInfo.localSizeX * mPIInfo.buffer), 
      recvMagneticFieldDown(mPIInfo.localSizeX * mPIInfo.buffer), 
      recvMagneticFieldUp(mPIInfo.localSizeX * mPIInfo.buffer), 

      sendElectricFieldLeft(mPIInfo.localNy * mPIInfo.buffer), 
      sendElectricFieldRight(mPIInfo.localNy * mPIInfo.buffer), 
      recvElectricFieldLeft(mPIInfo.localNy * mPIInfo.buffer), 
      recvElectricFieldRight(mPIInfo.localNy * mPIInfo.buffer), 

      sendElectricFieldDown(mPIInfo.localSizeX * mPIInfo.buffer), 
      sendElectricFieldUp(mPIInfo.localSizeX * mPIInfo.buffer), 
      recvElectricFieldDown(mPIInfo.localSizeX * mPIInfo.buffer), 
      recvElectricFieldUp(mPIInfo.localSizeX * mPIInfo.buffer), 

      sendCurrentFieldLeft(mPIInfo.localNy * mPIInfo.buffer), 
      sendCurrentFieldRight(mPIInfo.localNy * mPIInfo.buffer), 
      recvCurrentFieldLeft(mPIInfo.localNy * mPIInfo.buffer), 
      recvCurrentFieldRight(mPIInfo.localNy * mPIInfo.buffer), 

      sendCurrentFieldDown(mPIInfo.localSizeX * mPIInfo.buffer), 
      sendCurrentFieldUp(mPIInfo.localSizeX * mPIInfo.buffer), 
      recvCurrentFieldDown(mPIInfo.localSizeX * mPIInfo.buffer), 
      recvCurrentFieldUp(mPIInfo.localSizeX * mPIInfo.buffer), 

      sendZerothMomentLeft(mPIInfo.localNy * mPIInfo.buffer), 
      sendZerothMomentRight(mPIInfo.localNy * mPIInfo.buffer), 
      recvZerothMomentLeft(mPIInfo.localNy * mPIInfo.buffer), 
      recvZerothMomentRight(mPIInfo.localNy * mPIInfo.buffer), 

      sendZerothMomentDown(mPIInfo.localSizeX * mPIInfo.buffer), 
      sendZerothMomentUp(mPIInfo.localSizeX * mPIInfo.buffer), 
      recvZerothMomentDown(mPIInfo.localSizeX * mPIInfo.buffer), 
      recvZerothMomentUp(mPIInfo.localSizeX * mPIInfo.buffer), 

      sendFirstMomentLeft(mPIInfo.localNy * mPIInfo.buffer), 
      sendFirstMomentRight(mPIInfo.localNy * mPIInfo.buffer), 
      recvFirstMomentLeft(mPIInfo.localNy * mPIInfo.buffer), 
      recvFirstMomentRight(mPIInfo.localNy * mPIInfo.buffer), 

      sendFirstMomentDown(mPIInfo.localSizeX * mPIInfo.buffer), 
      sendFirstMomentUp(mPIInfo.localSizeX * mPIInfo.buffer), 
      recvFirstMomentDown(mPIInfo.localSizeX * mPIInfo.buffer), 
      recvFirstMomentUp(mPIInfo.localSizeX * mPIInfo.buffer)  
{

    cudaMalloc(&device_mPIInfo, sizeof(PIC2DMPI::MPIInfo));
    cudaMemcpy(device_mPIInfo, &mPIInfo, sizeof(PIC2DMPI::MPIInfo), cudaMemcpyHostToDevice);

}

