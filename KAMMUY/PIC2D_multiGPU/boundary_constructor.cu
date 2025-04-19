#include "boundary.hpp"


BoundaryPIC::BoundaryPIC(PIC2DMPI::MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo), 
      
      sendParticlesSpeciesLeft(mPIInfo.mpiBufNumParticles), 
      sendParticlesSpeciesRight(mPIInfo.mpiBufNumParticles), 
      recvParticlesSpeciesLeft(mPIInfo.mpiBufNumParticles), 
      recvParticlesSpeciesRight(mPIInfo.mpiBufNumParticles), 

      sendMagneticFieldLeft(PIC2DConst::ny * mPIInfo.buffer), 
      sendMagneticFieldRight(PIC2DConst::ny * mPIInfo.buffer), 
      recvMagneticFieldLeft(PIC2DConst::ny * mPIInfo.buffer), 
      recvMagneticFieldRight(PIC2DConst::ny * mPIInfo.buffer), 

      sendElectricFieldLeft(PIC2DConst::ny * mPIInfo.buffer), 
      sendElectricFieldRight(PIC2DConst::ny * mPIInfo.buffer), 
      recvElectricFieldLeft(PIC2DConst::ny * mPIInfo.buffer), 
      recvElectricFieldRight(PIC2DConst::ny * mPIInfo.buffer), 

      sendCurrentFieldLeft(PIC2DConst::ny * mPIInfo.buffer), 
      sendCurrentFieldRight(PIC2DConst::ny * mPIInfo.buffer), 
      recvCurrentFieldLeft(PIC2DConst::ny * mPIInfo.buffer), 
      recvCurrentFieldRight(PIC2DConst::ny * mPIInfo.buffer), 

      sendZerothMomentLeft(PIC2DConst::ny * mPIInfo.buffer), 
      sendZerothMomentRight(PIC2DConst::ny * mPIInfo.buffer), 
      recvZerothMomentLeft(PIC2DConst::ny * mPIInfo.buffer), 
      recvZerothMomentRight(PIC2DConst::ny * mPIInfo.buffer), 

      sendFirstMomentLeft(PIC2DConst::ny * mPIInfo.buffer), 
      sendFirstMomentRight(PIC2DConst::ny * mPIInfo.buffer), 
      recvFirstMomentLeft(PIC2DConst::ny * mPIInfo.buffer), 
      recvFirstMomentRight(PIC2DConst::ny * mPIInfo.buffer), 

      sendSecondMomentLeft(PIC2DConst::ny * mPIInfo.buffer), 
      sendSecondMomentRight(PIC2DConst::ny * mPIInfo.buffer), 
      recvSecondMomentLeft(PIC2DConst::ny * mPIInfo.buffer), 
      recvSecondMomentRight(PIC2DConst::ny * mPIInfo.buffer)
{

    cudaMalloc(&device_mPIInfo, sizeof(PIC2DMPI::MPIInfo));
    cudaMemcpy(device_mPIInfo, &mPIInfo, sizeof(PIC2DMPI::MPIInfo), cudaMemcpyHostToDevice);

}

