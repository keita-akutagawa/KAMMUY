#include "boundary.hpp"


BoundaryMHD::BoundaryMHD(IdealMHD2DMPI::MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo), 

      sendULeft(IdealMHD2DConst::ny * mPIInfo.buffer), 
      sendURight(IdealMHD2DConst::ny * mPIInfo.buffer), 
      recvULeft(IdealMHD2DConst::ny * mPIInfo.buffer), 
      recvURight(IdealMHD2DConst::ny * mPIInfo.buffer)
{
}

