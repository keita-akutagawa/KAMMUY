#include "custom_boundary.hpp"


CustomBoundaryXLeft::CustomBoundaryXLeft(
    const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
    MHDConstParameter& mHDConstParameter, 
    const MHDGridParameter& mHDGridParameter 
)
  : NX(NX), 
    NY(NY), 
    mHDConstParameter(mHDConstParameter), 
    mHDGridParameter(mHDGridParameter)
{
}

CustomBoundaryXRight::CustomBoundaryXRight(
    const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
    MHDConstParameter& mHDConstParameter, 
    const MHDGridParameter& mHDGridParameter
)
  : NX(NX), 
    NY(NY), 
    mHDConstParameter(mHDConstParameter), 
    mHDGridParameter(mHDGridParameter)
{
}

CustomBoundaryYDown::CustomBoundaryYDown(
    const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
    MHDConstParameter& mHDConstParameter, 
    const MHDGridParameter& mHDGridParameter
)
  : NX(NX), 
    NY(NY), 
    mHDConstParameter(mHDConstParameter), 
    mHDGridParameter(mHDGridParameter)
{
}

CustomBoundaryYUp::CustomBoundaryYUp(
    const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
    MHDConstParameter& mHDConstParameter, 
    const MHDGridParameter& mHDGridParameter
)
  : NX(NX), 
    NY(NY), 
    mHDConstParameter(mHDConstParameter), 
    mHDGridParameter(mHDGridParameter)
{
}
