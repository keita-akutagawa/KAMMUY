#include "custom_boundary.hpp"


SMRCustomBoundaryXLeft::SMRCustomBoundaryXLeft(
    const MHDInt level, 
    MHDConstParameter& mHDConstParameter, 
    const MHDGridParameter& mHDGridParameter 
)
  : level(level), 
    mHDConstParameter(mHDConstParameter), 
    mHDGridParameter(mHDGridParameter), 
    NX(mHDGridParameter.NX[level - 1]), 
    NY(mHDGridParameter.NY[level - 1]),  
    SMR_NX(mHDGridParameter.NX[level]), 
    SMR_NY(mHDGridParameter.NY[level]),   
    START_INDEX_X(mHDGridParameter.START_INDEX_X[level]), 
    START_INDEX_Y(mHDGridParameter.START_INDEX_Y[level])   
{
}


SMRCustomBoundaryXRight::SMRCustomBoundaryXRight(
    const MHDInt level, 
    MHDConstParameter& mHDConstParameter, 
    const MHDGridParameter& mHDGridParameter
)
  : level(level), 
    mHDConstParameter(mHDConstParameter), 
    mHDGridParameter(mHDGridParameter), 
    NX(mHDGridParameter.NX[level - 1]), 
    NY(mHDGridParameter.NY[level - 1]),  
    SMR_NX(mHDGridParameter.NX[level]), 
    SMR_NY(mHDGridParameter.NY[level]),   
    START_INDEX_X(mHDGridParameter.START_INDEX_X[level]), 
    START_INDEX_Y(mHDGridParameter.START_INDEX_Y[level])   
{
}


SMRCustomBoundaryYDown::SMRCustomBoundaryYDown(
    const MHDInt level, 
    MHDConstParameter& mHDConstParameter, 
    const MHDGridParameter& mHDGridParameter
)
  : level(level), 
    mHDConstParameter(mHDConstParameter), 
    mHDGridParameter(mHDGridParameter), 
    NX(mHDGridParameter.NX[level - 1]), 
    NY(mHDGridParameter.NY[level - 1]),  
    SMR_NX(mHDGridParameter.NX[level]), 
    SMR_NY(mHDGridParameter.NY[level]),   
    START_INDEX_X(mHDGridParameter.START_INDEX_X[level]), 
    START_INDEX_Y(mHDGridParameter.START_INDEX_Y[level])   
{
}


SMRCustomBoundaryYUp::SMRCustomBoundaryYUp(
    const MHDInt level, 
    MHDConstParameter& mHDConstParameter, 
    const MHDGridParameter& mHDGridParameter
)
  : level(level), 
    mHDConstParameter(mHDConstParameter), 
    mHDGridParameter(mHDGridParameter), 
    NX(mHDGridParameter.NX[level - 1]), 
    NY(mHDGridParameter.NY[level - 1]),  
    SMR_NX(mHDGridParameter.NX[level]), 
    SMR_NY(mHDGridParameter.NY[level]),   
    START_INDEX_X(mHDGridParameter.START_INDEX_X[level]), 
    START_INDEX_Y(mHDGridParameter.START_INDEX_Y[level])   
{
}
