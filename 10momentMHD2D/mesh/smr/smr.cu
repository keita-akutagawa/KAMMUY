#include "smr.hpp"


SMR::SMR(
    nlohmann::json& configJSON, 
    const MHDInt level, 
    MHDConstParameter& mHDConstParameter, 
    const MHDGridParameter& mHDGridParameter
)
  : level(level), 
    COARSE_NX(mHDGridParameter.NX[level - 1]), 
    COARSE_NY(mHDGridParameter.NY[level - 1]), 
    SMR_NX(mHDGridParameter.NX[level]), 
    SMR_NY(mHDGridParameter.NY[level]), 
    SMR_DX(mHDGridParameter.DX[level]), 
    SMR_DY(mHDGridParameter.DY[level]), 
    START_INDEX_X(mHDGridParameter.START_INDEX_X[level]), 
    START_INDEX_Y(mHDGridParameter.START_INDEX_Y[level]), 
    END_INDEX_X(mHDGridParameter.END_INDEX_X[level]), 
    END_INDEX_Y(mHDGridParameter.END_INDEX_Y[level]), 
    mHDConstParameter(mHDConstParameter), 
    mHDGridParameter(mHDGridParameter)
{
}

