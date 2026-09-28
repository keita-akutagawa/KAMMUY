#include "smr_boundary.hpp"


SMRBoundary::SMRBoundary(
    const nlohmann::json& configJSON, 
    const MHDInt level, 
    MHDConstParameter& mHDConstParameter, 
    const MHDGridParameter& mHDGridParameter
)
{
    SMRBoundaryXLeft  = SMRBoundaryXLeftFactory::create( configJSON, level, mHDConstParameter, mHDGridParameter); 
    SMRBoundaryXRight = SMRBoundaryXRightFactory::create(configJSON, level, mHDConstParameter, mHDGridParameter); 
    SMRBoundaryYDown  = SMRBoundaryYDownFactory::create( configJSON, level, mHDConstParameter, mHDGridParameter); 
    SMRBoundaryYUp    = SMRBoundaryYUpFactory::create(   configJSON, level, mHDConstParameter, mHDGridParameter); 
}


void SMRBoundary::applyUForAllDirection(
    const thrust::device_vector<MHDValue>& UPast, 
    const thrust::device_vector<MHDValue>& UNext, 
    const MHDFloat timeRatio, 
    thrust::device_vector<MHDValue>& smrU
)
{
    SMRBoundaryXLeft->apply( UPast, UNext, timeRatio, smrU);
    SMRBoundaryXRight->apply(UPast, UNext, timeRatio, smrU);
    SMRBoundaryYDown->apply( UPast, UNext, timeRatio, smrU);
    SMRBoundaryYUp->apply(   UPast, UNext, timeRatio, smrU);
}


