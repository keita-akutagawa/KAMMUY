#include "boundary.hpp"


Boundary::Boundary(
    const nlohmann::json& configJSON, 
    const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
    MHDConstParameter& mHDConstParameter, 
    const MHDGridParameter& mHDGridParameter
) 
{
    boundaryXLeft  = BoundaryXLeftFactory::create( configJSON, NX, NY, mHDConstParameter, mHDGridParameter); 
    boundaryXRight = BoundaryXRightFactory::create(configJSON, NX, NY, mHDConstParameter, mHDGridParameter); 
    boundaryYDown  = BoundaryYDownFactory::create( configJSON, NX, NY, mHDConstParameter, mHDGridParameter); 
    boundaryYUp    = BoundaryYUpFactory::create(   configJSON, NX, NY, mHDConstParameter, mHDGridParameter); 
}


void Boundary::applyUForAllDirection(
    thrust::device_vector<MHDValue>& U
)
{
    boundaryXLeft->apply(U);
    boundaryXRight->apply(U);
    boundaryYDown->apply(U);
    boundaryYUp->apply(U);
}

