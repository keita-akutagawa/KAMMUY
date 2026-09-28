#ifndef MHD_BOUNDARY_HPP 
#define MHD_BOUNDARY_HPP 

#include <thrust/device_vector.h> 
#include <nlohmann/json.hpp>

#include "../mhd_value.hpp"
#include "../const_parameter.hpp"
#include "../grid_parameter.hpp"
#include "../type.hpp"
#include "boundary_for_one_side_factory.hpp"


class Boundary 
{
private: 

public:

    std::unique_ptr<BoundaryForOneSide> boundaryXLeft; 
    std::unique_ptr<BoundaryForOneSide> boundaryXRight; 
    std::unique_ptr<BoundaryForOneSide> boundaryYDown; 
    std::unique_ptr<BoundaryForOneSide> boundaryYUp; 

    Boundary(
        const nlohmann::json& configJSON, 
        const MHDUnsignedInt NX, const MHDUnsignedInt NY,
        MHDConstParameter& mHDConstParameter, 
        const MHDGridParameter& mHDGridParameter
    ); 

    void applyUForAllDirection(
        thrust::device_vector<MHDValue>& U
    ); 

    
private: 

};

#endif 
