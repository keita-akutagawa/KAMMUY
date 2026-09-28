#ifndef MHD_SMR_BOUNDARY_HPP 
#define MHD_SMR_BOUNDARY_HPP 

#include <thrust/device_vector.h> 
#include <nlohmann/json.hpp>

#include "../../../mhd_value.hpp"
#include "../../../const_parameter.hpp"
#include "../../../grid_parameter.hpp"
#include "../../../type.hpp"
#include "smr_boundary_for_one_side.hpp"
#include "smr_boundary_for_one_side_factory.hpp"


class SMRBoundary 
{
private: 

    std::unique_ptr<SMRBoundaryForOneSide> SMRBoundaryXLeft; 
    std::unique_ptr<SMRBoundaryForOneSide> SMRBoundaryXRight; 
    std::unique_ptr<SMRBoundaryForOneSide> SMRBoundaryYDown; 
    std::unique_ptr<SMRBoundaryForOneSide> SMRBoundaryYUp; 

public:
    SMRBoundary(
        const nlohmann::json& configJSON, 
        const MHDInt level, 
        MHDConstParameter& mHDConstParameter, 
        const MHDGridParameter& mHDGridParameter
    ); 

    void applyUForAllDirection(
        const thrust::device_vector<MHDValue>& UPast, 
        const thrust::device_vector<MHDValue>& UNext, 
        const MHDFloat timeRatio, 
        thrust::device_vector<MHDValue>& smrU
    ); 
    
private: 

};

#endif 
