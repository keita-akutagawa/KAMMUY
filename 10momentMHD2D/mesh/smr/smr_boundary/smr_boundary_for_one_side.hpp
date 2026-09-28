#ifndef MHD_SMR_BOUNDARY_FOR_ONE_SIDE_HPP 
#define MHD_SMR_BOUNDARY_FOR_ONE_SIDE_HPP 

#include <thrust/device_vector.h> 
#include "../../../mhd_value.hpp"
#include "../../../global_function.hpp"
#include "../../../../utils/get_index.hpp"


class SMRBoundaryForOneSide
{
private: 

public:
    virtual void apply(
        const thrust::device_vector<MHDValue>& coarseUPast, 
        const thrust::device_vector<MHDValue>& coarseUNext, 
        const MHDFloat timeRatio, 
        thrust::device_vector<MHDValue>& U
    ) = 0; 

    virtual ~SMRBoundaryForOneSide() {}

private: 

};

#endif 
