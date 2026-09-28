#ifndef MHD_BOUNDARY_FOR_ONE_SIDE_HPP 
#define MHD_BOUNDARY_FOR_ONE_SIDE_HPP 

#include <thrust/device_vector.h> 
#include "../mhd_value.hpp"
#include "../global_function.hpp"
#include "../../utils/get_index.hpp"


class BoundaryForOneSide
{
private: 

public:
    virtual void apply(
        thrust::device_vector<MHDValue>& U
    ) = 0; 

    virtual ~BoundaryForOneSide() {}

private: 

};

#endif 
