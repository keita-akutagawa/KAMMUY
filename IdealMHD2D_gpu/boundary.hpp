#ifndef CONST_MHD_BOUNDARY_H
#define CONST_MHD_BOUNDARY_H


#include <thrust/device_vector.h>
#include "const.hpp"
#include "conservation_parameter_struct.hpp"


class BoundaryMHD
{
private:

public:

    void periodicBoundaryX2nd(
        thrust::device_vector<ConservationParameter>& U
    );

    void symmetricBoundaryX2nd(
        thrust::device_vector<ConservationParameter>& U
    );

    void periodicBoundaryY2nd(
        thrust::device_vector<ConservationParameter>& U
    );

    void symmetricBoundaryY2nd(
        thrust::device_vector<ConservationParameter>& U
    );

private:

};


#endif

