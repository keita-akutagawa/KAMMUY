#ifndef BOUNDARY_MHD_H
#define BOUNDARY_MHD_H

#include <thrust/device_vector.h>
#include "const.hpp"
#include "conservation_parameter_struct.hpp"
#include "flux_struct.hpp"


class BoundaryMHD
{
private:

public:
    BoundaryMHD();

    void periodicBoundary_x(
        thrust::device_vector<ConservationParameter>& U
    );

    void periodicBoundary_y(
        thrust::device_vector<ConservationParameter>& U
    );

    void wallBoundary_y(
        thrust::device_vector<ConservationParameter>& U
    );

    void symmetricBoundary_y(
        thrust::device_vector<ConservationParameter>& U
    );

private:

};

#endif


