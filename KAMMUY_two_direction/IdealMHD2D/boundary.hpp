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

    void periodicBoundaryX2nd_U(
        thrust::device_vector<ConservationParameter>& U
    );

    void periodicBoundaryY2nd_U(
        thrust::device_vector<ConservationParameter>& U
    );

    void wallBoundaryY2nd_U(
        thrust::device_vector<ConservationParameter>& U
    );

    void wallBoundaryY2nd_flux(
        thrust::device_vector<Flux>& fluxF, 
        thrust::device_vector<Flux>& fluxG
    );

    void symmetricBoundaryY2nd_U(
        thrust::device_vector<ConservationParameter>& U
    );

private:

};

#endif


