#ifndef BOUNDARY_MHD_H
#define BOUNDARY_MHD_H

#include <thrust/device_vector.h>
#include "const.hpp"
#include "conservation_parameter_struct.hpp"
#include "flux_struct.hpp"
#include "mpi.hpp"


class BoundaryMHD
{
private:
    IdealMHD2DMPI::MPIInfo mPIInfo; 
    IdealMHD2DMPI::MPIInfo* device_mPIInfo; 

public:
    BoundaryMHD(IdealMHD2DMPI::MPIInfo& mPIInfo);

    void periodicBoundaryX2nd_U(
        thrust::device_vector<ConservationParameter>& U
    );

    void periodicBoundaryY2nd_U(
        thrust::device_vector<ConservationParameter>& U
    );

    void periodicBoundaryX2nd_flux(
        thrust::device_vector<Flux>& fluxF, 
        thrust::device_vector<Flux>& fluxG
    );

    void periodicBoundaryY2nd_flux(
        thrust::device_vector<Flux>& fluxF, 
        thrust::device_vector<Flux>& fluxG
    );

    /*
    void wallBoundaryX2nd_U(
        thrust::device_vector<ConservationParameter>& U
    );
    */

    void wallBoundaryY2nd_U(
        thrust::device_vector<ConservationParameter>& U
    );

    void wallBoundaryY2nd_flux(
        thrust::device_vector<Flux>& fluxF, 
        thrust::device_vector<Flux>& fluxG
    );

    /*
    void symmetricBoundaryX2nd_U(
        thrust::device_vector<ConservationParameter>& U
    );
    */

    void symmetricBoundaryY2nd_U(
        thrust::device_vector<ConservationParameter>& U
    );

    void symmetricBoundaryY2nd_flux(
        thrust::device_vector<Flux>& fluxF, 
        thrust::device_vector<Flux>& fluxG
    );

private:

};

#endif


