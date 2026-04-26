#ifndef INTERFACE_REMOVE_NOISE_H
#define INTERFACE_REMOVE_NOISE_H


#include <thrust/device_vector.h>
#include "const.hpp"
#include "../IdealMHD2D_multiGPU/const.hpp"
#include "../IdealMHD2D_multiGPU/conservation_parameter_struct.hpp"
#include "../IdealMHD2D_multiGPU/basic_parameter_struct.hpp"
#include "../PIC2D_singleGPU/const.hpp"
#include "../PIC2D_singleGPU/field_parameter_struct.hpp"
#include "../PIC2D_singleGPU/moment_struct.hpp"
#include "../IdealMHD2D_multiGPU/mpi.hpp"


class InterfaceNoiseRemover2D
{
private:
    IdealMHD2DMPI::MPIInfo& mPIInfoMHD;
    IdealMHD2DMPI::MPIInfo* device_mPIInfoMHD; 

    thrust::device_vector<ConservationParameter> tmpU;

public:
    InterfaceNoiseRemover2D(
        IdealMHD2DMPI::MPIInfo& mPIInfoMHD
    );

    void convolveU(
        thrust::device_vector<ConservationParameter>& U
    );

private:

};


#endif

