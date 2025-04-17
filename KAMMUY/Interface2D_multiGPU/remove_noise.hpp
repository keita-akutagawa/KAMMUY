#ifndef INTERFACE_REMOVE_NOISE_H
#define INTERFACE_REMOVE_NOISE_H


#include <thrust/device_vector.h>
#include "const.hpp"
#include "../IdealMHD2D_multiGPU/const.hpp"
#include "../IdealMHD2D_multiGPU/conservation_parameter_struct.hpp"
#include "../IdealMHD2D_multiGPU/basic_parameter_struct.hpp"
#include "../PIC2D_multiGPU/const.hpp"
#include "../PIC2D_multiGPU/field_parameter_struct.hpp"
#include "../PIC2D_multiGPU/moment_struct.hpp"
#include "../IdealMHD2D_multiGPU/mpi.hpp"
#include "../PIC2D_multiGPU/mpi.hpp"


class InterfaceNoiseRemover2D
{
private:
    IdealMHD2DMPI::MPIInfo& mPIInfoMHD;
    PIC2DMPI::MPIInfo& mPIInfoPIC;
    IdealMHD2DMPI::MPIInfo* device_mPIInfoMHD; 
    PIC2DMPI::MPIInfo* device_mPIInfoPIC; 

    thrust::device_vector<ConservationParameter> tmpU;

public:
    InterfaceNoiseRemover2D(
        IdealMHD2DMPI::MPIInfo& mPIInfoMHD, 
        PIC2DMPI::MPIInfo& mPIInfoPIC
    );

    void convolveU(
        thrust::device_vector<ConservationParameter>& U
    );

private:

};


#endif

