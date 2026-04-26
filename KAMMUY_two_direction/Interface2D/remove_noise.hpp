#ifndef INTERFACE_REMOVE_NOISE_H
#define INTERFACE_REMOVE_NOISE_H


#include <thrust/device_vector.h>
#include "const.hpp"
#include "../IdealMHD2D/const.hpp"
#include "../IdealMHD2D/conservation_parameter_struct.hpp"
#include "../IdealMHD2D/basic_parameter_struct.hpp"
#include "../PIC2D/const.hpp"
#include "../PIC2D/field_parameter_struct.hpp"
#include "../PIC2D/moment_struct.hpp"


class InterfaceNoiseRemover2D
{
private:
    thrust::device_vector<ConservationParameter> tmpU;

public:
    InterfaceNoiseRemover2D();

    void convolveU(
        thrust::device_vector<ConservationParameter>& U
    );

private:

};


#endif

