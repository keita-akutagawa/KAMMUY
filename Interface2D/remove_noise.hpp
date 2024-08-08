#ifndef CONST_INTERFACE_REMOVE_NOISE_H
#define CONST_INTERFACE_REMOVE_NOISE_H


#include <thrust/device_vector.h>
#include "const.hpp"
#include "../IdealMHD2D_gpu/const.hpp"
#include "../IdealMHD2D_gpu/conservation_parameter_struct.hpp"
#include "../PIC2D_gpu/const.hpp"
#include "../PIC2D_gpu/field_parameter_struct.hpp"
#include "../PIC2D_gpu/moment_struct.hpp"


class InterfaceNoiseRemover2D
{
private:
    int indexOfInterfaceStartInMHD;
    int indexOfInterfaceStartInPIC;
    int interfaceLength; 
    int indexOfInterfaceEndInMHD;
    int indexOfInterfaceEndInPIC;

    int windowSize;

    thrust::device_vector<MagneticField> tmpB;
    thrust::device_vector<ElectricField> tmpE;
    thrust::device_vector<CurrentField> tmpCurrent;
    thrust::device_vector<ZerothMoment> tmpZerothMoment;
    thrust::device_vector<FirstMoment> tmpFirstMoment;
    thrust::device_vector<ConservationParameter> tmpU;

public:
    InterfaceNoiseRemover2D(
        int indexStartMHD, 
        int indexStartPIC, 
        int interfaceLength, 
        int windowSizeForConvolution
    );

    void convolveFields(
        thrust::device_vector<MagneticField>& B, 
        thrust::device_vector<ElectricField>& E, 
        thrust::device_vector<CurrentField>& current
    );

    void convolveMoments(
        thrust::device_vector<ZerothMoment>& zerothMomentIon, 
        thrust::device_vector<ZerothMoment>& zerothMomentElectron, 
        thrust::device_vector<FirstMoment>& firstMomentIon, 
        thrust::device_vector<FirstMoment>& firstMomentElectron
    );

    void convolveU(
        thrust::device_vector<ConservationParameter>& U
    );

private:

};


#endif

