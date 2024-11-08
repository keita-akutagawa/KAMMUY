#ifndef INTERFACE_REMOVE_NOISE_H
#define INTERFACE_REMOVE_NOISE_H


#include <thrust/device_vector.h>
#include "const.hpp"
#include "../IdealMHD2D_multiGPU/const.hpp"
#include "../IdealMHD2D_multiGPU/conservation_parameter_struct.hpp"
#include "../PIC2D_multiGPU/const.hpp"
#include "../PIC2D_multiGPU/field_parameter_struct.hpp"
#include "../PIC2D_multiGPU/moment_struct.hpp"


class InterfaceNoiseRemover2D
{
private:
    int indexOfInterfaceStartInMHD;
    int indexOfInterfaceStartInPIC;
    int interfaceLength; 
    int windowSize;
    int nx_Interface; 
    int ny_Interface;

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
        int windowSizeForConvolution, 
        int nx_Interface, 
        int ny_Interface
    );


    void convolve_lower_magneticField(
        thrust::device_vector<MagneticField>& B
    );

    void convolve_lower_electricField(
        thrust::device_vector<ElectricField>& E
    );

    void convolve_lower_currentField(
        thrust::device_vector<CurrentField>& current
    );

    void convolveMoments_lower(
        thrust::device_vector<ZerothMoment>& zerothMomentIon, 
        thrust::device_vector<ZerothMoment>& zerothMomentElectron, 
        thrust::device_vector<FirstMoment>& firstMomentIon, 
        thrust::device_vector<FirstMoment>& firstMomentElectron
    );

    void convolveU_lower(
        thrust::device_vector<ConservationParameter>& U
    );


    void convolve_upper_magneticField(
        thrust::device_vector<MagneticField>& B
    );

    void convolve_upper_electricField(
        thrust::device_vector<ElectricField>& E
    );

    void convolve_upper_currentField(
        thrust::device_vector<CurrentField>& current
    );

    void convolveMoments_upper(
        thrust::device_vector<ZerothMoment>& zerothMomentIon, 
        thrust::device_vector<ZerothMoment>& zerothMomentElectron, 
        thrust::device_vector<FirstMoment>& firstMomentIon, 
        thrust::device_vector<FirstMoment>& firstMomentElectron
    );

    void convolveU_upper(
        thrust::device_vector<ConservationParameter>& U
    );

private:

};


#endif

