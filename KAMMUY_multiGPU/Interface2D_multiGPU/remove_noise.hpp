#ifndef INTERFACE_REMOVE_NOISE_H
#define INTERFACE_REMOVE_NOISE_H


#include <thrust/device_vector.h>
#include "const.hpp"
#include "../IdealMHD2D_multiGPU/const.hpp"
#include "../IdealMHD2D_multiGPU/conservation_parameter_struct.hpp"
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

    int indexOfInterfaceStartInMHD_lower;
    int indexOfInterfaceStartInPIC_lower;
    int indexOfInterfaceStartInMHD_upper;
    int indexOfInterfaceStartInPIC_upper;
    int interfaceLength; 
    int windowSize;
    int nxInterface; 
    int nyInterface;

    thrust::device_vector<MagneticField> tmpB;
    thrust::device_vector<ElectricField> tmpE;
    thrust::device_vector<CurrentField> tmpCurrent;
    thrust::device_vector<ZerothMoment> tmpZerothMoment;
    thrust::device_vector<FirstMoment> tmpFirstMoment;
    thrust::device_vector<ConservationParameter> tmpU;

public:
    InterfaceNoiseRemover2D(
        IdealMHD2DMPI::MPIInfo& mPIInfoMHD, 
        PIC2DMPI::MPIInfo& mPIInfoPIC, 
        int indexOfInterfaceStartInMHD_lower, 
        int indexOfInterfaceStartInPIC_lower, 
        int indexOfInterfaceStartInMHD_upper, 
        int indexOfInterfaceStartInPIC_upper, 
        int interfaceLength, 
        int windowSizeForConvolution, 
        int nxInterface, 
        int nyInterface
    );


    void convolve_magneticField(
        thrust::device_vector<MagneticField>& B, 
        bool isLower, bool isUpper
    );

    void convolve_electricField(
        thrust::device_vector<ElectricField>& E, 
        bool isLower, bool isUpper
    );

    void convolve_currentField(
        thrust::device_vector<CurrentField>& current, 
        bool isLower, bool isUpper
    );

    void convolveMoments(
        thrust::device_vector<ZerothMoment>& zerothMomentIon, 
        thrust::device_vector<ZerothMoment>& zerothMomentElectron, 
        thrust::device_vector<FirstMoment>& firstMomentIon, 
        thrust::device_vector<FirstMoment>& firstMomentElectron, 
        bool isLower, bool isUpper
    );

    void convolveU(
        thrust::device_vector<ConservationParameter>& U, 
        bool isLower, bool isUpper
    );

private:

};


#endif

