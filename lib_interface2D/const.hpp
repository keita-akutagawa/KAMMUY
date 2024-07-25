#ifndef CONST_STRUCT_H
#define CONST_STRUCT_H


namespace Interface2DConst
{
    extern const float PI;

    extern const int interfaceLength;
    extern const int indexOfInterfaceStartInMHD;
    extern const int indexOfInterfaceStartInPIC;

    extern const int windowSizeForRemoveNoiseByConvolution;


    extern __constant__ float device_PI;

    extern __constant__ int interfaceLength;
    extern __constant__ int indexOfInterfaceStartInMHD;
    extern __constant__ int indexOfInterfaceStartInPIC;

    extern __constant__ int windowSizeForRemoveNoiseByConvolution;

    void initializeDeviceConstants();

}

#endif