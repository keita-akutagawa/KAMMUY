#ifndef CONST_STRUCT_H
#define CONST_STRUCT_H


extern const float PI;

extern const int interfaceLength;
extern const int indexOfInterfaceStartInMHD;
extern const int indexOfInterfaceStartInPIC;

extern const int windowSizeForRemoveNoiseByConvolution;


extern __constant__ float device_PI;

void initializeDeviceConstants();

#endif