#include "const.hpp"


void initializeDeviceConstants()
{
    cudaMemcpyToSymbol(device_PI, &PI, sizeof(float));

    cudaMemcpyToSymbol(device_interfaceLength, &interfaceLength, sizeof(int));
    cudaMemcpyToSymbol(device_indexOfInterfaceStartInMHD, &indexOfInterfaceStartInMHD, sizeof(int));
    cudaMemcpyToSymbol(device_indexOfInterfaceStartInPIC, &indexOfInterfaceStartInPIC, sizeof(int));

    cudaMemcpyToSymbol(device_windowSizeForRemoveNoiseByConvolution, &windowSizeForRemoveNoiseByConvolution, sizeof(int));
}


