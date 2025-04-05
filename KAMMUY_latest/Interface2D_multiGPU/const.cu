#include "const.hpp"


void Interface2DConst::initializeDeviceConstants()
{
    cudaMemcpyToSymbol(device_gridSizeRatio, &gridSizeRatio, sizeof(int));

    cudaMemcpyToSymbol(device_EPS, &EPS, sizeof(double));
    cudaMemcpyToSymbol(device_PI, &PI, sizeof(double));

    cudaMemcpyToSymbol(device_interfaceLength, &interfaceLength, sizeof(int));
    cudaMemcpyToSymbol(device_deltaForInterlockingFunction, &deltaForInterlockingFunction, sizeof(double));
    cudaMemcpyToSymbol(device_indexOfInterfaceStartInMHD, &indexOfInterfaceStartInMHD, sizeof(int));

    cudaMemcpyToSymbol(device_nx, &nx, sizeof(int));
    cudaMemcpyToSymbol(device_ny, &ny, sizeof(int));

    cudaMemcpyToSymbol(device_reloadParticlesTotalNum, &reloadParticlesTotalNum, sizeof(unsigned long long));
}


