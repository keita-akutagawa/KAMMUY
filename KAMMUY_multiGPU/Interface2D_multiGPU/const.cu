#include "const.hpp"


void Interface2DConst::initializeDeviceConstants_Interface()
{
    cudaMemcpyToSymbol(device_EPS, &EPS, sizeof(double));
    cudaMemcpyToSymbol(device_PI, &PI, sizeof(double));

    cudaMemcpyToSymbol(device_windowSizeForConvolution, &windowSizeForConvolution, sizeof(int));

    cudaMemcpyToSymbol(device_reloadParticlesTotalNumIon, &reloadParticlesTotalNumIon, sizeof(unsigned long long));
}


