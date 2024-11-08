#include "const.hpp"


using namespace Interface2DConst;


void initializeDeviceConstants_Interface()
{
    cudaMemcpyToSymbol(device_EPS, &EPS, sizeof(double));
    cudaMemcpyToSymbol(device_PI, &PI, sizeof(double));

    cudaMemcpyToSymbol(device_windowSizeForConvolution, &windowSizeForConvolution, sizeof(int));

    cudaMemcpyToSymbol(device_reloadParticlesTotalNumIon, &reloadParticlesTotalNumIon, sizeof(unsigned long long));
    cudaMemcpyToSymbol(device_reloadParticlesTotalNumElectron, &reloadParticlesTotalNumElectron, sizeof(unsigned long long));
}


