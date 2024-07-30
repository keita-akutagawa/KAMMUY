#include "const.hpp"


using namespace Interface2DConst;


void initializeDeviceConstants()
{
    cudaMemcpyToSymbol(device_PI, &PI, sizeof(float));

    cudaMemcpyToSymbol(device_windowSizeForRemoveNoiseByConvolution, &windowSizeForRemoveNoiseByConvolution, sizeof(int));

    cudaMemcpyToSymbol(device_reloadParticlesTotalNumIon, &reloadParticlesTotalNumIon, sizeof(unsigned long long));
    cudaMemcpyToSymbol(device_reloadParticlesTotalNumElectron, &reloadParticlesTotalNumElectron, sizeof(unsigned long long));
}


