#ifndef CONST_INTERFACE_H
#define CONST_INTERFACE_H


namespace Interface2DConst
{
    extern const double PI;

    extern const int windowSizeForConvolution;

    extern const unsigned long long reloadParticlesTotalNumIon;
    extern const unsigned long long reloadParticlesTotalNumElectron;


    extern __constant__ double device_PI;

    extern __constant__ int device_windowSizeForConvolution;

    extern __constant__ unsigned long long device_reloadParticlesTotalNumIon;
    extern __constant__ unsigned long long device_reloadParticlesTotalNumElectron;

}

void initializeDeviceConstants_Interface();


#endif
