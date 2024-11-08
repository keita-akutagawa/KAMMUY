#ifndef CONST_INTERFACE_H
#define CONST_INTERFACE_H


namespace Interface2DConst
{
    extern const double EPS;
    extern const double PI;

    extern const int windowSizeForConvolution;

    extern const unsigned long long reloadParticlesTotalNum;


    extern __constant__ double device_EPS;
    extern __constant__ double device_PI;

    extern __constant__ int device_windowSizeForConvolution;

    extern __constant__ unsigned long long device_reloadParticlesTotalNum;


    void initializeDeviceConstants_Interface();

}


#endif
