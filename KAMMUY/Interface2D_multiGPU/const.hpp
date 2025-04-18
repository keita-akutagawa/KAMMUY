#ifndef CONST_INTERFACE_H
#define CONST_INTERFACE_H


namespace Interface2DConst
{
    extern const double EPS;
    extern const double PI;

    extern const int gridSizeRatio; 

    extern const int interfaceLength; 
    extern const double deltaForInterlockingFunction; 
    extern const int indexOfInterfaceStartInMHD; 

    extern const int convolutionCount; 

    extern const int nx; 
    extern const int ny; 

    extern const unsigned long long reloadParticlesTotalNum;


    extern __constant__ double device_EPS;
    extern __constant__ double device_PI;

    extern __constant__ int device_gridSizeRatio; 

    extern __constant__ int device_interfaceLength; 
    extern __constant__ double device_deltaForInterlockingFunction; 
    extern __constant__ int device_indexOfInterfaceStartInMHD; 

    extern __constant__ int device_convolutionCount; 

    extern __constant__ int device_nx; 
    extern __constant__ int device_ny; 

    extern __constant__ unsigned long long device_reloadParticlesTotalNum;


    void initializeDeviceConstants();

}


#endif
