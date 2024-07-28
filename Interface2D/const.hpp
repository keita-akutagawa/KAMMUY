
namespace Interface2DConst
{
    extern const float PI;

    extern const int interfaceLength;
    extern const int indexOfInterfaceStartInMHD;
    extern const int indexOfInterfaceStartInPIC;

    extern const int windowSizeForRemoveNoiseByConvolution;

    extern const unsigned long long reloadParticlesTotalNumIon;
    extern const unsigned long long reloadParticlesTotalNumElectron;


    extern __constant__ float device_PI;

    extern __constant__ int device_interfaceLength;
    extern __constant__ int device_indexOfInterfaceStartInMHD;
    extern __constant__ int device_indexOfInterfaceStartInPIC;

    extern __constant__ int device_windowSizeForRemoveNoiseByConvolution;

    extern __constant__ unsigned long long device_reloadParticlesTotalNumIon;
    extern __constant__ unsigned long long device_reloadParticlesTotalNumElectron;

    void initializeDeviceConstants();

}

