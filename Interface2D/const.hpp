
namespace Interface2DConst
{
    extern const float PI;

    extern const int windowSizeForRemoveNoiseByConvolution;

    extern const unsigned long long reloadParticlesTotalNumIon;
    extern const unsigned long long reloadParticlesTotalNumElectron;


    extern __constant__ float device_PI;

    extern __constant__ int device_windowSizeForRemoveNoiseByConvolution;

    extern __constant__ unsigned long long device_reloadParticlesTotalNumIon;
    extern __constant__ unsigned long long device_reloadParticlesTotalNumElectron;

    void initializeDeviceConstants();

}

