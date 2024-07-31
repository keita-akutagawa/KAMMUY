#ifndef CONST_PIC_H
#define CONST_PIC_H

namespace PIC2DConst
{
    extern const float c_PIC;
    extern const float epsilon0_PIC;
    extern const float mu0_PIC;
    extern const float dOfLangdonMarderTypeCorrection_PIC;
    extern const float EPS_PIC;

    extern const int nx_PIC;
    extern const float dx_PIC;
    extern const float xmin_PIC; 
    extern const float xmax_PIC;

    extern const int ny_PIC;
    extern const float dy_PIC;
    extern const float ymin_PIC; 
    extern const float ymax_PIC;

    extern float dt_PIC;

    extern const int numberDensityIon_PIC;
    extern const int numberDensityElectron_PIC;

    extern const unsigned long long totalNumIon_PIC;
    extern const unsigned long long totalNumElectron_PIC;
    extern const unsigned long long totalNumParticles_PIC;

    extern unsigned long long existNumIon_PIC;
    extern unsigned long long existNumElectron_PIC;

    extern const float B0_PIC;

    extern const float mRatio_PIC;
    extern const float mIon_PIC;
    extern const float mElectron_PIC;

    extern const float tRatio_PIC;
    extern const float tIon_PIC;
    extern const float tElectron_PIC;

    extern const float qRatio_PIC;
    extern const float qIon_PIC;
    extern const float qElectron_PIC;

    extern const float omegaPe_PIC;
    extern const float omegaPi_PIC;
    extern const float omegaCe_PIC;
    extern const float omegaCi_PIC;

    extern const float debyeLength_PIC;
    extern const float ionInertialLength_PIC; 

    extern const float vThIon_PIC;
    extern const float vThElectron_PIC;
    extern const float bulkVxIon_PIC;
    extern const float bulkVyIon_PIC;
    extern const float bulkVzIon_PIC;
    extern const float bulkVxElectron_PIC;
    extern const float bulkVyElectron_PIC;
    extern const float bulkVzElectron_PIC;

    extern const int totalStep_PIC;
    extern float totalTime_PIC;



    extern __constant__ float device_c_PIC;
    extern __constant__ float device_epsilon0_PIC;
    extern __constant__ float device_mu0_PIC;
    extern __constant__ float device_dOfLangdonMarderTypeCorrection_PIC;
    extern __constant__ float device_EPS_PIC;

    extern __constant__ int device_nx_PIC;
    extern __constant__ float device_dx_PIC;
    extern __constant__ float device_xmin_PIC; 
    extern __constant__ float device_xmax_PIC;

    extern __constant__ int device_ny_PIC;
    extern __constant__ float device_dy_PIC;
    extern __constant__ float device_ymin_PIC; 
    extern __constant__ float device_ymax_PIC;

    extern __device__ float device_dt_PIC;

    extern __constant__ int device_numberDensityIon_PIC;
    extern __constant__ int device_numberDensityElectron_PIC;

    extern __constant__ unsigned long long device_totalNumIon_PIC;
    extern __constant__ unsigned long long device_totalNumElectron_PIC;
    extern __constant__ unsigned long long device_totalNumParticles_PIC;

    extern __device__ unsigned long long device_existNumIon_PIC;
    extern __device__ unsigned long long device_existNumElectron_PIC;

    extern __constant__ float device_B0_PIC;

    extern __constant__ float device_mRatio_PIC;
    extern __constant__ float device_mIon_PIC;
    extern __constant__ float device_mElectron_PIC;

    extern __constant__ float device_tRatio_PIC;
    extern __constant__ float device_tIon_PIC;
    extern __constant__ float device_tElectron_PIC;

    extern __constant__ float device_qRatio_PIC;
    extern __constant__ float device_qIon_PIC;
    extern __constant__ float device_qElectron_PIC;

    extern __constant__ float device_omegaPe_PIC;
    extern __constant__ float device_omegaPi_PIC;
    extern __constant__ float device_omegaCe_PIC;
    extern __constant__ float device_omegaCi_PIC;

    extern __constant__ float device_debyeLength_PIC;
    extern __constant__ float device_ionInertialLength_PIC; 

    extern __constant__ float device_vThIon_PIC;
    extern __constant__ float device_vThElectron_PIC;
    extern __constant__ float device_bulkVxIon_PIC;
    extern __constant__ float device_bulkVyIon_PIC;
    extern __constant__ float device_bulkVzIon_PIC;
    extern __constant__ float device_bulkVxElectron_PIC;
    extern __constant__ float device_bulkVyElectron_PIC;
    extern __constant__ float device_bulkVzElectron_PIC;

    extern __constant__ int device_totalStep_PIC;
    extern __device__ float device_totalTime_PIC;

}

void initializeDeviceConstants_PIC();


#endif

