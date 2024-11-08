#ifndef CONST_PIC_H
#define CONST_PIC_H

namespace PIC2DConst
{
    extern const double c_PIC;
    extern const double epsilon0_PIC;
    extern const double mu0_PIC;
    extern const double dOfLangdonMarderTypeCorrection_PIC;
    extern const double EPS_PIC;

    extern const int nx_PIC;
    extern const double dx_PIC;
    extern const double xmin_PIC; 
    extern const double xmax_PIC;

    extern const int ny_PIC;
    extern const double dy_PIC;
    extern const double ymin_PIC; 
    extern const double ymax_PIC;

    extern double dt_PIC;

    extern const int numberDensityIon_PIC;
    extern const int numberDensityElectron_PIC;

    extern const unsigned long long totalNumIon_PIC;
    extern const unsigned long long totalNumElectron_PIC;
    extern const unsigned long long totalNumParticles_PIC;

    extern unsigned long long existNumIon_PIC;
    extern unsigned long long existNumElectron_PIC;

    extern const double b0_PIC;

    extern const double mRatio_PIC;
    extern const double mIon_PIC;
    extern const double mElectron_PIC;

    extern const double tRatio_PIC;
    extern const double tIon_PIC;
    extern const double tElectron_PIC;

    extern const double qRatio_PIC;
    extern const double qIon_PIC;
    extern const double qElectron_PIC;

    extern const double omegaPe_PIC;
    extern const double omegaPi_PIC;
    extern const double omegaCe_PIC;
    extern const double omegaCi_PIC;

    extern const double debyeLength_PIC;
    extern const double ionInertialLength_PIC; 

    extern const double vThIon_PIC;
    extern const double vThElectron_PIC;
    extern const double bulkVxIon_PIC;
    extern const double bulkVyIon_PIC;
    extern const double bulkVzIon_PIC;
    extern const double bulkVxElectron_PIC;
    extern const double bulkVyElectron_PIC;
    extern const double bulkVzElectron_PIC;

    extern const int totalStep_PIC;
    extern double totalTime_PIC;



    extern __constant__ double device_c_PIC;
    extern __constant__ double device_epsilon0_PIC;
    extern __constant__ double device_mu0_PIC;
    extern __constant__ double device_dOfLangdonMarderTypeCorrection_PIC;
    extern __constant__ double device_EPS_PIC;

    extern __constant__ int device_nx_PIC;
    extern __constant__ double device_dx_PIC;
    extern __constant__ double device_xmin_PIC; 
    extern __constant__ double device_xmax_PIC;

    extern __constant__ int device_ny_PIC;
    extern __constant__ double device_dy_PIC;
    extern __constant__ double device_ymin_PIC; 
    extern __constant__ double device_ymax_PIC;

    extern __device__ double device_dt_PIC;

    extern __constant__ int device_numberDensityIon_PIC;
    extern __constant__ int device_numberDensityElectron_PIC;

    extern __constant__ unsigned long long device_totalNumIon_PIC;
    extern __constant__ unsigned long long device_totalNumElectron_PIC;
    extern __constant__ unsigned long long device_totalNumParticles_PIC;

    extern __device__ unsigned long long device_existNumIon_PIC;
    extern __device__ unsigned long long device_existNumElectron_PIC;

    extern __constant__ double device_b0_PIC;

    extern __constant__ double device_mRatio_PIC;
    extern __constant__ double device_mIon_PIC;
    extern __constant__ double device_mElectron_PIC;

    extern __constant__ double device_tRatio_PIC;
    extern __constant__ double device_tIon_PIC;
    extern __constant__ double device_tElectron_PIC;

    extern __constant__ double device_qRatio_PIC;
    extern __constant__ double device_qIon_PIC;
    extern __constant__ double device_qElectron_PIC;

    extern __constant__ double device_omegaPe_PIC;
    extern __constant__ double device_omegaPi_PIC;
    extern __constant__ double device_omegaCe_PIC;
    extern __constant__ double device_omegaCi_PIC;

    extern __constant__ double device_debyeLength_PIC;
    extern __constant__ double device_ionInertialLength_PIC; 

    extern __constant__ double device_vThIon_PIC;
    extern __constant__ double device_vThElectron_PIC;
    extern __constant__ double device_bulkVxIon_PIC;
    extern __constant__ double device_bulkVyIon_PIC;
    extern __constant__ double device_bulkVzIon_PIC;
    extern __constant__ double device_bulkVxElectron_PIC;
    extern __constant__ double device_bulkVyElectron_PIC;
    extern __constant__ double device_bulkVzElectron_PIC;

    extern __constant__ int device_totalStep_PIC;
    extern __device__ double device_totalTime_PIC;

}

void initializeDeviceConstants_PIC();


#endif

