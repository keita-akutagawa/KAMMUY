#ifndef CONST_PIC_H
#define CONST_PIC_H


namespace PIC2DConst
{

    extern const double c;
    extern const double epsilon0;
    extern const double mu0;
    extern const double dOfLangdonMarderTypeCorrection;
    extern const double EPS;

    extern const int nx;
    extern const double dx;
    extern const double xmin; 
    extern const double xmax;

    extern const int ny;
    extern const double dy;
    extern const double ymin; 
    extern const double ymax;

    extern double dt;

    extern const int numberDensityIon;
    extern const int numberDensityElectron;

    extern const unsigned long long totalNumIon;
    extern const unsigned long long totalNumElectron;
    extern const unsigned long long totalNumParticles;


    extern const double B0;

    extern const double mRatio;
    extern const double mIon;
    extern const double mElectron;

    extern const double tRatio;
    extern const double tIon;
    extern const double tElectron;

    extern const double qRatio;
    extern const double qIon;
    extern const double qElectron;

    extern const double omegaPe;
    extern const double omegaPi;
    extern const double omegaCe;
    extern const double omegaCi;

    extern const double debyeLength;
    extern const double ionInertialLength; 

    extern const double vThIon;
    extern const double vThElectron;
    extern const double bulkVxIon;
    extern const double bulkVyIon;
    extern const double bulkVzIon;
    extern const double bulkVxElectron;
    extern const double bulkVyElectron;
    extern const double bulkVzElectron;

    extern const int totalStep;
    extern double totalTime;


    extern __constant__ double device_c;
    extern __constant__ double device_epsilon0;
    extern __constant__ double device_mu0;
    extern __constant__ double device_dOfLangdonMarderTypeCorrection;
    extern __constant__ double device_EPS;

    extern __constant__ int device_nx;
    extern __constant__ double device_dx;
    extern __constant__ double device_xmin; 
    extern __constant__ double device_xmax;

    extern __constant__ int device_ny;
    extern __constant__ double device_dy;
    extern __constant__ double device_ymin; 
    extern __constant__ double device_ymax;

    extern __device__ double device_dt;

    extern __constant__ int device_numberDensityIon;
    extern __constant__ int device_numberDensityElectron;

    extern __constant__ double device_B0;

    extern __constant__ double device_mRatio;
    extern __constant__ double device_mIon;
    extern __constant__ double device_mElectron;

    extern __constant__ double device_tRatio;
    extern __constant__ double device_tIon;
    extern __constant__ double device_tElectron;

    extern __constant__ double device_qRatio;
    extern __constant__ double device_qIon;
    extern __constant__ double device_qElectron;

    extern __constant__ double device_omegaPe;
    extern __constant__ double device_omegaPi;
    extern __constant__ double device_omegaCe;
    extern __constant__ double device_omegaCi;

    extern __constant__ double device_debyeLength;
    extern __constant__ double device_ionInertialLength; 

    extern __constant__ double device_vThIon;
    extern __constant__ double device_vThElectron;
    extern __constant__ double device_bulkVxIon;
    extern __constant__ double device_bulkVyIon;
    extern __constant__ double device_bulkVzIon;
    extern __constant__ double device_bulkVxElectron;
    extern __constant__ double device_bulkVyElectron;
    extern __constant__ double device_bulkVzElectron;

    extern __constant__ int device_totalStep;
    extern __device__ double device_totalTime;


    void initializeDeviceConstants();

}

#endif
