#ifndef CONST_STRUCT_H
#define CONST_STRUCT_H

#include <string>


namespace IdealMHD2DConst
{
    extern const std::string MTXfilename;
    extern const std::string jsonFilenameForSolver;

    extern const double EPS;
    extern const double PI;

    extern double eta; 
    extern double viscosity; 

    extern const double B0; 
    extern const double rho0; 
    extern const double p0; 

    extern const double dx;
    extern const double xmin;
    extern const double xmax;
    extern const int nx;

    extern const double dy;
    extern const double ymin;
    extern const double ymax;
    extern const int ny;

    extern const double CFL;
    extern const double gamma;

    extern double dt;

    extern const int totalStep;
    extern double totalTime;


    extern __constant__ double device_EPS;
    extern __constant__ double device_PI;

    extern __device__ double device_eta; 
    extern __device__ double device_viscosity; 

    extern __constant__ double device_B0; 
    extern __constant__ double device_rho0; 
    extern __constant__ double device_p0; 

    extern __constant__ double device_dx;
    extern __constant__ double device_xmin;
    extern __constant__ double device_xmax;
    extern __constant__ int device_nx;

    extern __constant__ double device_dy;
    extern __constant__ double device_ymin;
    extern __constant__ double device_ymax;
    extern __constant__ int device_ny;

    extern __constant__ double device_CFL;
    extern __constant__ double device_gamma;

    extern __device__ double device_dt;


    void initializeDeviceConstants();
    
}

#endif


