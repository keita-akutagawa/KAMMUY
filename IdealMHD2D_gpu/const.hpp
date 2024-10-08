#ifndef CONST_MHD_H
#define CONST_MHD_H

namespace IdealMHD2DConst
{
    extern const double EPS_MHD;
    extern const double PI_MHD;

    extern const double b0_MHD;

    extern const double rho0_MHD;
    extern const double u0_MHD;
    extern const double v0_MHD;
    extern const double w0_MHD;
    extern const double bX0_MHD;
    extern const double bY0_MHD;
    extern const double bZ0_MHD;
    extern const double p0_MHD;
    extern const double e0_MHD;

    extern const double dx_MHD;
    extern const double xmin_MHD;
    extern const double xmax_MHD;
    extern const int nx_MHD;

    extern const double dy_MHD;
    extern const double ymin_MHD;
    extern const double ymax_MHD;
    extern const int ny_MHD;

    extern const double CFL_MHD;
    extern const double gamma_MHD;

    extern double dt_MHD;

    extern const int totalStep_MHD;
    extern double totalTime_MHD;


    extern __constant__ double device_EPS_MHD;
    extern __constant__ double device_PI_MHD;

    extern __constant__ double device_b0_MHD;

    extern __constant__ double device_rho0_MHD;
    extern __constant__ double device_u0_MHD;
    extern __constant__ double device_v0_MHD;
    extern __constant__ double device_w0_MHD;
    extern __constant__ double device_bX0_MHD;
    extern __constant__ double device_bY0_MHD;
    extern __constant__ double device_bZ0_MHD;
    extern __constant__ double device_p0_MHD;
    extern __constant__ double device_e0_MHD;

    extern __constant__ double device_dx_MHD;
    extern __constant__ double device_xmin_MHD;
    extern __constant__ double device_xmax_MHD;
    extern __constant__ int device_nx_MHD;

    extern __constant__ double device_dy_MHD;
    extern __constant__ double device_ymin_MHD;
    extern __constant__ double device_ymax_MHD;
    extern __constant__ int device_ny_MHD;

    extern __constant__ double device_CFL_MHD;
    extern __constant__ double device_gamma_MHD;

    extern __device__ double device_dt_MHD;

}

void initializeDeviceConstants_MHD();

#endif

