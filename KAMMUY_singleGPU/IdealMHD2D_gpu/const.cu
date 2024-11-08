#include "const.hpp"


using namespace IdealMHD2DConst;

void initializeDeviceConstants_MHD() {
    cudaMemcpyToSymbol(device_EPS_MHD, &EPS_MHD, sizeof(double));
    cudaMemcpyToSymbol(device_PI_MHD, &PI_MHD, sizeof(double));

    cudaMemcpyToSymbol(device_b0_MHD, &b0_MHD, sizeof(double));

    cudaMemcpyToSymbol(device_rho0_MHD, &rho0_MHD, sizeof(double));
    cudaMemcpyToSymbol(device_u0_MHD, &u0_MHD, sizeof(double));
    cudaMemcpyToSymbol(device_v0_MHD, &v0_MHD, sizeof(double));
    cudaMemcpyToSymbol(device_w0_MHD, &w0_MHD, sizeof(double));
    cudaMemcpyToSymbol(device_bX0_MHD, &bX0_MHD, sizeof(double));
    cudaMemcpyToSymbol(device_bY0_MHD, &bY0_MHD, sizeof(double));
    cudaMemcpyToSymbol(device_bZ0_MHD, &bZ0_MHD, sizeof(double));
    cudaMemcpyToSymbol(device_p0_MHD, &p0_MHD, sizeof(double));
    cudaMemcpyToSymbol(device_e0_MHD, &e0_MHD, sizeof(double));

    cudaMemcpyToSymbol(device_dx_MHD, &dx_MHD, sizeof(double));
    cudaMemcpyToSymbol(device_xmin_MHD, &xmin_MHD, sizeof(double));
    cudaMemcpyToSymbol(device_xmax_MHD, &xmax_MHD, sizeof(double));
    cudaMemcpyToSymbol(device_nx_MHD, &nx_MHD, sizeof(int));

    cudaMemcpyToSymbol(device_dy_MHD, &dy_MHD, sizeof(double));
    cudaMemcpyToSymbol(device_ymin_MHD, &ymin_MHD, sizeof(double));
    cudaMemcpyToSymbol(device_ymax_MHD, &ymax_MHD, sizeof(double));
    cudaMemcpyToSymbol(device_ny_MHD, &ny_MHD, sizeof(int));

    cudaMemcpyToSymbol(device_CFL_MHD, &CFL_MHD, sizeof(double));
    cudaMemcpyToSymbol(device_gamma_MHD, &gamma_MHD, sizeof(double));

    cudaMemcpyToSymbol(device_dt_MHD, &dt_MHD, sizeof(double));
}
