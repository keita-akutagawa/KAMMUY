#include "const.hpp"


using namespace IdealMHD2DConst;

void initializeDeviceConstants() {
    cudaMemcpyToSymbol(device_EPS_MHD, &EPS_MHD, sizeof(double));
    cudaMemcpyToSymbol(device_PI_MHD, &PI_MHD, sizeof(double));

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
