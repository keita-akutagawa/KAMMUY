#include "const.hpp"


void IdealMHD2DConst::initializeDeviceConstants() {
    cudaMemcpyToSymbol(device_EPS, &EPS, sizeof(double));
    cudaMemcpyToSymbol(device_PI, &PI, sizeof(double));

    cudaMemcpyToSymbol(device_eta, &eta, sizeof(double));
    cudaMemcpyToSymbol(device_viscosity, &viscosity, sizeof(double));

    cudaMemcpyToSymbol(device_B0, &B0, sizeof(double));
    cudaMemcpyToSymbol(device_rho0, &rho0, sizeof(double));
    cudaMemcpyToSymbol(device_p0, &p0, sizeof(double));

    cudaMemcpyToSymbol(device_dx, &dx, sizeof(double));
    cudaMemcpyToSymbol(device_xmin, &xmin, sizeof(double));
    cudaMemcpyToSymbol(device_xmax, &xmax, sizeof(double));
    cudaMemcpyToSymbol(device_nx, &nx, sizeof(int));

    cudaMemcpyToSymbol(device_dy, &dy, sizeof(double));
    cudaMemcpyToSymbol(device_ymin, &ymin, sizeof(double));
    cudaMemcpyToSymbol(device_ymax, &ymax, sizeof(double));
    cudaMemcpyToSymbol(device_ny, &ny, sizeof(int));

    cudaMemcpyToSymbol(device_CFL, &CFL, sizeof(double));
    cudaMemcpyToSymbol(device_gamma, &gamma, sizeof(double));

    cudaMemcpyToSymbol(device_dt, &dt, sizeof(double));
}
