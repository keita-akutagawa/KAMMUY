#include "boundary.hpp"


using namespace IdealMHD2DConst;

__global__
void symmetricBoundaryX2nd_kernel(ConservationParameter* U)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < device_ny_MHD) {
        U[j + 0 * device_ny_MHD].rho  = U[j + 5 * device_ny_MHD].rho;
        U[j + 0 * device_ny_MHD].rhoU = U[j + 5 * device_ny_MHD].rhoU;
        U[j + 0 * device_ny_MHD].rhoV = U[j + 5 * device_ny_MHD].rhoV;
        U[j + 0 * device_ny_MHD].rhoW = U[j + 5 * device_ny_MHD].rhoW;
        U[j + 0 * device_ny_MHD].bY   = U[j + 5 * device_ny_MHD].bY;
        U[j + 0 * device_ny_MHD].bZ   = U[j + 5 * device_ny_MHD].bZ;
        U[j + 0 * device_ny_MHD].e    = U[j + 5 * device_ny_MHD].e;
        U[j + 1 * device_ny_MHD].rho  = U[j + 4 * device_ny_MHD].rho;
        U[j + 1 * device_ny_MHD].rhoU = U[j + 4 * device_ny_MHD].rhoU;
        U[j + 1 * device_ny_MHD].rhoV = U[j + 4 * device_ny_MHD].rhoV;
        U[j + 1 * device_ny_MHD].rhoW = U[j + 4 * device_ny_MHD].rhoW;
        U[j + 1 * device_ny_MHD].bY   = U[j + 4 * device_ny_MHD].bY;
        U[j + 1 * device_ny_MHD].bZ   = U[j + 4 * device_ny_MHD].bZ;
        U[j + 1 * device_ny_MHD].e    = U[j + 4 * device_ny_MHD].e;
        U[j + 2 * device_ny_MHD].rho  = U[j + 3 * device_ny_MHD].rho;
        U[j + 2 * device_ny_MHD].rhoU = U[j + 3 * device_ny_MHD].rhoU;
        U[j + 2 * device_ny_MHD].rhoV = U[j + 3 * device_ny_MHD].rhoV;
        U[j + 2 * device_ny_MHD].rhoW = U[j + 3 * device_ny_MHD].rhoW;
        U[j + 2 * device_ny_MHD].bY   = U[j + 3 * device_ny_MHD].bY;
        U[j + 2 * device_ny_MHD].bZ   = U[j + 3 * device_ny_MHD].bZ;
        U[j + 2 * device_ny_MHD].e    = U[j + 3 * device_ny_MHD].e;


        U[j + (device_nx_MHD - 1) * device_ny_MHD].rho  = U[j + (device_nx_MHD - 6) * device_ny_MHD].rho;
        U[j + (device_nx_MHD - 1) * device_ny_MHD].rhoU = U[j + (device_nx_MHD - 6) * device_ny_MHD].rhoU;
        U[j + (device_nx_MHD - 1) * device_ny_MHD].rhoV = U[j + (device_nx_MHD - 6) * device_ny_MHD].rhoV;
        U[j + (device_nx_MHD - 1) * device_ny_MHD].rhoW = U[j + (device_nx_MHD - 6) * device_ny_MHD].rhoW;
        U[j + (device_nx_MHD - 1) * device_ny_MHD].bY   = U[j + (device_nx_MHD - 6) * device_ny_MHD].bY;
        U[j + (device_nx_MHD - 1) * device_ny_MHD].bZ   = U[j + (device_nx_MHD - 6) * device_ny_MHD].bZ;
        U[j + (device_nx_MHD - 1) * device_ny_MHD].e    = U[j + (device_nx_MHD - 6) * device_ny_MHD].e;
        U[j + (device_nx_MHD - 2) * device_ny_MHD].rho  = U[j + (device_nx_MHD - 5) * device_ny_MHD].rho;
        U[j + (device_nx_MHD - 2) * device_ny_MHD].rhoU = U[j + (device_nx_MHD - 5) * device_ny_MHD].rhoU;
        U[j + (device_nx_MHD - 2) * device_ny_MHD].rhoV = U[j + (device_nx_MHD - 5) * device_ny_MHD].rhoV;
        U[j + (device_nx_MHD - 2) * device_ny_MHD].rhoW = U[j + (device_nx_MHD - 5) * device_ny_MHD].rhoW;
        U[j + (device_nx_MHD - 2) * device_ny_MHD].bY   = U[j + (device_nx_MHD - 5) * device_ny_MHD].bY;
        U[j + (device_nx_MHD - 2) * device_ny_MHD].bZ   = U[j + (device_nx_MHD - 5) * device_ny_MHD].bZ;
        U[j + (device_nx_MHD - 2) * device_ny_MHD].e    = U[j + (device_nx_MHD - 5) * device_ny_MHD].e;
        U[j + (device_nx_MHD - 3) * device_ny_MHD].rho  = U[j + (device_nx_MHD - 4) * device_ny_MHD].rho;
        U[j + (device_nx_MHD - 3) * device_ny_MHD].rhoU = U[j + (device_nx_MHD - 4) * device_ny_MHD].rhoU;
        U[j + (device_nx_MHD - 3) * device_ny_MHD].rhoV = U[j + (device_nx_MHD - 4) * device_ny_MHD].rhoV;
        U[j + (device_nx_MHD - 3) * device_ny_MHD].rhoW = U[j + (device_nx_MHD - 4) * device_ny_MHD].rhoW;
        U[j + (device_nx_MHD - 3) * device_ny_MHD].bY   = U[j + (device_nx_MHD - 4) * device_ny_MHD].bY;
        U[j + (device_nx_MHD - 3) * device_ny_MHD].bZ   = U[j + (device_nx_MHD - 4) * device_ny_MHD].bZ;
        U[j + (device_nx_MHD - 3) * device_ny_MHD].e    = U[j + (device_nx_MHD - 4) * device_ny_MHD].e;
    }
}

__global__
void symmetricBoundaryX2ndBX_kernel(ConservationParameter* U)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (0 < j && j < device_ny_MHD) {
        U[j + 2 * device_ny_MHD].bX = U[j + 3 * device_ny_MHD].bX 
                                + (U[j + 3 * device_ny_MHD].bY - U[j - 1 + 3 * device_ny_MHD].bY) / device_dy_MHD * device_dx_MHD;
        U[j + 1 * device_ny_MHD].bX = U[j + 2 * device_ny_MHD].bX 
                                + (U[j + 2 * device_ny_MHD].bY - U[j - 1 + 2 * device_ny_MHD].bY) / device_dy_MHD * device_dx_MHD;
        U[j + 0 * device_ny_MHD].bX = U[j + 1 * device_ny_MHD].bX 
                                + (U[j + 1 * device_ny_MHD].bY - U[j - 1 + 1 * device_ny_MHD].bY) / device_dy_MHD * device_dx_MHD;
        
        U[j + (device_nx_MHD - 4) * device_ny_MHD].bX = U[j + (device_nx_MHD - 5) * device_ny_MHD].bX 
                                              - (U[j + (device_nx_MHD - 4) * device_ny_MHD].bY - U[j - 1 + (device_nx_MHD - 4) * device_ny_MHD].bY) / device_dy_MHD * device_dx_MHD;
        U[j + (device_nx_MHD - 3) * device_ny_MHD].bX = U[j + (device_nx_MHD - 4) * device_ny_MHD].bX 
                                              - (U[j + (device_nx_MHD - 3) * device_ny_MHD].bY - U[j - 1 + (device_nx_MHD - 3) * device_ny_MHD].bY) / device_dy_MHD * device_dx_MHD;
        U[j + (device_nx_MHD - 2) * device_ny_MHD].bX = U[j + (device_nx_MHD - 3) * device_ny_MHD].bX 
                                              - (U[j + (device_nx_MHD - 2) * device_ny_MHD].bY - U[j - 1 + (device_nx_MHD - 2) * device_ny_MHD].bY) / device_dy_MHD * device_dx_MHD;
        U[j + (device_nx_MHD - 1) * device_ny_MHD].bX = U[j + (device_nx_MHD - 2) * device_ny_MHD].bX 
                                              - (U[j + (device_nx_MHD - 1) * device_ny_MHD].bY - U[j - 1 + (device_nx_MHD - 1) * device_ny_MHD].bY) / device_dy_MHD * device_dx_MHD;

    }

    if (j == 0) {
        U[j + 2 * device_ny_MHD].bX = U[j + 3 * device_ny_MHD].bX;
        U[j + 1 * device_ny_MHD].bX = U[j + 2 * device_ny_MHD].bX;
        U[j + 0 * device_ny_MHD].bX = U[j + 1 * device_ny_MHD].bX;
        
        U[j + (device_nx_MHD - 4) * device_ny_MHD].bX = U[j + (device_nx_MHD - 5) * device_ny_MHD].bX;
        U[j + (device_nx_MHD - 3) * device_ny_MHD].bX = U[j + (device_nx_MHD - 4) * device_ny_MHD].bX;
        U[j + (device_nx_MHD - 2) * device_ny_MHD].bX = U[j + (device_nx_MHD - 3) * device_ny_MHD].bX;
        U[j + (device_nx_MHD - 1) * device_ny_MHD].bX = U[j + (device_nx_MHD - 2) * device_ny_MHD].bX;

    }

}

<<<<<<< HEAD
=======

>>>>>>> eac0c155459d54f9a2b54a35825e563d97fca8c6
void BoundaryMHD::symmetricBoundaryX2nd(
    thrust::device_vector<ConservationParameter>& U
)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (ny_MHD + threadsPerBlock - 1) / threadsPerBlock;

    symmetricBoundaryX2nd_kernel<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(U.data()));

    cudaDeviceSynchronize();

    symmetricBoundaryX2ndBX_kernel<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(U.data()));

    cudaDeviceSynchronize();
}


__global__
void symmetricBoundaryY2nd_kernel(ConservationParameter* U)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < device_nx_MHD) {
        U[0 + i * device_ny_MHD].rho  = U[5 + i * device_ny_MHD].rho;
        U[0 + i * device_ny_MHD].rhoU = U[5 + i * device_ny_MHD].rhoU;
        U[0 + i * device_ny_MHD].rhoV = U[5 + i * device_ny_MHD].rhoV;
        U[0 + i * device_ny_MHD].rhoW = U[5 + i * device_ny_MHD].rhoW;
        U[0 + i * device_ny_MHD].bX   = U[5 + i * device_ny_MHD].bX;
        U[0 + i * device_ny_MHD].bZ   = U[5 + i * device_ny_MHD].bZ;
        U[0 + i * device_ny_MHD].e    = U[5 + i * device_ny_MHD].e;
        U[1 + i * device_ny_MHD].rho  = U[4 + i * device_ny_MHD].rho;
        U[1 + i * device_ny_MHD].rhoU = U[4 + i * device_ny_MHD].rhoU;
        U[1 + i * device_ny_MHD].rhoV = U[4 + i * device_ny_MHD].rhoV;
        U[1 + i * device_ny_MHD].rhoW = U[4 + i * device_ny_MHD].rhoW;
        U[1 + i * device_ny_MHD].bX   = U[4 + i * device_ny_MHD].bX;
        U[1 + i * device_ny_MHD].bZ   = U[4 + i * device_ny_MHD].bZ;
        U[1 + i * device_ny_MHD].e    = U[4 + i * device_ny_MHD].e;
        U[2 + i * device_ny_MHD].rho  = U[3 + i * device_ny_MHD].rho;
        U[2 + i * device_ny_MHD].rhoU = U[3 + i * device_ny_MHD].rhoU;
        U[2 + i * device_ny_MHD].rhoV = U[3 + i * device_ny_MHD].rhoV;
        U[2 + i * device_ny_MHD].rhoW = U[3 + i * device_ny_MHD].rhoW;
        U[2 + i * device_ny_MHD].bX   = U[3 + i * device_ny_MHD].bX;
        U[2 + i * device_ny_MHD].bZ   = U[3 + i * device_ny_MHD].bZ;
        U[2 + i * device_ny_MHD].e    = U[3 + i * device_ny_MHD].e;
        

        U[device_ny_MHD - 1 + i * device_ny_MHD].rho  = U[device_ny_MHD - 6 + i * device_ny_MHD].rho;
        U[device_ny_MHD - 1 + i * device_ny_MHD].rhoU = U[device_ny_MHD - 6 + i * device_ny_MHD].rhoU;
        U[device_ny_MHD - 1 + i * device_ny_MHD].rhoV = U[device_ny_MHD - 6 + i * device_ny_MHD].rhoV;
        U[device_ny_MHD - 1 + i * device_ny_MHD].rhoW = U[device_ny_MHD - 6 + i * device_ny_MHD].rhoW;
        U[device_ny_MHD - 1 + i * device_ny_MHD].bX   = U[device_ny_MHD - 6 + i * device_ny_MHD].bX;
        U[device_ny_MHD - 1 + i * device_ny_MHD].bZ   = U[device_ny_MHD - 6 + i * device_ny_MHD].bZ;
        U[device_ny_MHD - 1 + i * device_ny_MHD].e    = U[device_ny_MHD - 6 + i * device_ny_MHD].e;
        U[device_ny_MHD - 2 + i * device_ny_MHD].rho  = U[device_ny_MHD - 5 + i * device_ny_MHD].rho;
        U[device_ny_MHD - 2 + i * device_ny_MHD].rhoU = U[device_ny_MHD - 5 + i * device_ny_MHD].rhoU;
        U[device_ny_MHD - 2 + i * device_ny_MHD].rhoV = U[device_ny_MHD - 5 + i * device_ny_MHD].rhoV;
        U[device_ny_MHD - 2 + i * device_ny_MHD].rhoW = U[device_ny_MHD - 5 + i * device_ny_MHD].rhoW;
        U[device_ny_MHD - 2 + i * device_ny_MHD].bX   = U[device_ny_MHD - 5 + i * device_ny_MHD].bX;
        U[device_ny_MHD - 2 + i * device_ny_MHD].bZ   = U[device_ny_MHD - 5 + i * device_ny_MHD].bZ;
        U[device_ny_MHD - 2 + i * device_ny_MHD].e    = U[device_ny_MHD - 5 + i * device_ny_MHD].e;
        U[device_ny_MHD - 3 + i * device_ny_MHD].rho  = U[device_ny_MHD - 4 + i * device_ny_MHD].rho;
        U[device_ny_MHD - 3 + i * device_ny_MHD].rhoU = U[device_ny_MHD - 4 + i * device_ny_MHD].rhoU;
        U[device_ny_MHD - 3 + i * device_ny_MHD].rhoV = U[device_ny_MHD - 4 + i * device_ny_MHD].rhoV;
        U[device_ny_MHD - 3 + i * device_ny_MHD].rhoW = U[device_ny_MHD - 4 + i * device_ny_MHD].rhoW;
        U[device_ny_MHD - 3 + i * device_ny_MHD].bX   = U[device_ny_MHD - 4 + i * device_ny_MHD].bX;
        U[device_ny_MHD - 3 + i * device_ny_MHD].bZ   = U[device_ny_MHD - 4 + i * device_ny_MHD].bZ;
        U[device_ny_MHD - 3 + i * device_ny_MHD].e    = U[device_ny_MHD - 4 + i * device_ny_MHD].e;
    }
}

__global__
void symmetricBoundaryY2ndBY_kernel(ConservationParameter* U)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (0 < i && i < device_nx_MHD) {
        U[2 + i * device_ny_MHD].bY = U[3 + i * device_ny_MHD].bY
                                + (U[3 + i * device_ny_MHD].bX - U[3 + (i - 1) * device_ny_MHD].bX) / device_dx_MHD * device_dy_MHD;
        U[1 + i * device_ny_MHD].bY = U[2 + i * device_ny_MHD].bY
                                + (U[2 + i * device_ny_MHD].bX - U[2 + (i - 1) * device_ny_MHD].bX) / device_dx_MHD * device_dy_MHD;
        U[0 + i * device_ny_MHD].bY = U[1 + i * device_ny_MHD].bY
                                + (U[1 + i * device_ny_MHD].bX - U[1 + (i - 1) * device_ny_MHD].bX) / device_dx_MHD * device_dy_MHD;
        
        U[device_ny_MHD - 4 + i * device_ny_MHD].bY = U[device_ny_MHD - 5 + i * device_ny_MHD].bY
                                            - (U[device_ny_MHD - 4 + i * device_ny_MHD].bX - U[device_ny_MHD - 4 + (i - 1) * device_ny_MHD].bX) / device_dx_MHD * device_dy_MHD;
        U[device_ny_MHD - 3 + i * device_ny_MHD].bY = U[device_ny_MHD - 4 + i * device_ny_MHD].bY
                                            - (U[device_ny_MHD - 3 + i * device_ny_MHD].bX - U[device_ny_MHD - 3 + (i - 1) * device_ny_MHD].bX) / device_dx_MHD * device_dy_MHD;
        U[device_ny_MHD - 2 + i * device_ny_MHD].bY = U[device_ny_MHD - 3 + i * device_ny_MHD].bY
                                            - (U[device_ny_MHD - 2 + i * device_ny_MHD].bX - U[device_ny_MHD - 2 + (i - 1) * device_ny_MHD].bX) / device_dx_MHD * device_dy_MHD;
        U[device_ny_MHD - 1 + i * device_ny_MHD].bY = U[device_ny_MHD - 2 + i * device_ny_MHD].bY
                                            - (U[device_ny_MHD - 1 + i * device_ny_MHD].bX - U[device_ny_MHD - 1 + (i - 1) * device_ny_MHD].bX) / device_dx_MHD * device_dy_MHD;
    }

    if (i == 0) {
        U[2 + i * device_ny_MHD].bY = U[3 + i * device_ny_MHD].bY;
        U[1 + i * device_ny_MHD].bY = U[2 + i * device_ny_MHD].bY;
        U[0 + i * device_ny_MHD].bY = U[1 + i * device_ny_MHD].bY;
        
        U[device_ny_MHD - 4 + i * device_ny_MHD].bY = U[device_ny_MHD - 5 + i * device_ny_MHD].bY;
        U[device_ny_MHD - 3 + i * device_ny_MHD].bY = U[device_ny_MHD - 4 + i * device_ny_MHD].bY;
        U[device_ny_MHD - 2 + i * device_ny_MHD].bY = U[device_ny_MHD - 3 + i * device_ny_MHD].bY;
        U[device_ny_MHD - 1 + i * device_ny_MHD].bY = U[device_ny_MHD - 2 + i * device_ny_MHD].bY;
    }
}


void BoundaryMHD::symmetricBoundaryY2nd(
    thrust::device_vector<ConservationParameter>& U
)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (nx_MHD + threadsPerBlock - 1) / threadsPerBlock;

    symmetricBoundaryY2nd_kernel<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(U.data()));

    cudaDeviceSynchronize();

    symmetricBoundaryY2ndBY_kernel<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(U.data()));

    cudaDeviceSynchronize();
}

//////////////////////////////////////////////////

__global__
void periodicBoundaryX2nd_kernel(ConservationParameter* U)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < device_ny_MHD) {
        U[j + 0 * device_ny_MHD].rho  = U[j + (device_nx_MHD-6) * device_ny_MHD].rho;
        U[j + 0 * device_ny_MHD].rhoU = U[j + (device_nx_MHD-6) * device_ny_MHD].rhoU;
        U[j + 0 * device_ny_MHD].rhoV = U[j + (device_nx_MHD-6) * device_ny_MHD].rhoV;
        U[j + 0 * device_ny_MHD].rhoW = U[j + (device_nx_MHD-6) * device_ny_MHD].rhoW;
        U[j + 0 * device_ny_MHD].bX   = U[j + (device_nx_MHD-6) * device_ny_MHD].bX;
        U[j + 0 * device_ny_MHD].bY   = U[j + (device_nx_MHD-6) * device_ny_MHD].bY;
        U[j + 0 * device_ny_MHD].bZ   = U[j + (device_nx_MHD-6) * device_ny_MHD].bZ;
        U[j + 0 * device_ny_MHD].e    = U[j + (device_nx_MHD-6) * device_ny_MHD].e;
        U[j + 1 * device_ny_MHD].rho  = U[j + (device_nx_MHD-5) * device_ny_MHD].rho;
        U[j + 1 * device_ny_MHD].rhoU = U[j + (device_nx_MHD-5) * device_ny_MHD].rhoU;
        U[j + 1 * device_ny_MHD].rhoV = U[j + (device_nx_MHD-5) * device_ny_MHD].rhoV;
        U[j + 1 * device_ny_MHD].rhoW = U[j + (device_nx_MHD-5) * device_ny_MHD].rhoW;
        U[j + 1 * device_ny_MHD].bX   = U[j + (device_nx_MHD-5) * device_ny_MHD].bX;
        U[j + 1 * device_ny_MHD].bY   = U[j + (device_nx_MHD-5) * device_ny_MHD].bY;
        U[j + 1 * device_ny_MHD].bZ   = U[j + (device_nx_MHD-5) * device_ny_MHD].bZ;
        U[j + 1 * device_ny_MHD].e    = U[j + (device_nx_MHD-5) * device_ny_MHD].e;
        U[j + 2 * device_ny_MHD].rho  = U[j + (device_nx_MHD-4) * device_ny_MHD].rho;
        U[j + 2 * device_ny_MHD].rhoU = U[j + (device_nx_MHD-4) * device_ny_MHD].rhoU;
        U[j + 2 * device_ny_MHD].rhoV = U[j + (device_nx_MHD-4) * device_ny_MHD].rhoV;
        U[j + 2 * device_ny_MHD].rhoW = U[j + (device_nx_MHD-4) * device_ny_MHD].rhoW;
        U[j + 2 * device_ny_MHD].bX   = U[j + (device_nx_MHD-4) * device_ny_MHD].bX;
        U[j + 2 * device_ny_MHD].bY   = U[j + (device_nx_MHD-4) * device_ny_MHD].bY;
        U[j + 2 * device_ny_MHD].bZ   = U[j + (device_nx_MHD-4) * device_ny_MHD].bZ;
        U[j + 2 * device_ny_MHD].e    = U[j + (device_nx_MHD-4) * device_ny_MHD].e;

        U[j + (device_nx_MHD-3) * device_ny_MHD].rho  = U[j + 3 * device_ny_MHD].rho;
        U[j + (device_nx_MHD-3) * device_ny_MHD].rhoU = U[j + 3 * device_ny_MHD].rhoU;
        U[j + (device_nx_MHD-3) * device_ny_MHD].rhoV = U[j + 3 * device_ny_MHD].rhoV;
        U[j + (device_nx_MHD-3) * device_ny_MHD].rhoW = U[j + 3 * device_ny_MHD].rhoW;
        U[j + (device_nx_MHD-3) * device_ny_MHD].bX   = U[j + 3 * device_ny_MHD].bX;
        U[j + (device_nx_MHD-3) * device_ny_MHD].bY   = U[j + 3 * device_ny_MHD].bY;
        U[j + (device_nx_MHD-3) * device_ny_MHD].bZ   = U[j + 3 * device_ny_MHD].bZ;
        U[j + (device_nx_MHD-3) * device_ny_MHD].e    = U[j + 3 * device_ny_MHD].e;
        U[j + (device_nx_MHD-2) * device_ny_MHD].rho  = U[j + 4 * device_ny_MHD].rho;
        U[j + (device_nx_MHD-2) * device_ny_MHD].rhoU = U[j + 4 * device_ny_MHD].rhoU;
        U[j + (device_nx_MHD-2) * device_ny_MHD].rhoV = U[j + 4 * device_ny_MHD].rhoV;
        U[j + (device_nx_MHD-2) * device_ny_MHD].rhoW = U[j + 4 * device_ny_MHD].rhoW;
        U[j + (device_nx_MHD-2) * device_ny_MHD].bX   = U[j + 4 * device_ny_MHD].bX;
        U[j + (device_nx_MHD-2) * device_ny_MHD].bY   = U[j + 4 * device_ny_MHD].bY;
        U[j + (device_nx_MHD-2) * device_ny_MHD].bZ   = U[j + 4 * device_ny_MHD].bZ;
        U[j + (device_nx_MHD-2) * device_ny_MHD].e    = U[j + 4 * device_ny_MHD].e;
        U[j + (device_nx_MHD-1) * device_ny_MHD].rho  = U[j + 5 * device_ny_MHD].rho;
        U[j + (device_nx_MHD-1) * device_ny_MHD].rhoU = U[j + 5 * device_ny_MHD].rhoU;
        U[j + (device_nx_MHD-1) * device_ny_MHD].rhoV = U[j + 5 * device_ny_MHD].rhoV;
        U[j + (device_nx_MHD-1) * device_ny_MHD].rhoW = U[j + 5 * device_ny_MHD].rhoW;
        U[j + (device_nx_MHD-1) * device_ny_MHD].bX   = U[j + 5 * device_ny_MHD].bX;
        U[j + (device_nx_MHD-1) * device_ny_MHD].bY   = U[j + 5 * device_ny_MHD].bY;
        U[j + (device_nx_MHD-1) * device_ny_MHD].bZ   = U[j + 5 * device_ny_MHD].bZ;
        U[j + (device_nx_MHD-1) * device_ny_MHD].e    = U[j + 5 * device_ny_MHD].e;
    }
}

void BoundaryMHD::periodicBoundaryX2nd(
    thrust::device_vector<ConservationParameter>& U
)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (ny_MHD + threadsPerBlock - 1) / threadsPerBlock;

    periodicBoundaryX2nd_kernel<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(U.data()));

    cudaDeviceSynchronize();
}


__global__
void periodicBoundaryY2nd_kernel(ConservationParameter* U)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < device_nx_MHD) {
        U[0 + i * device_ny_MHD].rho  = U[device_ny_MHD-6 + i * device_ny_MHD].rho;
        U[0 + i * device_ny_MHD].rhoU = U[device_ny_MHD-6 + i * device_ny_MHD].rhoU;
        U[0 + i * device_ny_MHD].rhoV = U[device_ny_MHD-6 + i * device_ny_MHD].rhoV;
        U[0 + i * device_ny_MHD].rhoW = U[device_ny_MHD-6 + i * device_ny_MHD].rhoW;
        U[0 + i * device_ny_MHD].bX   = U[device_ny_MHD-6 + i * device_ny_MHD].bX;
        U[0 + i * device_ny_MHD].bY   = U[device_ny_MHD-6 + i * device_ny_MHD].bY;
        U[0 + i * device_ny_MHD].bZ   = U[device_ny_MHD-6 + i * device_ny_MHD].bZ;
        U[0 + i * device_ny_MHD].e    = U[device_ny_MHD-6 + i * device_ny_MHD].e;
        U[1 + i * device_ny_MHD].rho  = U[device_ny_MHD-5 + i * device_ny_MHD].rho;
        U[1 + i * device_ny_MHD].rhoU = U[device_ny_MHD-5 + i * device_ny_MHD].rhoU;
        U[1 + i * device_ny_MHD].rhoV = U[device_ny_MHD-5 + i * device_ny_MHD].rhoV;
        U[1 + i * device_ny_MHD].rhoW = U[device_ny_MHD-5 + i * device_ny_MHD].rhoW;
        U[1 + i * device_ny_MHD].bX   = U[device_ny_MHD-5 + i * device_ny_MHD].bX;
        U[1 + i * device_ny_MHD].bY   = U[device_ny_MHD-5 + i * device_ny_MHD].bY;
        U[1 + i * device_ny_MHD].bZ   = U[device_ny_MHD-5 + i * device_ny_MHD].bZ;
        U[1 + i * device_ny_MHD].e    = U[device_ny_MHD-5 + i * device_ny_MHD].e;
        U[2 + i * device_ny_MHD].rho  = U[device_ny_MHD-4 + i * device_ny_MHD].rho;
        U[2 + i * device_ny_MHD].rhoU = U[device_ny_MHD-4 + i * device_ny_MHD].rhoU;
        U[2 + i * device_ny_MHD].rhoV = U[device_ny_MHD-4 + i * device_ny_MHD].rhoV;
        U[2 + i * device_ny_MHD].rhoW = U[device_ny_MHD-4 + i * device_ny_MHD].rhoW;
        U[2 + i * device_ny_MHD].bX   = U[device_ny_MHD-4 + i * device_ny_MHD].bX;
        U[2 + i * device_ny_MHD].bY   = U[device_ny_MHD-4 + i * device_ny_MHD].bY;
        U[2 + i * device_ny_MHD].bZ   = U[device_ny_MHD-4 + i * device_ny_MHD].bZ;
        U[2 + i * device_ny_MHD].e    = U[device_ny_MHD-4 + i * device_ny_MHD].e;

        U[device_ny_MHD-3 + i * device_ny_MHD].rho  = U[3 + i * device_ny_MHD].rho;
        U[device_ny_MHD-3 + i * device_ny_MHD].rhoU = U[3 + i * device_ny_MHD].rhoU;
        U[device_ny_MHD-3 + i * device_ny_MHD].rhoV = U[3 + i * device_ny_MHD].rhoV;
        U[device_ny_MHD-3 + i * device_ny_MHD].rhoW = U[3 + i * device_ny_MHD].rhoW;
        U[device_ny_MHD-3 + i * device_ny_MHD].bX   = U[3 + i * device_ny_MHD].bX;
        U[device_ny_MHD-3 + i * device_ny_MHD].bY   = U[3 + i * device_ny_MHD].bY;
        U[device_ny_MHD-3 + i * device_ny_MHD].bZ   = U[3 + i * device_ny_MHD].bZ;
        U[device_ny_MHD-3 + i * device_ny_MHD].e    = U[3 + i * device_ny_MHD].e;
        U[device_ny_MHD-2 + i * device_ny_MHD].rho  = U[4 + i * device_ny_MHD].rho;
        U[device_ny_MHD-2 + i * device_ny_MHD].rhoU = U[4 + i * device_ny_MHD].rhoU;
        U[device_ny_MHD-2 + i * device_ny_MHD].rhoV = U[4 + i * device_ny_MHD].rhoV;
        U[device_ny_MHD-2 + i * device_ny_MHD].rhoW = U[4 + i * device_ny_MHD].rhoW;
        U[device_ny_MHD-2 + i * device_ny_MHD].bX   = U[4 + i * device_ny_MHD].bX;
        U[device_ny_MHD-2 + i * device_ny_MHD].bY   = U[4 + i * device_ny_MHD].bY;
        U[device_ny_MHD-2 + i * device_ny_MHD].bZ   = U[4 + i * device_ny_MHD].bZ;
        U[device_ny_MHD-2 + i * device_ny_MHD].e    = U[4 + i * device_ny_MHD].e;
        U[device_ny_MHD-1 + i * device_ny_MHD].rho  = U[5 + i * device_ny_MHD].rho;
        U[device_ny_MHD-1 + i * device_ny_MHD].rhoU = U[5 + i * device_ny_MHD].rhoU;
        U[device_ny_MHD-1 + i * device_ny_MHD].rhoV = U[5 + i * device_ny_MHD].rhoV;
        U[device_ny_MHD-1 + i * device_ny_MHD].rhoW = U[5 + i * device_ny_MHD].rhoW;
        U[device_ny_MHD-1 + i * device_ny_MHD].bX   = U[5 + i * device_ny_MHD].bX;
        U[device_ny_MHD-1 + i * device_ny_MHD].bY   = U[5 + i * device_ny_MHD].bY;
        U[device_ny_MHD-1 + i * device_ny_MHD].bZ   = U[5 + i * device_ny_MHD].bZ;
        U[device_ny_MHD-1 + i * device_ny_MHD].e    = U[5 + i * device_ny_MHD].e;
    }
}

void BoundaryMHD::periodicBoundaryY2nd(
    thrust::device_vector<ConservationParameter>& U
)
{
    int threadsPerBlock = 256; 
    int blocksPerGrid = (nx_MHD + threadsPerBlock - 1) / threadsPerBlock; 

    periodicBoundaryY2nd_kernel<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(U.data()));

    cudaDeviceSynchronize();
}


