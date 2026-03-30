#include "boundary.hpp"


BoundaryMHD::BoundaryMHD()
{
}


__global__ void periodicBoundary_x_kernel(
    ConservationParameter* U, 
    const int buffer
)
{
    unsigned long long j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < idealmhd2d_const::device_ny) {
        for (int buf = 0; buf < buffer; buf++) {            
            U[j + buf * IdealMHD2DConst::device_ny] = U[j + (idealmhd2d_const::device_nx - 2 * buffer + buf) * IdealMHD2DConst::device_ny];
        }
        
        for (int buf = 0; buf < buffer; buf++) {
            U[j + (IdealMHD2DConst::device_nx - buffer + buf) * IdealMHD2DConst::device_ny] = U[j + (buffer + buf) * IdealMHD2DConst::device_ny];
        }
    }
}

void BoundaryMHD::periodicBoundary_x(
    thrust::device_vector<ConservationParameter>& U
)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (IdealMHD2DConst::ny + threadsPerBlock - 1) / threadsPerBlock;

    periodicBoundary_x_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        IdealMHD2DConst::buffer
    );
    cudaDeviceSynchronize();
}


__global__ void periodicBoundary_y_kernel(
    ConservationParameter* U, 
    int buffer
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < IdealMHD2DConst::device_nx) {
        for (int buf = 0; buf < buffer; buf++) {            
            U[buf + i * IdealMHD2DConst::device_ny] = U[IdealMHD2DConst::device_ny - 2 * buffer + buf + i * IdealMHD2DConst::device_ny];
        }
        
        for (int buf = 0; buf < buffer; buf++) {
            U[IdealMHD2DConst::device_ny - buffer + buf + i * IdealMHD2DConst::device_ny] = U[buf + buffer + i * IdealMHD2DConst::device_ny];
        }
    }
}


void BoundaryMHD::periodicBoundaryY2nd_U(
    thrust::device_vector<ConservationParameter>& U
)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (IdealMHD2DConst::nx + threadsPerBlock - 1) / threadsPerBlock;

    periodicBoundary_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        IdealMHD2DConst::buffer
    );
    cudaDeviceSynchronize();
}

///////////////////////

__global__ void wallBoundaryY2nd_U_kernel(
    ConservationParameter* U, 
    int localSizeX, 
    int buffer
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < localSizeX) {
        int index = 0 + i * IdealMHD2DConst::device_ny;

        double rho, u, v, w, bX, bY, bZ, p, e;
        ConservationParameter wallU;

        rho = U[index + buffer].rho;
        u   = U[index + buffer].rhoU / rho; 
        v   = U[index + buffer].rhoV / rho; 
        w   = U[index + buffer].rhoW / rho;
        bX  = U[index + buffer].bX; 
        bY  = U[index + buffer].bY;
        bZ  = U[index + buffer].bZ;
        e   = U[index + buffer].e;
        p   = (IdealMHD2DConst::device_gamma - 1.0)
            * (e - 0.5 * rho * (u * u + v * v + w * w)
            - 0.5 * (bX * bX + bY * bY + bZ * bZ));
        
        wallU.rho = rho;
        wallU.rhoU = rho * 0.0; wallU.rhoV = rho * 0.0; wallU.rhoW = rho * 0.0;
        wallU.bX = bX; wallU.bY = 0.0; wallU.bZ = bZ;
        e = p / (IdealMHD2DConst::device_gamma - 1.0) + 0.5 * rho * (0.0 * 0.0 + 0.0 * 0.0 + 0.0 * 0.0)
        + 0.5 * (bX * bX + 0.0 * 0.0 + bZ * bZ); 
        wallU.e = e;

        for (int buf = 0; buf < buffer; buf++) {            
            U[index + buf] = wallU;
        }
        

        index = IdealMHD2DConst::device_ny - 1 + i * IdealMHD2DConst::device_ny;

        rho = U[index - buffer].rho;
        u   = U[index - buffer].rhoU / rho; 
        v   = U[index - buffer].rhoV / rho; 
        w   = U[index - buffer].rhoW / rho;
        bX  = U[index - buffer].bX; 
        bY  = U[index - buffer].bY;
        bZ  = U[index - buffer].bZ;
        e   = U[index - buffer].e;
        p   = (IdealMHD2DConst::device_gamma - 1.0)
            * (e - 0.5 * rho * (u * u + v * v + w * w)
            - 0.5 * (bX * bX + bY * bY + bZ * bZ));
        
        wallU.rho = rho;
        wallU.rhoU = rho * 0.0; wallU.rhoV = rho * 0.0; wallU.rhoW = rho * 0.0;
        wallU.bX = bX; wallU.bY = 0.0; wallU.bZ = bZ;
        e = p / (IdealMHD2DConst::device_gamma - 1.0) + 0.5 * rho * (0.0 * 0.0 + 0.0 * 0.0 + 0.0 * 0.0)
        + 0.5 * (bX * bX + 0.0 * 0.0 + bZ * bZ); 
        wallU.e = e;

        for (int buf = 0; buf < buffer; buf++) {
            U[index - buf] = wallU;
        }
    }
}

void BoundaryMHD::wallBoundaryY2nd_U(
    thrust::device_vector<ConservationParameter>& U
)
{
    MPI_Barrier(MPI_COMM_WORLD); 

    int threadsPerBlock = 256;
    int blocksPerGrid = (mPIInfo.localSizeX + threadsPerBlock - 1) / threadsPerBlock;

    wallBoundaryY2nd_U_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        mPIInfo.localSizeX, 
        mPIInfo.buffer
    );
    cudaDeviceSynchronize();
}


__global__
void symmetricBoundaryY2nd_U_kernel(
    ConservationParameter* U, 
    int localSizeX, 
    int buffer
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < localSizeX) {
        int index = 0 + i * IdealMHD2DConst::device_ny;

        for (int buf = 0; buf < buffer; buf++) {
            U[index + buf] = U[index + buffer];
        }
        

        index = IdealMHD2DConst::device_ny - 1 + i * IdealMHD2DConst::device_ny;

        for (int buf = 0; buf < buffer; buf++) {
            U[index - buf] = U[index - buffer];
        }
    }
}

void BoundaryMHD::symmetricBoundaryY2nd_U(
    thrust::device_vector<ConservationParameter>& U
)
{
    MPI_Barrier(MPI_COMM_WORLD); 
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (mPIInfo.localSizeX + threadsPerBlock - 1) / threadsPerBlock;

    symmetricBoundaryY2nd_U_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        mPIInfo.localSizeX, 
        mPIInfo.buffer
    );
    cudaDeviceSynchronize();
}


