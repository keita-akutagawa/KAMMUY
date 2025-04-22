#include "boundary.hpp"


BoundaryMHD::BoundaryMHD(IdealMHD2DMPI::MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo), 

      sendULeft(IdealMHD2DConst::ny * mPIInfo.buffer), 
      sendURight(IdealMHD2DConst::ny * mPIInfo.buffer), 
      recvULeft(IdealMHD2DConst::ny * mPIInfo.buffer), 
      recvURight(IdealMHD2DConst::ny * mPIInfo.buffer)
{
}


void BoundaryMHD::periodicBoundaryX2nd_U(
    thrust::device_vector<ConservationParameter>& U
)
{
    IdealMHD2DMPI::sendrecv_U_x(
        U, 
        sendULeft, sendURight, 
        recvULeft, recvURight, 
        mPIInfo
    ); 
}


__global__
void periodicBoundaryY2nd_U_kernel(
    ConservationParameter* U, 
    int localSizeX, 
    int buffer
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < localSizeX) {
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
    int blocksPerGrid = (mPIInfo.localSizeX + threadsPerBlock - 1) / threadsPerBlock;

    periodicBoundaryY2nd_U_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        mPIInfo.localSizeX, 
        mPIInfo.buffer
    );
    cudaDeviceSynchronize();
}

///////////////////////

__global__
void wallBoundaryY2nd_U_kernel(
    ConservationParameter* U, 
    int localSizeX, 
    int buffer
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

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
    int i = blockIdx.x * blockDim.x + threadIdx.x;

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


