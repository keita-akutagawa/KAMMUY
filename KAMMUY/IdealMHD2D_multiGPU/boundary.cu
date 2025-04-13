#include "boundary.hpp"


BoundaryMHD::BoundaryMHD(IdealMHD2DMPI::MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo), 

      sendULeft(IdealMHD2DConst::ny * mPIInfo.buffer), 
      sendURight(IdealMHD2DConst::ny * mPIInfo.buffer), 
      recvULeft(IdealMHD2DConst::ny * mPIInfo.buffer), 
      recvURight(IdealMHD2DConst::ny * mPIInfo.buffer)
{

    cudaMalloc(&device_mPIInfo, sizeof(IdealMHD2DMPI::MPIInfo));
    cudaMemcpy(device_mPIInfo, &mPIInfo, sizeof(IdealMHD2DMPI::MPIInfo), cudaMemcpyHostToDevice);
    
}


void BoundaryMHD::periodicBoundaryX2nd_U(
    thrust::device_vector<ConservationParameter>& U
)
{
    MPI_Barrier(MPI_COMM_WORLD); 
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
    IdealMHD2DMPI::MPIInfo* device_mPIInfo
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    IdealMHD2DMPI::MPIInfo mPIInfo = *device_mPIInfo;

    if (i < localSizeX) {
        for (int buf = 0; buf < mPIInfo.buffer; buf++) {            
            U[buf + i * IdealMHD2DConst::device_ny] = U[IdealMHD2DConst::device_ny - 2 * mPIInfo.buffer + buf + i * IdealMHD2DConst::device_ny];
        }
        
        for (int buf = 0; buf < mPIInfo.buffer; buf++) {
            U[IdealMHD2DConst::device_ny - mPIInfo.buffer + buf + i * IdealMHD2DConst::device_ny] = U[buf + mPIInfo.buffer + i * IdealMHD2DConst::device_ny];
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
        device_mPIInfo
    );
    cudaDeviceSynchronize();
}

///////////////////////

__global__
void wallBoundaryY2nd_U_kernel(
    ConservationParameter* U, 
    int localSizeX, 
    IdealMHD2DMPI::MPIInfo* device_mPIInfo
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    IdealMHD2DMPI::MPIInfo mPIInfo = *device_mPIInfo;

    if (i < localSizeX) {
        int index = 0 + i * IdealMHD2DConst::device_ny;

        double rho, u, v, w, bX, bY, bZ, p, e;
        ConservationParameter wallU;

        rho = U[index + mPIInfo.buffer].rho;
        u   = U[index + mPIInfo.buffer].rhoU / rho; 
        v   = U[index + mPIInfo.buffer].rhoV / rho; 
        w   = U[index + mPIInfo.buffer].rhoW / rho;
        bX  = U[index + mPIInfo.buffer].bX; 
        bY  = U[index + mPIInfo.buffer].bY;
        bZ  = U[index + mPIInfo.buffer].bZ;
        e   = U[index + mPIInfo.buffer].e;
        p   = (IdealMHD2DConst::device_gamma - 1.0)
            * (e - 0.5 * rho * (u * u + v * v + w * w)
            - 0.5 * (bX * bX + bY * bY + bZ * bZ));
        
        wallU.rho = rho;
        wallU.rhoU = rho * 0.0; wallU.rhoV = rho * 0.0; wallU.rhoW = rho * 0.0;
        wallU.bX = bX; wallU.bY = 0.0; wallU.bZ = bZ;
        e = p / (IdealMHD2DConst::device_gamma - 1.0) + 0.5 * rho * (0.0 * 0.0 + 0.0 * 0.0 + 0.0 * 0.0)
        + 0.5 * (bX * bX + 0.0 * 0.0 + bZ * bZ); 
        wallU.e = e;

        for (int buf = 0; buf < mPIInfo.buffer; buf++) {            
            U[index + buf] = wallU;
        }
        

        index = IdealMHD2DConst::device_ny - 1 + i * IdealMHD2DConst::device_ny;

        rho = U[index - mPIInfo.buffer].rho;
        u   = U[index - mPIInfo.buffer].rhoU / rho; 
        v   = U[index - mPIInfo.buffer].rhoV / rho; 
        w   = U[index - mPIInfo.buffer].rhoW / rho;
        bX  = U[index - mPIInfo.buffer].bX; 
        bY  = U[index - mPIInfo.buffer].bY;
        bZ  = U[index - mPIInfo.buffer].bZ;
        e   = U[index - mPIInfo.buffer].e;
        p   = (IdealMHD2DConst::device_gamma - 1.0)
            * (e - 0.5 * rho * (u * u + v * v + w * w)
            - 0.5 * (bX * bX + bY * bY + bZ * bZ));
        
        wallU.rho = rho;
        wallU.rhoU = rho * 0.0; wallU.rhoV = rho * 0.0; wallU.rhoW = rho * 0.0;
        wallU.bX = bX; wallU.bY = 0.0; wallU.bZ = bZ;
        e = p / (IdealMHD2DConst::device_gamma - 1.0) + 0.5 * rho * (0.0 * 0.0 + 0.0 * 0.0 + 0.0 * 0.0)
        + 0.5 * (bX * bX + 0.0 * 0.0 + bZ * bZ); 
        wallU.e = e;

        for (int buf = 0; buf < mPIInfo.buffer; buf++) {
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
        device_mPIInfo
    );
    cudaDeviceSynchronize();
}


__global__
void symmetricBoundaryY2nd_U_kernel(
    ConservationParameter* U, 
    int localSizeX, 
    IdealMHD2DMPI::MPIInfo* device_mPIInfo
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    IdealMHD2DMPI::MPIInfo mPIInfo = *device_mPIInfo;

    if (i < localSizeX) {
        int index = 0 + i * IdealMHD2DConst::device_ny;

        for (int buf = 0; buf < mPIInfo.buffer; buf++) {
            U[index + buf] = U[index + mPIInfo.buffer];
        }
        

        index = IdealMHD2DConst::device_ny - 1 + i * IdealMHD2DConst::device_ny;

        for (int buf = 0; buf < mPIInfo.buffer; buf++) {
            U[index - buf] = U[index - mPIInfo.buffer];
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
        device_mPIInfo
    );
    cudaDeviceSynchronize();
}


