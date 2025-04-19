#include "boundary.hpp"



void BoundaryPIC::periodicBoundaryZerothMoment_x(
    thrust::device_vector<ZerothMoment>& zerothMoment
)
{
    MPI_Barrier(MPI_COMM_WORLD); 
    PIC2DMPI::sendrecv_zerothMoment_x(
        zerothMoment, 
        sendZerothMomentLeft, sendZerothMomentRight, 
        recvZerothMomentLeft, recvZerothMomentRight, 
        mPIInfo
    ); 
}


void freeBoundaryZerothMoment_x(
    thrust::device_vector<ZerothMoment>& zerothMoment
)
{
    std::cout << "Not writtern yet. Finish your calculation now!" << std::endl;
}


__global__ void freeBoundaryZerothMoment_y_kernel(
    ZerothMoment* zerothMoment,  
    int localSizeX, 
    int buffer
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < localSizeX) {
        zerothMoment[0 + PIC2DConst::device_ny * i] = zerothMoment[1 + PIC2DConst::device_ny * i];
        zerothMoment[PIC2DConst::device_ny - 1 + PIC2DConst::device_ny * i] = zerothMoment[PIC2DConst::device_ny - 2 + PIC2DConst::device_ny * i];
    }
}

void BoundaryPIC::freeBoundaryZerothMoment_y(
    thrust::device_vector<ZerothMoment>& zerothMoment
)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (mPIInfo.localSizeX + threadsPerBlock - 1) / threadsPerBlock;

    freeBoundaryZerothMoment_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(zerothMoment.data()), 
        mPIInfo.localSizeX,
        mPIInfo.buffer
    );
    cudaDeviceSynchronize();
}


//////////////////////////////////////////////////


void BoundaryPIC::periodicBoundaryFirstMoment_x(
    thrust::device_vector<FirstMoment>& firstMoment
)
{
    MPI_Barrier(MPI_COMM_WORLD); 

    PIC2DMPI::sendrecv_firstMoment_x(
        firstMoment, 
        sendFirstMomentLeft, sendFirstMomentRight, 
        recvFirstMomentLeft, recvFirstMomentRight, 
        mPIInfo
    ); 
}


void freeBoundaryFirstMoment_x(
    thrust::device_vector<FirstMoment>& firstMoment
)
{
    std::cout << "Not writtern yet. Finish your calculation now!" << std::endl;
}


__global__ void freeBoundaryFirstMoment_y_kernel(
    FirstMoment* firstMoment,  
    int localSizeX,
    int buffer
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < localSizeX) {
        firstMoment[0 + PIC2DConst::device_ny * i] = firstMoment[1 + PIC2DConst::device_ny * i];
        firstMoment[PIC2DConst::device_ny - 1 + PIC2DConst::device_ny * i] = firstMoment[PIC2DConst::device_ny - 2 + PIC2DConst::device_ny * i];
    }
}

void BoundaryPIC::freeBoundaryFirstMoment_y(
    thrust::device_vector<FirstMoment>& firstMoment
)
{
    MPI_Barrier(MPI_COMM_WORLD); 
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (mPIInfo.localSizeX + threadsPerBlock - 1) / threadsPerBlock;

    freeBoundaryFirstMoment_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(firstMoment.data()), 
        mPIInfo.localSizeX, 
        mPIInfo.buffer
    );
    cudaDeviceSynchronize();
}


//////////////////////////////////////////////////


void BoundaryPIC::periodicBoundarySecondMoment_x(
    thrust::device_vector<SecondMoment>& secondMoment
)
{
    MPI_Barrier(MPI_COMM_WORLD); 

    PIC2DMPI::sendrecv_secondMoment_x(
        secondMoment, 
        sendSecondMomentLeft, sendSecondMomentRight, 
        recvSecondMomentLeft, recvSecondMomentRight, 
        mPIInfo
    ); 
}


void freeBoundarySecondMoment_x(
    thrust::device_vector<SecondMoment>& secondMoment
)
{
    std::cout << "Not writtern yet. Finish your calculation now!" << std::endl;
}


__global__ void freeBoundarySecondMoment_y_kernel(
    SecondMoment* secondMoment,  
    int localSizeX,
    int buffer
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < localSizeX) {
        secondMoment[0 + PIC2DConst::device_ny * i] = secondMoment[1 + PIC2DConst::device_ny * i];
        secondMoment[PIC2DConst::device_ny - 1 + PIC2DConst::device_ny * i] = secondMoment[PIC2DConst::device_ny - 2 + PIC2DConst::device_ny * i];
    }
}

void BoundaryPIC::freeBoundarySecondMoment_y(
    thrust::device_vector<SecondMoment>& secondMoment
)
{
    MPI_Barrier(MPI_COMM_WORLD); 
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (mPIInfo.localSizeX + threadsPerBlock - 1) / threadsPerBlock;

    freeBoundarySecondMoment_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(secondMoment.data()), 
        mPIInfo.localSizeX, 
        mPIInfo.buffer
    );
    cudaDeviceSynchronize();
}

