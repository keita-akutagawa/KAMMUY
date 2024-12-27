#include "boundary.hpp"


void BoundaryPIC::periodicBoundaryZerothMoment_x(
    thrust::device_vector<ZerothMoment>& zerothMoment
)
{
    MPI_Barrier(MPI_COMM_WORLD);
}


void BoundaryPIC::periodicBoundaryZerothMoment_y(
    thrust::device_vector<ZerothMoment>& zerothMoment
)
{
    MPI_Barrier(MPI_COMM_WORLD);
}


void freeBoundaryZerothMoment_x(
    thrust::device_vector<ZerothMoment>& zerothMoment
)
{
    std::cout << "Not writtern yet. Finish your calculation now!" << std::endl;
}


__global__ void freeBoundaryZerothMoment_y_kernel(
    ZerothMoment* zerothMoment,  
    int localSizeX, int localSizeY, 
    int buffer, 
    PIC2DMPI::MPIInfo* device_mPIInfo
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    PIC2DMPI::MPIInfo mPIInfo = *device_mPIInfo;

    if (i < localSizeX) {
        if (mPIInfo.localGridY == 0) {
            for (int j = 0; j < buffer; j++) {
                zerothMoment[j + localSizeY * i] = zerothMoment[buffer + localSizeY * i];
            }
        }

        if (mPIInfo.localGridY == mPIInfo.gridY - 1) {
            for (int j = 0; j < buffer; j++) {
                zerothMoment[localSizeY - 1 - j + localSizeY * i] = zerothMoment[localSizeY - 1 - buffer + localSizeY * i];
            }
        }
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
        mPIInfo.localSizeX, mPIInfo.localSizeY, 
        mPIInfo.buffer, 
        device_mPIInfo
    );
    cudaDeviceSynchronize();
}


//////////////////////////////////////////////////


void BoundaryPIC::periodicBoundaryFirstMoment_x(
    thrust::device_vector<FirstMoment>& firstMoment
)
{
    MPI_Barrier(MPI_COMM_WORLD);
}


void BoundaryPIC::periodicBoundaryFirstMoment_y(
    thrust::device_vector<FirstMoment>& firstMoment
)
{
    MPI_Barrier(MPI_COMM_WORLD);
}


void freeBoundaryFirstMoment_x(
    thrust::device_vector<FirstMoment>& firstMoment
)
{
    std::cout << "Not writtern yet. Finish your calculation now!" << std::endl;
}


__global__ void freeBoundaryFirstMoment_y_kernel(
    FirstMoment* firstMoment,  
    int localSizeX, int localSizeY, 
    int buffer, 
    PIC2DMPI::MPIInfo* device_mPIInfo
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    PIC2DMPI::MPIInfo mPIInfo = *device_mPIInfo;

    if (i < localSizeX) {
        if (mPIInfo.localGridY == 0) {
            for (int j = 0; j < buffer; j++) {
                firstMoment[j + localSizeY * i] = firstMoment[buffer + localSizeY * i];
            }
        }

        if (mPIInfo.localGridY == mPIInfo.gridY - 1) {
            for (int j = 0; j < buffer; j++) {
                firstMoment[localSizeY - 1 - j + localSizeY * i] = firstMoment[localSizeY - 1 - buffer + localSizeY * i];
            }
        }
    }
}

void BoundaryPIC::freeBoundaryFirstMoment_y(
    thrust::device_vector<FirstMoment>& firstMoment
)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (mPIInfo.localSizeX + threadsPerBlock - 1) / threadsPerBlock;

    freeBoundaryFirstMoment_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(firstMoment.data()), 
        mPIInfo.localSizeX, mPIInfo.localSizeY, 
        mPIInfo.buffer, 
        device_mPIInfo
    );
    cudaDeviceSynchronize();
}


