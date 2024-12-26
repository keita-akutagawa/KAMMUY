#include "boundary.hpp"



//////////

void BoundaryPIC::periodicBoundaryB_x(
    thrust::device_vector<MagneticField>& B
)
{
    MPI_Barrier(MPI_COMM_WORLD);
}


void BoundaryPIC::periodicBoundaryB_y(
    thrust::device_vector<MagneticField>& B
)
{
    MPI_Barrier(MPI_COMM_WORLD);
}


void freeBoundaryB_x(
    thrust::device_vector<MagneticField>& B
)
{
    std::cout << "Not writtern yet. Finish your calculation now!" << std::endl;
}


__global__ void freeBoundaryB_y_kernel(
    MagneticField* B, 
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
                B[j + localSizeY * i] = B[buffer + localSizeY * i];
            }
        }

        if (mPIInfo.localGridY == mPIInfo.gridY - 1) {
            for (int j = 0; j < buffer; j++) {
                B[localSizeY - 1 - j + localSizeY * i] = B[localSizeY - 1 - buffer + localSizeY * i];
            }
        }
    }
}

void BoundaryPIC::freeBoundaryB_y(
    thrust::device_vector<MagneticField>& B
)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (mPIInfo.localSizeX + threadsPerBlock - 1) / threadsPerBlock;

    freeBoundaryB_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(B.data()), 
        mPIInfo.localSizeX, mPIInfo.localSizeY, 
        mPIInfo.buffer, 
        device_mPIInfo
    );
    cudaDeviceSynchronize();
}

//////////////////////////////////////////////////

void BoundaryPIC::periodicBoundaryE_x(
    thrust::device_vector<ElectricField>& E
)
{
    MPI_Barrier(MPI_COMM_WORLD);
}


void BoundaryPIC::periodicBoundaryE_y(
    thrust::device_vector<ElectricField>& E
)
{
    MPI_Barrier(MPI_COMM_WORLD);
}


void freeBoundaryE_x(
    thrust::device_vector<ElectricField>& E
)
{
    std::cout << "Not writtern yet. Finish your calculation now!" << std::endl;
}


__global__ void freeBoundaryE_y_kernel(
    ElectricField* E, 
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
                E[j + localSizeY * i] = E[buffer + localSizeY * i];
            }
        }

        if (mPIInfo.localGridY == mPIInfo.gridY - 1) {
            for (int j = 0; j < buffer; j++) {
                E[localSizeY - 1 - j + localSizeY * i] = E[localSizeY - 1 - buffer + localSizeY * i];
            }
        }
    }
}

void BoundaryPIC::freeBoundaryE_y(
    thrust::device_vector<ElectricField>& E
)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (mPIInfo.localSizeX + threadsPerBlock - 1) / threadsPerBlock;

    freeBoundaryE_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(E.data()), 
        mPIInfo.localSizeX, mPIInfo.localSizeY, 
        mPIInfo.buffer, 
        device_mPIInfo
    );
    cudaDeviceSynchronize();
}

//////////////////////////////////////////////////

void BoundaryPIC::periodicBoundaryCurrent_x(
    thrust::device_vector<CurrentField>& current
)
{
    MPI_Barrier(MPI_COMM_WORLD);
}


void BoundaryPIC::periodicBoundaryCurrent_y(
    thrust::device_vector<CurrentField>& current
)
{
    MPI_Barrier(MPI_COMM_WORLD);
}


void freeBoundaryCurrent_x(
    thrust::device_vector<CurrentField>& current
)
{
    std::cout << "Not writtern yet. Finish your calculation now!" << std::endl;
}


__global__ void freeBoundaryCurrent_y_kernel(
    CurrentField* current,  
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
                current[j + localSizeY * i] = current[buffer + localSizeY * i];
            }
        }

        if (mPIInfo.localGridY == mPIInfo.gridY - 1) {
            for (int j = 0; j < buffer; j++) {
                current[localSizeY - 1 - j + localSizeY * i] = current[localSizeY - 1 - buffer + localSizeY * i];
            }
        }
    }
}

void BoundaryPIC::freeBoundaryCurrent_y(
    thrust::device_vector<CurrentField>& current
)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (mPIInfo.localSizeX + threadsPerBlock - 1) / threadsPerBlock;

    freeBoundaryCurrent_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(current.data()), 
        mPIInfo.localSizeX, mPIInfo.localSizeY, 
        mPIInfo.buffer, 
        device_mPIInfo
    );
    cudaDeviceSynchronize();
}



