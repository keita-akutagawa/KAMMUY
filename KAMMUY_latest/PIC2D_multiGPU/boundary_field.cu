#include "boundary.hpp"



//////////

void BoundaryPIC::periodicBoundaryB_x(
    thrust::device_vector<MagneticField>& B
)
{
    MPI_Barrier(MPI_COMM_WORLD); 

    PIC2DMPI::sendrecv_magneticField_x(
        B, 
        sendMagneticFieldLeft, sendMagneticFieldRight, 
        recvMagneticFieldLeft, recvMagneticFieldRight, 
        mPIInfo
    ); 
}


void freeBoundaryB_x(
    thrust::device_vector<MagneticField>& B
)
{
    std::cout << "Not writtern yet. Finish your calculation now!" << std::endl;
}


__global__ void freeBoundaryB_y_kernel(
    MagneticField* B, 
    int localSizeX, 
    int buffer, 
    PIC2DMPI::MPIInfo* device_mPIInfo
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    PIC2DMPI::MPIInfo mPIInfo = *device_mPIInfo;

    if (i < localSizeX) {
        for (int j = 0; j < buffer; j++) {
            B[j + PIC2DConst::device_ny * i] = B[buffer + PIC2DConst::device_ny * i];
        }
        for (int j = 0; j < buffer; j++) {
            B[PIC2DConst::device_ny - 1 - j + PIC2DConst::device_ny * i] = B[PIC2DConst::device_ny - 1 - buffer + PIC2DConst::device_ny * i];
        }
    }
}

void BoundaryPIC::freeBoundaryB_y(
    thrust::device_vector<MagneticField>& B
)
{
    MPI_Barrier(MPI_COMM_WORLD); 

    int threadsPerBlock = 256;
    int blocksPerGrid = (mPIInfo.localSizeX + threadsPerBlock - 1) / threadsPerBlock;

    freeBoundaryB_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(B.data()), 
        mPIInfo.localSizeX, 
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

    PIC2DMPI::sendrecv_electricField_x(
        E, 
        sendElectricFieldLeft, sendElectricFieldRight, 
        recvElectricFieldLeft, recvElectricFieldRight, 
        mPIInfo
    ); 
}


void freeBoundaryE_x(
    thrust::device_vector<ElectricField>& E
)
{
    std::cout << "Not writtern yet. Finish your calculation now!" << std::endl;
}


__global__ void freeBoundaryE_y_kernel(
    ElectricField* E, 
    int localSizeX, 
    int buffer, 
    PIC2DMPI::MPIInfo* device_mPIInfo
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    PIC2DMPI::MPIInfo mPIInfo = *device_mPIInfo;

    if (i < localSizeX) {
        for (int j = 0; j < buffer; j++) {
            E[j + PIC2DConst::device_ny * i] = E[buffer + PIC2DConst::device_ny * i];
        }
        for (int j = 0; j < buffer; j++) {
            E[PIC2DConst::device_ny - 1 - j + PIC2DConst::device_ny * i] = E[PIC2DConst::device_ny - 1 - buffer + PIC2DConst::device_ny * i];
        }
    }
}

void BoundaryPIC::freeBoundaryE_y(
    thrust::device_vector<ElectricField>& E
)
{
    MPI_Barrier(MPI_COMM_WORLD); 

    int threadsPerBlock = 256;
    int blocksPerGrid = (mPIInfo.localSizeX + threadsPerBlock - 1) / threadsPerBlock;

    freeBoundaryE_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(E.data()), 
        mPIInfo.localSizeX, 
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

    PIC2DMPI::sendrecv_currentField_x(
        current, 
        sendCurrentFieldLeft, sendCurrentFieldRight, 
        recvCurrentFieldLeft, recvCurrentFieldRight, 
        mPIInfo
    ); 
}


void freeBoundaryCurrent_x(
    thrust::device_vector<CurrentField>& current
)
{
    std::cout << "Not writtern yet. Finish your calculation now!" << std::endl;
}


__global__ void freeBoundaryCurrent_y_kernel(
    CurrentField* current,  
    int localSizeX, 
    int buffer, 
    PIC2DMPI::MPIInfo* device_mPIInfo
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    PIC2DMPI::MPIInfo mPIInfo = *device_mPIInfo;

    if (i < localSizeX) {
        for (int j = 0; j < buffer; j++) {
            current[j + PIC2DConst::device_ny * i] = current[buffer + PIC2DConst::device_ny * i];
        }
        for (int j = 0; j < buffer; j++) {
            current[PIC2DConst::device_ny - 1 - j + PIC2DConst::device_ny * i] = current[PIC2DConst::device_ny - 1 - buffer + PIC2DConst::device_ny * i];
        }
    }
}

void BoundaryPIC::freeBoundaryCurrent_y(
    thrust::device_vector<CurrentField>& current
)
{
    MPI_Barrier(MPI_COMM_WORLD); 
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (mPIInfo.localSizeX + threadsPerBlock - 1) / threadsPerBlock;

    freeBoundaryCurrent_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(current.data()), 
        mPIInfo.localSizeX,  
        mPIInfo.buffer, 
        device_mPIInfo
    );
    cudaDeviceSynchronize();
}



