#include "boundary.hpp"


__global__ void freeBoundaryB_x_kernel(
    MagneticField* B
)
{
    unsigned long long j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < PIC2DConst::device_ny) {
        B[j + PIC2DConst::device_ny * 0] = B[j + PIC2DConst::device_ny * 1];
        B[j + PIC2DConst::device_ny * (PIC2DConst::device_nx - 1)] = B[j + PIC2DConst::device_ny * (PIC2DConst::device_nx - 2)];
    }
}

void BoundaryPIC::freeBoundaryB_x(
    thrust::device_vector<MagneticField>& B
)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (PIC2DConst::ny + threadsPerBlock - 1) / threadsPerBlock;

    freeBoundaryB_x_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(B.data())
    );
    cudaDeviceSynchronize();
}


__global__ void freeBoundaryB_y_kernel(
    MagneticField* B
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < PIC2DConst::device_nx) {
        B[0 + PIC2DConst::device_ny * i] = B[1 + PIC2DConst::device_ny * i];
        B[PIC2DConst::device_ny - 1 + PIC2DConst::device_ny * i] = B[PIC2DConst::device_ny - 2 + PIC2DConst::device_ny * i];
    }
}


void BoundaryPIC::freeBoundaryB_y(
    thrust::device_vector<MagneticField>& B
)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (PIC2DConst::nx + threadsPerBlock - 1) / threadsPerBlock;

    freeBoundaryB_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(B.data())
    );
    cudaDeviceSynchronize();
}


//////////////////////////////////////////////////


__global__ void freeBoundaryE_x_kernel(
    ElectricField* E
)
{
    unsigned long long j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < PIC2DConst::device_ny) {
        E[j + PIC2DConst::device_ny * 0] = E[j + PIC2DConst::device_ny * 1];
        E[j + PIC2DConst::device_ny * (PIC2DConst::device_nx - 1)] = E[j + PIC2DConst::device_ny * (PIC2DConst::device_nx - 2)];
    }
}

void BoundaryPIC::freeBoundaryE_x(
    thrust::device_vector<ElectricField>& E
)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (PIC2DConst::ny + threadsPerBlock - 1) / threadsPerBlock;

    freeBoundaryE_x_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(E.data())\
    );
    cudaDeviceSynchronize();
}


__global__ void freeBoundaryE_y_kernel(
    ElectricField* E
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < PIC2DConst::device_nx) {
        E[0 + PIC2DConst::device_ny * i] = E[1 + PIC2DConst::device_ny * i];
        E[PIC2DConst::device_ny - 1 + PIC2DConst::device_ny * i] = E[PIC2DConst::device_ny - 2 + PIC2DConst::device_ny * i];
    }
}

void BoundaryPIC::freeBoundaryE_y(
    thrust::device_vector<ElectricField>& E
)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (PIC2DConst::nx + threadsPerBlock - 1) / threadsPerBlock;

    freeBoundaryE_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(E.data())\
    );
    cudaDeviceSynchronize();
}

//////////////////////////////////////////////////


__global__ void freeBoundaryCurrent_x_kernel(
    CurrentField* current
)
{
    unsigned long long j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < PIC2DConst::device_ny) {
        current[j + PIC2DConst::device_ny * 0] = current[j + PIC2DConst::device_ny * 1];
        current[j + PIC2DConst::device_ny * (PIC2DConst::device_nx - 1)] = current[j + PIC2DConst::device_ny * (PIC2DConst::device_nx - 2)];
    }
}

void BoundaryPIC::freeBoundaryCurrent_x(
    thrust::device_vector<CurrentField>& current
)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (PIC2DConst::ny + threadsPerBlock - 1) / threadsPerBlock;

    freeBoundaryCurrent_x_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(current.data())
    );
    cudaDeviceSynchronize();
}


__global__ void freeBoundaryCurrent_y_kernel(
    CurrentField* current
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < PIC2DConst::device_nx) {
        current[0 + PIC2DConst::device_ny * i] = current[1 + PIC2DConst::device_ny * i];
        current[PIC2DConst::device_ny - 1 + PIC2DConst::device_ny * i] = current[PIC2DConst::device_ny - 2 + PIC2DConst::device_ny * i];
    }
}


void BoundaryPIC::freeBoundaryCurrent_y(
    thrust::device_vector<CurrentField>& current
)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (PIC2DConst::nx + threadsPerBlock - 1) / threadsPerBlock;

    freeBoundaryCurrent_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(current.data())
    );
    cudaDeviceSynchronize();
}



