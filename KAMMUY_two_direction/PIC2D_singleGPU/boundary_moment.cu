#include "boundary.hpp"


__global__ void freeBoundaryZerothMoment_x_kernel(
    ZerothMoment* zerothMoment
)
{
    unsigned long long j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < PIC2DConst::device_ny) {
        zerothMoment[j + PIC2DConst::device_ny * 0] = zerothMoment[j + PIC2DConst::device_ny * 1];
        zerothMoment[j + PIC2DConst::device_ny * (PIC2DConst::device_nx - 1)] = zerothMoment[j + PIC2DConst::device_ny * (PIC2DConst::device_nx - 2)];
    }
}

void BoundaryPIC::freeBoundaryZerothMoment_x(
    thrust::device_vector<ZerothMoment>& zerothMoment
)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (PIC2DConst::ny + threadsPerBlock - 1) / threadsPerBlock;

    freeBoundaryZerothMoment_x_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(zerothMoment.data())
    );
    cudaDeviceSynchronize();
}


__global__ void freeBoundaryZerothMoment_y_kernel(
    ZerothMoment* zerothMoment
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < PIC2DConst::device_nx) {
        zerothMoment[0 + PIC2DConst::device_ny * i] = zerothMoment[1 + PIC2DConst::device_ny * i];
        zerothMoment[PIC2DConst::device_ny - 1 + PIC2DConst::device_ny * i] = zerothMoment[PIC2DConst::device_ny - 2 + PIC2DConst::device_ny * i];
    }
}

void BoundaryPIC::freeBoundaryZerothMoment_y(
    thrust::device_vector<ZerothMoment>& zerothMoment
)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (PIC2DConst::nx + threadsPerBlock - 1) / threadsPerBlock;

    freeBoundaryZerothMoment_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(zerothMoment.data())
    );
    cudaDeviceSynchronize();
}


//////////////////////////////////////////////////


__global__ void freeBoundaryFirstMoment_x_kernel(
    FirstMoment* firstMoment
)
{
    unsigned long long j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < PIC2DConst::device_ny) {
        firstMoment[j + PIC2DConst::device_ny * 0] = firstMoment[j + PIC2DConst::device_ny * 1];
        firstMoment[j + PIC2DConst::device_ny * (PIC2DConst::device_nx - 1)] = firstMoment[j + PIC2DConst::device_ny * (PIC2DConst::device_nx - 2)];
    }
}

void BoundaryPIC::freeBoundaryFirstMoment_x(
    thrust::device_vector<FirstMoment>& firstMoment
)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (PIC2DConst::ny + threadsPerBlock - 1) / threadsPerBlock;

    freeBoundaryFirstMoment_x_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(firstMoment.data())
    );
    cudaDeviceSynchronize();
}


__global__ void freeBoundaryFirstMoment_y_kernel(
    FirstMoment* firstMoment
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < PIC2DConst::device_nx) {
        firstMoment[0 + PIC2DConst::device_ny * i] = firstMoment[1 + PIC2DConst::device_ny * i];
        firstMoment[PIC2DConst::device_ny - 1 + PIC2DConst::device_ny * i] = firstMoment[PIC2DConst::device_ny - 2 + PIC2DConst::device_ny * i];
    }
}

void BoundaryPIC::freeBoundaryFirstMoment_y(
    thrust::device_vector<FirstMoment>& firstMoment
)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (PIC2DConst::nx + threadsPerBlock - 1) / threadsPerBlock;

    freeBoundaryFirstMoment_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(firstMoment.data())
    );
    cudaDeviceSynchronize();
}


//////////////////////////////////////////////////


__global__ void freeBoundarySecondMoment_x_kernel(
    SecondMoment* secondMoment
)
{
    unsigned long long j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < PIC2DConst::device_ny) {
        secondMoment[j + PIC2DConst::device_ny * 0] = secondMoment[j + PIC2DConst::device_ny * 1];
        secondMoment[j + PIC2DConst::device_ny * (PIC2DConst::device_nx - 1)] = secondMoment[j + PIC2DConst::device_ny * (PIC2DConst::device_nx - 2)];
    }
}

void BoundaryPIC::freeBoundarySecondMoment_x(
    thrust::device_vector<SecondMoment>& secondMoment
)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (PIC2DConst::ny + threadsPerBlock - 1) / threadsPerBlock;

    freeBoundarySecondMoment_x_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(secondMoment.data())
    );
    cudaDeviceSynchronize();
}


__global__ void freeBoundarySecondMoment_y_kernel(
    SecondMoment* secondMoment
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < PIC2DConst::device_nx) {
        secondMoment[0 + PIC2DConst::device_ny * i] = secondMoment[1 + PIC2DConst::device_ny * i];
        secondMoment[PIC2DConst::device_ny - 1 + PIC2DConst::device_ny * i] = secondMoment[PIC2DConst::device_ny - 2 + PIC2DConst::device_ny * i];
    }
}

void BoundaryPIC::freeBoundarySecondMoment_y(
    thrust::device_vector<SecondMoment>& secondMoment
)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (PIC2DConst::nx + threadsPerBlock - 1) / threadsPerBlock;

    freeBoundarySecondMoment_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(secondMoment.data())
    );
    cudaDeviceSynchronize();
}

