#include "muscl.hpp"


MUSCL::MUSCL(
    const MHDUnsignedInt NX, const MHDUnsignedInt NY,
    MHDConstParameter& mHDConstParameter
)
    : NX(NX), 
      NY(NY), 
      mHDConstParameter(mHDConstParameter), 

      centerMHDValue(NX * NY), 
      leftMHDValue  (NX * NY), 
      rightMHDValue (NX * NY)
{

}


void MUSCL::calculateReconstructedMHDValue(
    const thrust::device_vector<MHDValue>& U,
    const MHDUnsignedInt& shift
)
{
    calculateCenterMHDValue(U);
    calculateLeftMHDValue(shift);
    calculateRightMHDValue(shift);
}


__global__ static void calculateCenterMHDValue_kernel(
    const MHDValue* U, 
    const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
    MHDValue* centerMHDValue 
)
{
    MHDUnsignedInt i = blockIdx.x * blockDim.x + threadIdx.x;
    MHDUnsignedInt j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < NX) && (j < NY)) {
        MHDUnsignedLongLong index = getIndex<MHDUnsignedLongLong>(i, j, NX, NY);

        centerMHDValue[index] = U[index];
    }
}

void MUSCL::calculateCenterMHDValue(
    const thrust::device_vector<MHDValue>& U
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (NY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    calculateCenterMHDValue_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        NX, NY, 
        thrust::raw_pointer_cast(centerMHDValue.data())
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed at calculateCenterMHDValue_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed at calculateCenterMHDValue_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
}


__global__ static void calculateLeftMHDValue_kernel(
    const MHDValue* centerMHDValue, 
    const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
    const MHDUnsignedInt shift, 
    MHDValue* leftMHDValue
)
{
    MHDUnsignedInt i = blockIdx.x * blockDim.x + threadIdx.x;
    MHDUnsignedInt j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < NX - 1) && (0 < j) && (j < NY - 1)) {
        MHDUnsignedLongLong index = getIndex<MHDUnsignedLongLong>(i, j, NX, NY);

        const MHDValue& center = centerMHDValue[index];
        const MHDValue& left   = centerMHDValue[index - shift];
        const MHDValue& right  = centerMHDValue[index + shift];

        leftMHDValue[index] = center + 0.5 * minmod(center - left, right - center);
    }
}

void MUSCL::calculateLeftMHDValue(
    const MHDUnsignedInt& shift
) 
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (NY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    calculateLeftMHDValue_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(centerMHDValue.data()), 
        NX, NY, 
        shift, 
        thrust::raw_pointer_cast(leftMHDValue.data())
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed at calculateLeftMHDValue_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed at calculateLeftMHDValue_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
} 


__global__ static void calculateRightMHDValue_kernel(
    const MHDValue* centerMHDValue, 
    const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
    const MHDUnsignedInt shift, 
    MHDValue* rightMHDValue
)
{
    MHDUnsignedInt i = blockIdx.x * blockDim.x + threadIdx.x;
    MHDUnsignedInt j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < NX - 2) && (j < NY - 2)) {
        MHDUnsignedLongLong index = getIndex<MHDUnsignedLongLong>(i, j, NX, NY);

        const MHDValue& center = centerMHDValue[index];
        const MHDValue& right  = centerMHDValue[index + shift];
        const MHDValue& right2 = centerMHDValue[index + 2 * shift];

        rightMHDValue[index] = right - 0.5 * minmod(right - center, right2 - right);
    }
}

void MUSCL::calculateRightMHDValue(
    const MHDUnsignedInt& shift
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (NY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    calculateRightMHDValue_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(centerMHDValue.data()), 
        NX, NY, 
        shift, 
        thrust::raw_pointer_cast(rightMHDValue.data())
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed at calculateRightMHDValue_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed at calculateRightMHDValue_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
}


const thrust::device_vector<MHDValue>& MUSCL::getCenterMHDValueRef() const 
{
    return centerMHDValue; 
} 


const thrust::device_vector<MHDValue>& MUSCL::getLeftMHDValueRef() const 
{
    return leftMHDValue; 
} 


const thrust::device_vector<MHDValue>& MUSCL::getRightMHDValueRef() const 
{
    return rightMHDValue;
}




