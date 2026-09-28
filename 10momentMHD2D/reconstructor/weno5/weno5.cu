#include "weno5.hpp"


WENO5::WENO5(
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


void WENO5::calculateReconstructedMHDValue(
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


void WENO5::calculateCenterMHDValue(
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


__device__ MHDFloat calculateLeftMHDValueSingle(
    MHDFloat left2, MHDFloat left1, MHDFloat center, MHDFloat right1, MHDFloat right2
)
{
    MHDFloat U0 = (3.0 * left2 - 10.0 * left1 + 15.0 * center) / 8.0; 
    MHDFloat U1 = (-left1 + 6.0 * center + 3.0 * right1) / 8.0; 
    MHDFloat U2 = (3.0 * center + 6.0 * right1 - right2) / 8.0; 

    MHDFloat IS0 = 13.0 / 12.0 * pow(left2 - 2.0 * left1 + center, 2) 
                 + 1.0 / 4.0 * pow(left2 - 4.0 * left1 + 3.0 * center, 2); 
    MHDFloat IS1 = 13.0 / 12.0 * pow(left1 - 2.0 * center + right1, 2) 
                 + 1.0 / 4.0 * pow(left1 - right1, 2); 
    MHDFloat IS2 = 13.0 / 12.0 * pow(center - 2.0 * right1 + right2, 2) 
                 + 1.0 / 4.0 * pow(3.0 * center - 4.0 * right1 + right2, 2); 
    
    MHDFloat a0 = (1.0 / 16.0) / pow(IS0 + 1e-6, 2); 
    MHDFloat a1 = (5.0 / 8.0)  / pow(IS1 + 1e-6, 2); 
    MHDFloat a2 = (5.0 / 16.0) / pow(IS2 + 1e-6, 2); 

    MHDFloat w0 = a0 / (a0 + a1 + a2);
    MHDFloat w1 = a1 / (a0 + a1 + a2);
    MHDFloat w2 = a2 / (a0 + a1 + a2);

    return w0 * U0 + w1 * U1 + w2 * U2;
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

    if (2 <= i && i < NX - 2 && 2 <= j && j < NY - 2) {
        MHDUnsignedLongLong index = getIndex<MHDUnsignedLongLong>(i, j, NX, NY);

        const MHDValue& left2  = centerMHDValue[index - 2 * shift];
        const MHDValue& left1  = centerMHDValue[index - shift];
        const MHDValue& center = centerMHDValue[index];
        const MHDValue& right1 = centerMHDValue[index + shift];
        const MHDValue& right2 = centerMHDValue[index + 2 * shift];

        leftMHDValue[index].rho = calculateLeftMHDValueSingle(
            left2.rho, left1.rho, center.rho, right1.rho, right2.rho
        );
        leftMHDValue[index].u = calculateLeftMHDValueSingle(
            left2.u, left1.u, center.u, right1.u, right2.u
        );
        leftMHDValue[index].v = calculateLeftMHDValueSingle(
            left2.v, left1.v, center.v, right1.v, right2.v
        );
        leftMHDValue[index].w = calculateLeftMHDValueSingle(
            left2.w, left1.w, center.w, right1.w, right2.w
        );
        leftMHDValue[index].bX = calculateLeftMHDValueSingle(
            left2.bX, left1.bX, center.bX, right1.bX, right2.bX
        );
        leftMHDValue[index].bY = calculateLeftMHDValueSingle(
            left2.bY, left1.bY, center.bY, right1.bY, right2.bY
        );
        leftMHDValue[index].bZ = calculateLeftMHDValueSingle(
            left2.bZ, left1.bZ, center.bZ, right1.bZ, right2.bZ
        );
        leftMHDValue[index].pXX = calculateLeftMHDValueSingle(
            left2.pXX, left1.pXX, center.pXX, right1.pXX, right2.pXX
        );
        leftMHDValue[index].pYY = calculateLeftMHDValueSingle(
            left2.pYY, left1.pYY, center.pYY, right1.pYY, right2.pYY
        );
        leftMHDValue[index].pZZ = calculateLeftMHDValueSingle(
            left2.pZZ, left1.pZZ, center.pZZ, right1.pZZ, right2.pZZ
        );
        leftMHDValue[index].pXY = calculateLeftMHDValueSingle(
            left2.pXY, left1.pXY, center.pXY, right1.pXY, right2.pXY
        );
        leftMHDValue[index].pXZ = calculateLeftMHDValueSingle(
            left2.pXZ, left1.pXZ, center.pXZ, right1.pXZ, right2.pXZ
        );
        leftMHDValue[index].pYZ = calculateLeftMHDValueSingle(
            left2.pYZ, left1.pYZ, center.pYZ, right1.pYZ, right2.pYZ
        );
        leftMHDValue[index].psi = calculateLeftMHDValueSingle(
            left2.psi, left1.psi, center.psi, right1.psi, right2.psi
        );
    }
}


void WENO5::calculateLeftMHDValue(
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


__device__ MHDFloat calculateRightMHDValueSingle(
    MHDFloat left1, MHDFloat center, MHDFloat right1, MHDFloat right2, MHDFloat right3
)
{
    MHDFloat U0 = (3.0 * right3 - 10.0 * right2 + 15.0 * right1) / 8.0; 
    MHDFloat U1 = (-right2 + 6.0 * right1 + 3.0 * center) / 8.0; 
    MHDFloat U2 = (3.0 * right1 + 6.0 * center - left1) / 8.0; 

    MHDFloat IS0 = 13.0 / 12.0 * pow(right3 - 2.0 * right2 + right1, 2) 
                 + 1.0 / 4.0 * pow(right3 - 4.0 * right2 + 3.0 * right1, 2); 
    MHDFloat IS1 = 13.0 / 12.0 * pow(right2 - 2.0 * right1 + center, 2) 
                 + 1.0 / 4.0 * pow(right2 - center, 2); 
    MHDFloat IS2 = 13.0 / 12.0 * pow(right1 - 2.0 * center + left1, 2) 
                 + 1.0 / 4.0 * pow(3.0 * right1 - 4.0 * center + left1, 2); 
    
    MHDFloat a0 = (1.0 / 16.0) / pow(IS0 + 1e-6, 2); 
    MHDFloat a1 = (5.0 / 8.0)  / pow(IS1 + 1e-6, 2); 
    MHDFloat a2 = (5.0 / 16.0) / pow(IS2 + 1e-6, 2); 

    MHDFloat w0 = a0 / (a0 + a1 + a2);
    MHDFloat w1 = a1 / (a0 + a1 + a2);
    MHDFloat w2 = a2 / (a0 + a1 + a2);

    return w0 * U0 + w1 * U1 + w2 * U2;
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

    if (1 <= i && i < NX - 3 && 1 <= j && j < NY - 3) {
        MHDUnsignedLongLong index = getIndex<MHDUnsignedLongLong>(i, j, NX, NY);

        const MHDValue& left1  = centerMHDValue[index - shift];
        const MHDValue& center = centerMHDValue[index];
        const MHDValue& right1 = centerMHDValue[index + shift];
        const MHDValue& right2 = centerMHDValue[index + 2 * shift];
        const MHDValue& right3 = centerMHDValue[index + 3 * shift];

        rightMHDValue[index].rho = calculateRightMHDValueSingle(
            left1.rho, center.rho, right1.rho, right2.rho, right3.rho
        );
        rightMHDValue[index].u = calculateRightMHDValueSingle(
            left1.u, center.u, right1.u, right2.u, right3.u
        );
        rightMHDValue[index].v = calculateRightMHDValueSingle(
            left1.v, center.v, right1.v, right2.v, right3.v
        );
        rightMHDValue[index].w = calculateRightMHDValueSingle(
            left1.w, center.w, right1.w, right2.w, right3.w
        );
        rightMHDValue[index].bX = calculateRightMHDValueSingle(
            left1.bX, center.bX, right1.bX, right2.bX, right3.bX
        );
        rightMHDValue[index].bY = calculateRightMHDValueSingle(
            left1.bY, center.bY, right1.bY, right2.bY, right3.bY
        );
        rightMHDValue[index].bZ = calculateRightMHDValueSingle(
            left1.bZ, center.bZ, right1.bZ, right2.bZ, right3.bZ
        );
        rightMHDValue[index].pXX = calculateRightMHDValueSingle(
            left1.pXX, center.pXX, right1.pXX, right2.pXX, right3.pXX
        );
        rightMHDValue[index].pYY = calculateRightMHDValueSingle(
            left1.pYY, center.pYY, right1.pYY, right2.pYY, right3.pYY
        );
        rightMHDValue[index].pZZ = calculateRightMHDValueSingle(
            left1.pZZ, center.pZZ, right1.pZZ, right2.pZZ, right3.pZZ
        );
        rightMHDValue[index].pXY = calculateRightMHDValueSingle(
            left1.pXY, center.pXY, right1.pXY, right2.pXY, right3.pXY
        );
        rightMHDValue[index].pXZ = calculateRightMHDValueSingle(
            left1.pXZ, center.pXZ, right1.pXZ, right2.pXZ, right3.pXZ
        );
        rightMHDValue[index].pYZ = calculateRightMHDValueSingle(
            left1.pYZ, center.pYZ, right1.pYZ, right2.pYZ, right3.pYZ
        );
        rightMHDValue[index].psi = calculateRightMHDValueSingle(
            left1.psi, center.psi, right1.psi, right2.psi, right3.psi
        );
    }
}


void WENO5::calculateRightMHDValue(
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


const thrust::device_vector<MHDValue>& WENO5::getCenterMHDValueRef() const 
{
    return centerMHDValue; 
} 


const thrust::device_vector<MHDValue>& WENO5::getLeftMHDValueRef() const 
{
    return leftMHDValue; 
} 


const thrust::device_vector<MHDValue>& WENO5::getRightMHDValueRef() const 
{
    return rightMHDValue;
}

