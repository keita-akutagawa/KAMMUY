#include "free_boundary.hpp"


FreeBoundaryXLeft::FreeBoundaryXLeft(
    const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
    MHDConstParameter& mHDConstParameter, 
    const MHDGridParameter& mHDGridParameter 
)
  : NX(NX), 
    NY(NY), 
    mHDConstParameter(mHDConstParameter), 
    mHDGridParameter(mHDGridParameter)
{
}

FreeBoundaryXRight::FreeBoundaryXRight(
    const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
    MHDConstParameter& mHDConstParameter, 
    const MHDGridParameter& mHDGridParameter
)
  : NX(NX), 
    NY(NY), 
    mHDConstParameter(mHDConstParameter), 
    mHDGridParameter(mHDGridParameter)
{
}

FreeBoundaryYDown::FreeBoundaryYDown(
    const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
    MHDConstParameter& mHDConstParameter, 
    const MHDGridParameter& mHDGridParameter
)
  : NX(NX), 
    NY(NY), 
    mHDConstParameter(mHDConstParameter), 
    mHDGridParameter(mHDGridParameter)
{
}

FreeBoundaryYUp::FreeBoundaryYUp(
    const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
    MHDConstParameter& mHDConstParameter, 
    const MHDGridParameter& mHDGridParameter
)
  : NX(NX), 
    NY(NY), 
    mHDConstParameter(mHDConstParameter), 
    mHDGridParameter(mHDGridParameter)
{
}


__global__ static void freeBoundaryXLeft_kernel(
    const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
    const MHDUnsignedInt BUFFER, 
    MHDValue* U 
)
{
    MHDUnsignedInt j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < NY) {
        for (MHDUnsignedInt buf = 0; buf < BUFFER; buf++) {
            MHDUnsignedLongLong indexForCopy    = getIndex<MHDUnsignedLongLong>(BUFFER,  j, NX, NY);            
            MHDUnsignedLongLong indexForRewrite = getIndex<MHDUnsignedLongLong>(0 + buf, j, NX, NY);

            U[indexForRewrite] = U[indexForCopy];
        }
    }
}

void FreeBoundaryXLeft::apply(
    thrust::device_vector<MHDValue>& U
) 
{
    dim3 threadsPerBlock(1, 256);
    dim3 blocksPerGrid(1, 
                       (NY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    freeBoundaryXLeft_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        NX, NY,
        mHDGridParameter.BUFFER, 
        thrust::raw_pointer_cast(U.data())
    ); 
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed at freeBoundaryXLeft_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed at freeBoundaryXLeft_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
}


__global__ static void freeBoundaryXRight_kernel(
    const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
    const MHDUnsignedInt BUFFER, 
    MHDValue* U 
)
{
    MHDUnsignedInt j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < NY) {
        for (MHDUnsignedInt buf = 0; buf < BUFFER; buf++) {
            MHDUnsignedLongLong indexForCopy    = getIndex<MHDUnsignedLongLong>(NX - 1 - BUFFER, j, NX, NY);            
            MHDUnsignedLongLong indexForRewrite = getIndex<MHDUnsignedLongLong>(NX - 1 - buf,    j, NX, NY);

            U[indexForRewrite] = U[indexForCopy];
        }
    }
}


void FreeBoundaryXRight::apply(
    thrust::device_vector<MHDValue>& U
) 
{
    dim3 threadsPerBlock(1, 256);
    dim3 blocksPerGrid(1, 
                       (NY + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    freeBoundaryXRight_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        NX, NY,
        mHDGridParameter.BUFFER, 
        thrust::raw_pointer_cast(U.data()) 
    ); 
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed at freeBoundaryXRight_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed at freeBoundaryXRight_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
}


__global__ static void freeBoundaryYDown_kernel(
    const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
    const MHDUnsignedInt BUFFER, 
    MHDValue* U 
)
{
    MHDUnsignedInt i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < NX) {
        for (MHDUnsignedInt buf = 0; buf < BUFFER; buf++) {
            MHDUnsignedLongLong indexForCopy    = getIndex<MHDUnsignedLongLong>(i, BUFFER,  NX, NY);            
            MHDUnsignedLongLong indexForRewrite = getIndex<MHDUnsignedLongLong>(i, 0 + buf, NX, NY);

            U[indexForRewrite] = U[indexForCopy];
        }
    }
}


void FreeBoundaryYDown::apply(
    thrust::device_vector<MHDValue>& U
) 
{
    dim3 threadsPerBlock(256, 1);
    dim3 blocksPerGrid((NX + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       1);

    freeBoundaryYDown_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        NX, NY,
        mHDGridParameter.BUFFER, 
        thrust::raw_pointer_cast(U.data())
    ); 
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed at freeBoundaryYDown_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed at freeBoundaryYDown_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
}


__global__ static void freeBoundaryYUp_kernel(
    const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
    const MHDUnsignedInt BUFFER, 
    MHDValue* U 
)
{
    MHDUnsignedInt i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < NX) {
        for (MHDUnsignedInt buf = 0; buf < BUFFER; buf++) {
            MHDUnsignedLongLong indexForCopy    = getIndex<MHDUnsignedLongLong>(i, NY - 1 - BUFFER, NX, NY);            
            MHDUnsignedLongLong indexForRewrite = getIndex<MHDUnsignedLongLong>(i, NY - 1 - buf,    NX, NY);

            U[indexForRewrite] = U[indexForCopy];
        }
    }
}


void FreeBoundaryYUp::apply(
    thrust::device_vector<MHDValue>& U
) 
{
    dim3 threadsPerBlock(256, 1);
    dim3 blocksPerGrid((NX + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       1);

    freeBoundaryYUp_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        NX, NY,
        mHDGridParameter.BUFFER, 
        thrust::raw_pointer_cast(U.data())
    ); 
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed at freeBoundaryYUp_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed at freeBoundaryYUp_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
}
