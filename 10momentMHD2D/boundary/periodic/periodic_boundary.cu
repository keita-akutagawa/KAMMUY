#include "periodic_boundary.hpp"


PeriodicBoundaryXLeft::PeriodicBoundaryXLeft(
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

PeriodicBoundaryXRight::PeriodicBoundaryXRight(
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

PeriodicBoundaryYDown::PeriodicBoundaryYDown(
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

PeriodicBoundaryYUp::PeriodicBoundaryYUp(
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


__global__ static void periodicBoundaryXLeft_kernel(
    const MHDUnsignedInt NX, const MHDUnsignedInt NY,  
    const MHDUnsignedInt BUFFER, 
    MHDValue* U 
)
{
    MHDUnsignedInt j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < NY) {
        for (MHDUnsignedInt buf = 0; buf < BUFFER; buf++) {
            MHDUnsignedLongLong indexForCopy    = getIndex<MHDUnsignedLongLong>(NX - BUFFER * 2 + buf, j, NX, NY);            
            MHDUnsignedLongLong indexForRewrite = getIndex<MHDUnsignedLongLong>(0 + buf,               j, NX, NY);

            U[indexForRewrite] = U[indexForCopy];
        }
    }
}

void PeriodicBoundaryXLeft::apply(
    thrust::device_vector<MHDValue>& U
) 
{
    dim3 threadsPerBlock(1, 256);
    dim3 blocksPerGrid(1, 
                       (NY + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    periodicBoundaryXLeft_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        NX, NY,
        mHDGridParameter.BUFFER, 
        thrust::raw_pointer_cast(U.data())
    ); 
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed at periodicBoundaryXLeft_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed at periodicBoundaryXLeft_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
}


__global__ static void periodicBoundaryXRight_kernel(
    const MHDUnsignedInt NX, const MHDUnsignedInt NY,  
    const MHDUnsignedInt BUFFER, 
    MHDValue* U 
)
{
    MHDUnsignedInt j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < NY) {
        for (MHDUnsignedInt buf = 0; buf < BUFFER; buf++) {
            MHDUnsignedLongLong indexForCopy    = getIndex<MHDUnsignedLongLong>(BUFFER + buf,      j, NX, NY);            
            MHDUnsignedLongLong indexForRewrite = getIndex<MHDUnsignedLongLong>(NX - BUFFER + buf, j, NX, NY);

            U[indexForRewrite] = U[indexForCopy];
        }
    }
}


void PeriodicBoundaryXRight::apply(
    thrust::device_vector<MHDValue>& U
) 
{
    dim3 threadsPerBlock(1, 256);
    dim3 blocksPerGrid(1, 
                       (NY + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    periodicBoundaryXRight_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        NX, NY,
        mHDGridParameter.BUFFER, 
        thrust::raw_pointer_cast(U.data()) 
    ); 
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed at periodicBoundaryXRight_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed at periodicBoundaryXRight_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
}


__global__ static void periodicBoundaryYDown_kernel(
    const MHDUnsignedInt NX, const MHDUnsignedInt NY,  
    const MHDUnsignedInt BUFFER, 
    MHDValue* U 
)
{
    MHDUnsignedInt i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < NX) {
        for (MHDUnsignedInt buf = 0; buf < BUFFER; buf++) {
            MHDUnsignedLongLong indexForCopy    = getIndex<MHDUnsignedLongLong>(i, NY - BUFFER * 2 + buf, NX, NY);            
            MHDUnsignedLongLong indexForRewrite = getIndex<MHDUnsignedLongLong>(i, 0 + buf,               NX, NY);

            U[indexForRewrite] = U[indexForCopy];
        }
    }
}


void PeriodicBoundaryYDown::apply(
    thrust::device_vector<MHDValue>& U
) 
{
    dim3 threadsPerBlock(256, 1);
    dim3 blocksPerGrid((NX + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       1);

    periodicBoundaryYDown_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        NX, NY,
        mHDGridParameter.BUFFER, 
        thrust::raw_pointer_cast(U.data())
    ); 
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed at periodicBoundaryYDown_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed at periodicBoundaryYDown_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
}


__global__ static void periodicBoundaryYUp_kernel(
    const MHDUnsignedInt NX, const MHDUnsignedInt NY,  
    const MHDUnsignedInt BUFFER, 
    MHDValue* U 
)
{
    MHDUnsignedInt i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < NX) {
        for (MHDUnsignedInt buf = 0; buf < BUFFER; buf++) {
            MHDUnsignedLongLong indexForCopy    = getIndex<MHDUnsignedLongLong>(i, BUFFER + buf,      NX, NY);            
            MHDUnsignedLongLong indexForRewrite = getIndex<MHDUnsignedLongLong>(i, NY - BUFFER + buf, NX, NY);

            U[indexForRewrite] = U[indexForCopy];
        }
    }
}


void PeriodicBoundaryYUp::apply(
    thrust::device_vector<MHDValue>& U
) 
{
    dim3 threadsPerBlock(256, 1);
    dim3 blocksPerGrid((NX + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       1);
    
    periodicBoundaryYUp_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        NX, NY,
        mHDGridParameter.BUFFER, 
        thrust::raw_pointer_cast(U.data())
    ); 
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed at periodicBoundaryYUp_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed at periodicBoundaryYUp_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
}
