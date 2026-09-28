#include "interpolate_boundary.hpp"


SMRInterpolateBoundaryXLeft::SMRInterpolateBoundaryXLeft(
    const MHDInt level, 
    MHDConstParameter& mHDConstParameter, 
    const MHDGridParameter& mHDGridParameter 
)
  : level(level), 
    mHDConstParameter(mHDConstParameter), 
    mHDGridParameter(mHDGridParameter), 
    NX(mHDGridParameter.NX[level - 1]), 
    NY(mHDGridParameter.NY[level - 1]),
    SMR_NX(mHDGridParameter.NX[level]), 
    SMR_NY(mHDGridParameter.NY[level]),
    START_INDEX_X(mHDGridParameter.START_INDEX_X[level]), 
    START_INDEX_Y(mHDGridParameter.START_INDEX_Y[level])
{
}


SMRInterpolateBoundaryXRight::SMRInterpolateBoundaryXRight(
    const MHDInt level, 
    MHDConstParameter& mHDConstParameter, 
    const MHDGridParameter& mHDGridParameter
)
  : level(level), 
    mHDConstParameter(mHDConstParameter), 
    mHDGridParameter(mHDGridParameter), 
    NX(mHDGridParameter.NX[level - 1]), 
    NY(mHDGridParameter.NY[level - 1]),
    SMR_NX(mHDGridParameter.NX[level]), 
    SMR_NY(mHDGridParameter.NY[level]),
    START_INDEX_X(mHDGridParameter.START_INDEX_X[level]), 
    START_INDEX_Y(mHDGridParameter.START_INDEX_Y[level])
{
}


SMRInterpolateBoundaryYDown::SMRInterpolateBoundaryYDown(
    const MHDInt level, 
    MHDConstParameter& mHDConstParameter, 
    const MHDGridParameter& mHDGridParameter
)
  : level(level), 
    mHDConstParameter(mHDConstParameter), 
    mHDGridParameter(mHDGridParameter), 
    NX(mHDGridParameter.NX[level - 1]), 
    NY(mHDGridParameter.NY[level - 1]),
    SMR_NX(mHDGridParameter.NX[level]), 
    SMR_NY(mHDGridParameter.NY[level]),
    START_INDEX_X(mHDGridParameter.START_INDEX_X[level]), 
    START_INDEX_Y(mHDGridParameter.START_INDEX_Y[level])
{
}


SMRInterpolateBoundaryYUp::SMRInterpolateBoundaryYUp(
    const MHDInt level, 
    MHDConstParameter& mHDConstParameter, 
    const MHDGridParameter& mHDGridParameter
)
  : level(level), 
    mHDConstParameter(mHDConstParameter), 
    mHDGridParameter(mHDGridParameter), 
    NX(mHDGridParameter.NX[level - 1]), 
    NY(mHDGridParameter.NY[level - 1]),
    SMR_NX(mHDGridParameter.NX[level]), 
    SMR_NY(mHDGridParameter.NY[level]),
    START_INDEX_X(mHDGridParameter.START_INDEX_X[level]), 
    START_INDEX_Y(mHDGridParameter.START_INDEX_Y[level])
{
}


__device__ static inline MHDValue getHalfU(
    const MHDValue& coarseUPast, const MHDValue& coarseUNext, 
    const MHDFloat timeRatio
)
{
    return (1.0 - timeRatio) * coarseUPast + timeRatio * coarseUNext; 
}


__device__ static inline MHDValue calculateBoundaryMHDValue(
    const MHDValue* coarseUPast, const MHDValue* coarseUNext,  
    const MHDUnsignedLongLong indexForCoarseU, const MHDFloat timeRatio, 
    const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
    const MHDUnsignedInt i, const MHDUnsignedInt j
)
{
    MHDFloat cx = (i % 2 == 0) ? -0.25 : 0.25;
    MHDFloat cy = (j % 2 == 0) ? -0.25 : 0.25;

    MHDValue interpolatedU = getHalfU(coarseUPast[indexForCoarseU], coarseUNext[indexForCoarseU], timeRatio)
        + minmod(
            (getHalfU(coarseUPast[indexForCoarseU + NY], coarseUNext[indexForCoarseU + NY], timeRatio) - getHalfU(coarseUPast[indexForCoarseU], coarseUNext[indexForCoarseU], timeRatio)), 
            (getHalfU(coarseUPast[indexForCoarseU], coarseUNext[indexForCoarseU], timeRatio) - getHalfU(coarseUPast[indexForCoarseU - NY], coarseUNext[indexForCoarseU - NY], timeRatio))
        ) * cx 
        + minmod(
            (getHalfU(coarseUPast[indexForCoarseU + 1], coarseUNext[indexForCoarseU + 1], timeRatio) - getHalfU(coarseUPast[indexForCoarseU], coarseUNext[indexForCoarseU], timeRatio)), 
            (getHalfU(coarseUPast[indexForCoarseU], coarseUNext[indexForCoarseU], timeRatio) - getHalfU(coarseUPast[indexForCoarseU - 1], coarseUNext[indexForCoarseU - 1], timeRatio))
        ) * cy; 

    return interpolatedU;

    /*
    MHDValue interpolatedU; 

    if (i % 2 == 0 && j % 2 == 0) {
        interpolatedU = 0.75 * 0.75 * getHalfU(coarseUPast[indexForCoarseU], coarseUNext[indexForCoarseU], timeRatio) 
                        + 0.75 * 0.25 * getHalfU(coarseUPast[indexForCoarseU - NY], coarseUNext[indexForCoarseU - NY], timeRatio)
                        + 0.75 * 0.25 * getHalfU(coarseUPast[indexForCoarseU - 1], coarseUNext[indexForCoarseU - 1], timeRatio)
                        + 0.25 * 0.25 * getHalfU(coarseUPast[indexForCoarseU - NY - 1], coarseUNext[indexForCoarseU - NY - 1], timeRatio);
    } 
    if (i % 2 == 0 && j % 2 == 1) {
        interpolatedU = 0.75 * 0.75 * getHalfU(coarseUPast[indexForCoarseU], coarseUNext[indexForCoarseU], timeRatio) 
                      + 0.75 * 0.25 * getHalfU(coarseUPast[indexForCoarseU - NY], coarseUNext[indexForCoarseU - NY], timeRatio)
                      + 0.75 * 0.25 * getHalfU(coarseUPast[indexForCoarseU + 1], coarseUNext[indexForCoarseU + 1], timeRatio)
                      + 0.25 * 0.25 * getHalfU(coarseUPast[indexForCoarseU - NY + 1], coarseUNext[indexForCoarseU - NY + 1], timeRatio);
    }
    if (i % 2 == 1 && j % 2 == 0) {
        interpolatedU = 0.75 * 0.75 * getHalfU(coarseUPast[indexForCoarseU], coarseUNext[indexForCoarseU], timeRatio) 
                      + 0.75 * 0.25 * getHalfU(coarseUPast[indexForCoarseU + NY], coarseUNext[indexForCoarseU + NY], timeRatio)
                      + 0.75 * 0.25 * getHalfU(coarseUPast[indexForCoarseU - 1], coarseUNext[indexForCoarseU - 1], timeRatio)
                      + 0.25 * 0.25 * getHalfU(coarseUPast[indexForCoarseU + NY - 1], coarseUNext[indexForCoarseU + NY - 1], timeRatio);
    }
    if (i % 2 == 1 && j % 2 == 1) {
        interpolatedU = 0.75 * 0.75 * getHalfU(coarseUPast[indexForCoarseU], coarseUNext[indexForCoarseU], timeRatio) 
                      + 0.75 * 0.25 * getHalfU(coarseUPast[indexForCoarseU + NY], coarseUNext[indexForCoarseU + NY], timeRatio)
                      + 0.75 * 0.25 * getHalfU(coarseUPast[indexForCoarseU + 1], coarseUNext[indexForCoarseU + 1], timeRatio)
                      + 0.25 * 0.25 * getHalfU(coarseUPast[indexForCoarseU + NY + 1], coarseUNext[indexForCoarseU + NY + 1], timeRatio);
    }

    return interpolatedU;
    */
}


__global__ static void smrInterpolateBoundaryXLeft_kernel(
    const MHDValue* coarseUPast, const MHDValue* coarseUNext, 
    const MHDUnsignedInt NX, const MHDUnsignedInt NY,  
    const MHDUnsignedInt SMR_NX, const MHDUnsignedInt SMR_NY,  
    const MHDUnsignedInt START_INDEX_X, const MHDUnsignedInt START_INDEX_Y,  
    const MHDUnsignedInt BUFFER, 
    const MHDFloat timeRatio, 
    MHDValue* smrU 
)
{
    MHDUnsignedInt j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < SMR_NY) {
        for (MHDUnsignedInt i = 0; i < BUFFER; i++) {
            MHDUnsignedLongLong indexForSMRU = getIndex<MHDUnsignedLongLong>(i, j, SMR_NX, SMR_NY);

            MHDUnsignedInt indexX = START_INDEX_X + i / 2; 
            MHDUnsignedInt indexY = START_INDEX_Y + j / 2;  
            MHDUnsignedLongLong indexForCoarseU = getIndex<MHDUnsignedLongLong>(indexX, indexY, NX, NY);
            
            smrU[indexForSMRU] = calculateBoundaryMHDValue(
                coarseUPast, coarseUNext, indexForCoarseU, timeRatio, 
                NX, NY, i, j
            ); 
        }
    }
}


void SMRInterpolateBoundaryXLeft::apply(
    const thrust::device_vector<MHDValue>& coarseUPast, 
    const thrust::device_vector<MHDValue>& coarseUNext, 
    const MHDFloat timeRatio, 
    thrust::device_vector<MHDValue>& smrU
) 
{
    dim3 threadsPerBlock(1, 256);
    dim3 blocksPerGrid(1, 
                       (SMR_NY + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    smrInterpolateBoundaryXLeft_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(coarseUPast.data()), 
        thrust::raw_pointer_cast(coarseUNext.data()), 
        NX, NY, 
        SMR_NX, SMR_NY, 
        mHDGridParameter.BUFFER, 
        START_INDEX_X, START_INDEX_Y, 
        timeRatio,    
        thrust::raw_pointer_cast(smrU.data())
    ); 
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed at smrInterpolateBoundaryXLeft_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed at smrInterpolateBoundaryXLeft_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
}


__global__ static void smrInterpolateBoundaryXRight_kernel(
    const MHDValue* coarseUPast, const MHDValue* coarseUNext, 
    const MHDUnsignedInt NX, const MHDUnsignedInt NY,  
    const MHDUnsignedInt SMR_NX, const MHDUnsignedInt SMR_NY,  
    const MHDUnsignedInt BUFFER, 
    const MHDUnsignedInt START_INDEX_X, const MHDUnsignedInt START_INDEX_Y,  
    const MHDFloat timeRatio, 
    MHDValue* smrU 
)
{
    MHDUnsignedInt j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < SMR_NY) {
        for (MHDUnsignedInt i = SMR_NX - BUFFER; i < SMR_NX; i++) {
            MHDUnsignedLongLong indexForSMRU = getIndex<MHDUnsignedLongLong>(i, j, SMR_NX, SMR_NY);

            MHDUnsignedInt indexX = START_INDEX_X + i / 2; 
            MHDUnsignedInt indexY = START_INDEX_Y + j / 2;  
            MHDUnsignedLongLong indexForCoarseU = getIndex<MHDUnsignedLongLong>(indexX, indexY, NX, NY);
            
            smrU[indexForSMRU] = calculateBoundaryMHDValue(
                coarseUPast, coarseUNext, indexForCoarseU, timeRatio, 
                NX, NY, i, j
            ); 
        }
    }
}


void SMRInterpolateBoundaryXRight::apply(
    const thrust::device_vector<MHDValue>& coarseUPast, 
    const thrust::device_vector<MHDValue>& coarseUNext, 
    const MHDFloat timeRatio, 
    thrust::device_vector<MHDValue>& smrU
) 
{
    dim3 threadsPerBlock(1, 256);
    dim3 blocksPerGrid(1, 
                       (SMR_NY + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    smrInterpolateBoundaryXRight_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(coarseUPast.data()), 
        thrust::raw_pointer_cast(coarseUNext.data()), 
        NX, NY, 
        SMR_NX, SMR_NY, 
        mHDGridParameter.BUFFER, 
        START_INDEX_X, START_INDEX_Y, 
        timeRatio,   
        thrust::raw_pointer_cast(smrU.data())
    ); 
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed at smrInterpolateBoundaryXRight_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed at smrInterpolateBoundaryXRight_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
}


__global__ static void smrInterpolateBoundaryYDown_kernel(
    const MHDValue* coarseUPast, const MHDValue* coarseUNext, 
    const MHDUnsignedInt NX, const MHDUnsignedInt NY,  
    const MHDUnsignedInt SMR_NX, const MHDUnsignedInt SMR_NY,  
    const MHDUnsignedInt BUFFER, 
    const MHDUnsignedInt START_INDEX_X, const MHDUnsignedInt START_INDEX_Y,  
    const MHDFloat timeRatio, 
    MHDValue* smrU 
)
{
    MHDUnsignedInt i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < SMR_NX) {
        for (MHDUnsignedInt j = 0; j < BUFFER; j++) {
            MHDUnsignedLongLong indexForSMRU = getIndex<MHDUnsignedLongLong>(i, j, SMR_NX, SMR_NY);

            MHDUnsignedInt indexX = START_INDEX_X + i / 2; 
            MHDUnsignedInt indexY = START_INDEX_Y + j / 2;  
            MHDUnsignedLongLong indexForCoarseU = getIndex<MHDUnsignedLongLong>(indexX, indexY, NX, NY);
            
            smrU[indexForSMRU] = calculateBoundaryMHDValue(
                coarseUPast, coarseUNext, indexForCoarseU, timeRatio, 
                NX, NY, i, j
            ); 
        }
    }
}


void SMRInterpolateBoundaryYDown::apply(
    const thrust::device_vector<MHDValue>& coarseUPast, 
    const thrust::device_vector<MHDValue>& coarseUNext, 
    const MHDFloat timeRatio, 
    thrust::device_vector<MHDValue>& smrU
) 
{
    dim3 threadsPerBlock(256, 1);
    dim3 blocksPerGrid((SMR_NX + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       1);
    
    smrInterpolateBoundaryYDown_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(coarseUPast.data()), 
        thrust::raw_pointer_cast(coarseUNext.data()), 
        NX, NY, 
        SMR_NX, SMR_NY, 
        mHDGridParameter.BUFFER, 
        START_INDEX_X, START_INDEX_Y, 
        timeRatio,   
        thrust::raw_pointer_cast(smrU.data())
    ); 
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed at smrInterpolateBoundaryYDown_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed at smrInterpolateBoundaryYDown_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
}


__global__ static void smrInterpolateBoundaryYUp_kernel(
    const MHDValue* coarseUPast, const MHDValue* coarseUNext, 
    const MHDUnsignedInt NX, const MHDUnsignedInt NY,  
    const MHDUnsignedInt SMR_NX, const MHDUnsignedInt SMR_NY,  
    const MHDUnsignedInt BUFFER, 
    const MHDUnsignedInt START_INDEX_X, const MHDUnsignedInt START_INDEX_Y,  
    const MHDFloat timeRatio, 
    MHDValue* smrU 
)
{
    MHDUnsignedInt i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < SMR_NX) {
        for (MHDUnsignedInt j = SMR_NY - BUFFER; j < SMR_NY; j++) {
            MHDUnsignedLongLong indexForSMRU = getIndex<MHDUnsignedLongLong>(i, j, SMR_NX, SMR_NY);

            MHDUnsignedInt indexX = START_INDEX_X + i / 2; 
            MHDUnsignedInt indexY = START_INDEX_Y + j / 2;  
            MHDUnsignedLongLong indexForCoarseU = getIndex<MHDUnsignedLongLong>(indexX, indexY, NX, NY);
            
            smrU[indexForSMRU] = calculateBoundaryMHDValue(
                coarseUPast, coarseUNext, indexForCoarseU, timeRatio, 
                NX, NY, i, j
            ); 
        }
    }
}


void SMRInterpolateBoundaryYUp::apply(
    const thrust::device_vector<MHDValue>& coarseUPast, 
    const thrust::device_vector<MHDValue>& coarseUNext, 
    const MHDFloat timeRatio, 
    thrust::device_vector<MHDValue>& smrU
) 
{
    dim3 threadsPerBlock(256, 1);
    dim3 blocksPerGrid((SMR_NX + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       1);
    
    smrInterpolateBoundaryYUp_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(coarseUPast.data()), 
        thrust::raw_pointer_cast(coarseUNext.data()), 
        NX, NY, 
        SMR_NX, SMR_NY, 
        mHDGridParameter.BUFFER, 
        START_INDEX_X, START_INDEX_Y, 
        timeRatio,    
        thrust::raw_pointer_cast(smrU.data())
    ); 
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed at smrInterpolateBoundaryYUp_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed at smrInterpolateBoundaryYUp_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
}
