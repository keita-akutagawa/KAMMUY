#include "smr.hpp"


__global__ static void synchronize_kernel(
    const MHDValue* smrU, 
    const MHDUnsignedInt COARSE_NX, const MHDUnsignedInt COARSE_NY, 
    const MHDUnsignedInt START_INDEX_X, const MHDUnsignedInt START_INDEX_Y, 
    const MHDUnsignedInt END_INDEX_X, const MHDUnsignedInt END_INDEX_Y, 
    const MHDUnsignedInt SMR_NX, const MHDUnsignedInt SMR_NY, 
    const MHDUnsignedInt BUFFER, 
    MHDValue* coarseU 
)
{
    MHDUnsignedInt i = blockIdx.x * blockDim.x + threadIdx.x;
    MHDUnsignedInt j = blockIdx.y * blockDim.y + threadIdx.y;

    if (BUFFER / 2 <= i && i < END_INDEX_X - START_INDEX_X - BUFFER / 2 && 
        BUFFER / 2 <= j && j < END_INDEX_Y - START_INDEX_Y - BUFFER / 2) 
    {
        MHDUnsignedInt indexX = START_INDEX_X + i; 
        MHDUnsignedInt indexY = START_INDEX_Y + j; 
        MHDUnsignedLongLong indexForU = getIndex<MHDUnsignedLongLong>(indexX, indexY, COARSE_NX, COARSE_NY);

        MHDValue averagedU;
        for (int ii = 0; ii < 2; ii++) {
            for (int jj = 0; jj < 2; jj++) {
                MHDUnsignedLongLong indexForSMRU = getIndex<MHDUnsignedLongLong>(
                    i * 2 + ii, j * 2 + jj,
                    SMR_NX, SMR_NY
                );
                averagedU += smrU[indexForSMRU] / 4.0;
            }
        }
        
        coarseU[indexForU] = averagedU;
    }
}


void SMR::synchronizeOneLayer(
    const thrust::device_vector<MHDValue>& U, 
    thrust::device_vector<MHDValue>& coarseU
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((END_INDEX_X - START_INDEX_X + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (END_INDEX_Y - START_INDEX_Y + threadsPerBlock.y - 1) / threadsPerBlock.y);

    synchronize_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        COARSE_NX, COARSE_NY, 
        START_INDEX_X, START_INDEX_Y, 
        END_INDEX_X, END_INDEX_Y, 
        SMR_NX, SMR_NY,
        mHDGridParameter.BUFFER, 
        thrust::raw_pointer_cast(coarseU.data())
    ); 
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed at synchronize_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed at synchronize_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
}

