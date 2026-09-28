#include "noise_remover.hpp"


NoiseRemover2D::NoiseRemover2D(
    const MHDUnsignedInt level, 
    MHDConstParameter& mHDConstParameter, 
    const MHDGridParameter& mHDGridParameter
)
    : mHDConstParameter(mHDConstParameter),
      mHDGridParameter(mHDGridParameter),

      NX_MHD(mHDGridParameter.NX[level]), 
      NY_MHD(mHDGridParameter.NY[level]), 

      tmpU(NX_MHD * NY_MHD)
{
}


__global__ void convolutionU_kernel(
    const MHDUnsignedInt NX_MHD, const MHDUnsignedInt NY_MHD, 
    const MHDUnsignedInt BUFFER, 
    const MHDValue* tmpU, 
    MHDValue* U
)
{
    MHDUnsignedInt i = blockIdx.x * blockDim.x + threadIdx.x;
    MHDUnsignedInt j = blockIdx.y * blockDim.y + threadIdx.y;

    if (BUFFER <= i && i < NX_MHD - BUFFER && BUFFER <= j && j < NY_MHD - BUFFER) {

        MHDValue averagedMHDValue; 
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                MHDUnsignedInt localI = i + dx;
                MHDUnsignedInt localJ = j + dy;

                MHDUnsignedLongLong localIndexMHD = getIndex<MHDUnsignedLongLong>(localI, localJ, NX_MHD, NY_MHD);

                averagedMHDValue += tmpU[localIndexMHD] / 9.0; 
            }
        }

        MHDUnsignedLongLong indexMHD = getIndex<MHDUnsignedLongLong>(i, j, NX_MHD, NY_MHD);
        U[indexMHD] = averagedMHDValue; 
    }
}


void NoiseRemover2D::convolutionU(
    thrust::device_vector<MHDValue>& U
)
{
    thrust::copy(U.begin(), U.end(), tmpU.begin());

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((NX_MHD + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (NY_MHD + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    convolutionU_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        NX_MHD, NY_MHD, 
        mHDGridParameter.BUFFER, 
        thrust::raw_pointer_cast(tmpU.data()), 
        thrust::raw_pointer_cast(U.data())
    );
    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess) {
        printf("Kernel launch failed at convolutionU_kernel: %s\n", cudaGetErrorString(err1));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err1 = cudaDeviceSynchronize();
    if (err1 != cudaSuccess) {
        printf("Kernel execution failed at convolutionU_kernel: %s\n", cudaGetErrorString(err1));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
}

