#include "remove_noise.hpp"


InterfaceNoiseRemover2D::InterfaceNoiseRemover2D(
    IdealMHD2DMPI::MPIInfo& mPIInfoMHD, 
    PIC2DMPI::MPIInfo& mPIInfoPIC
)
    : mPIInfoMHD(mPIInfoMHD), 
      mPIInfoPIC(mPIInfoPIC), 

      tmpU(mPIInfoMHD.localSizeX * IdealMHD2DConst::ny)
{

    cudaMalloc(&device_mPIInfoMHD, sizeof(IdealMHD2DMPI::MPIInfo));
    cudaMemcpy(device_mPIInfoMHD, &mPIInfoMHD, sizeof(IdealMHD2DMPI::MPIInfo), cudaMemcpyHostToDevice);
    cudaMalloc(&device_mPIInfoPIC, sizeof(PIC2DMPI::MPIInfo));
    cudaMemcpy(device_mPIInfoPIC, &mPIInfoPIC, sizeof(PIC2DMPI::MPIInfo), cudaMemcpyHostToDevice);
    
}


__global__ void convolveU_kernel(
    const ConservationParameter* tmpU, 
    ConservationParameter* U, 
    int localSizeXMHD
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < localSizeXMHD - 1 && 0 < j && j < IdealMHD2DConst::device_ny - 1) {
        int indexMHD = j + i * IdealMHD2DConst::device_ny;

        ConservationParameter convolvedU; 

        for (int windowX = -1; windowX <= 1; windowX++) {
            for (int windowY = -1; windowY <= 1; windowY++) {
                int localIndex = indexMHD + windowY + windowX * IdealMHD2DConst::device_ny; 

                //if (windowX == 0 && windowY == 0) {
                //    convolvedU += 4.0 * tmpU[localIndex];
                //} else if (windowX == 0 || windowY == 0) {
                //    convolvedU += 2.0 * tmpU[localIndex]; 
                //} else {
                //    convolvedU += 1.0 * tmpU[localIndex];
                //}
                convolvedU += tmpU[localIndex];
            }
        }
        //convolvedU = convolvedU / 16.0; 
        convolvedU = convolvedU / 9.0; 

        U[indexMHD] = convolvedU;
    }
}


void InterfaceNoiseRemover2D::convolveU(
    thrust::device_vector<ConservationParameter>& U
)
{
    thrust::copy(U.begin(), U.end(), tmpU.begin());

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfoMHD.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (IdealMHD2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    convolveU_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpU.data()), 
        thrust::raw_pointer_cast(U.data()), 
        mPIInfoMHD.localSizeX
    );
    cudaDeviceSynchronize();
}

