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
        double rho, u, v, w, bX, bY, bZ, e, p;

        BasicParameter convolvedBasicParameter; 

        const double kernel[3][3] = {
            {1, 1, 1},
            {1, 1, 1},
            {1, 1, 1}
        };
        const double kernelSum = 9.0; 

        for (int windowX = -1; windowX <= 1; windowX++) {
            for (int windowY = -1; windowY <= 1; windowY++) {
                int localIndex = indexMHD + windowY + windowX * IdealMHD2DConst::device_ny; 
                
                BasicParameter basicParameter; 

                rho = tmpU[localIndex].rho; 
                u   = tmpU[localIndex].rhoU / rho;
                v   = tmpU[localIndex].rhoV / rho;
                w   = tmpU[localIndex].rhoW / rho; 
                bX  = tmpU[localIndex].bX;
                bY  = tmpU[localIndex].bY;
                bZ  = tmpU[localIndex].bZ;
                e   = tmpU[localIndex].e;
                p   = (IdealMHD2DConst::device_gamma - 1.0)
                    * (e - 0.5 * rho * (u * u + v * v + w * w)
                    - 0.5 * (bX * bX + bY * bY + bZ * bZ));
                
                basicParameter.rho = rho;
                basicParameter.u   = u;
                basicParameter.v   = v;
                basicParameter.w   = w;
                basicParameter.bX  = bX;
                basicParameter.bY  = bY;
                basicParameter.bZ  = bZ;
                basicParameter.p   = p;

                double weight = kernel[windowX + 1][windowY + 1];
                convolvedBasicParameter += basicParameter * weight;
            }
        }
        convolvedBasicParameter = convolvedBasicParameter / kernelSum; 

        rho = convolvedBasicParameter.rho;
        u   = convolvedBasicParameter.u;
        v   = convolvedBasicParameter.v;   
        w   = convolvedBasicParameter.w;   
        bX  = convolvedBasicParameter.bX;  
        bY  = convolvedBasicParameter.bY;  
        bZ  = convolvedBasicParameter.bZ;  
        p   = convolvedBasicParameter.p;   
        e   = p / (IdealMHD2DConst::device_gamma - 1.0)
            + 0.5 * rho * (u * u + v * v + w * w)
            + 0.5 * (bX * bX + bY * bY + bZ * bZ); 

        U[indexMHD].rho  = rho;
        U[indexMHD].rhoU = rho * u;
        U[indexMHD].rhoV = rho * v;
        U[indexMHD].rhoW = rho * w;
        U[indexMHD].bX   = bX;
        U[indexMHD].bY   = bY;
        U[indexMHD].bZ   = bZ;
        U[indexMHD].e    = e;
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

