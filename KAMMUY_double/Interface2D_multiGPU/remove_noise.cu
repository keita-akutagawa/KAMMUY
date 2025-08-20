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
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localSizeXMHD && j < IdealMHD2DConst::device_ny) {
        unsigned long long indexMHD = j + i * IdealMHD2DConst::device_ny;

        double rho, u, v, w, bX, bY, bZ, e, p;

        BasicParameter convolvedBasicParameter; 

        const double kernel[3][3] = {
            {1.0, 2.0, 1.0},
            {2.0, 4.0, 2.0},
            {1.0, 2.0, 1.0}
        };
        double kernelSum = 0.0; 

        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                int localI = i + dx;
                int localJ = j + dy;

                if (0 <= localI && localI < localSizeXMHD &&
                    0 <= localJ && localJ < IdealMHD2DConst::device_ny)
                {
                    unsigned long long localIndex = indexMHD + dy + dx * IdealMHD2DConst::device_ny; 
                    
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

                    double weight = kernel[dx + 1][dy + 1];
                    convolvedBasicParameter += basicParameter * weight;
                    kernelSum += weight; 
                }
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

