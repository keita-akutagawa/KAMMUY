#include "calculate_Q.hpp"


CalculateQ::CalculateQ(IdealMHD2DMPI::MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo), 
      muscl(mPIInfo)
{
}

__global__ void getBasicParamter_kernel(
    const ConservationParameter* U, 
    BasicParameter* dQ, 
    int localSizeX, int shiftForNeighbor
)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localSizeX - 1 && j < IdealMHD2DConst::device_ny - 1) {

        double rho, u, v, w, bX, bY, bZ, e, p, psi;
        int index = j + i * IdealMHD2DConst::device_ny;

        rho = U[index].rho;
        u   = U[index].rhoU / rho;
        v   = U[index].rhoV / rho;
        w   = U[index].rhoW / rho;
        bX  = U[index].bX;
        bY  = U[index].bY;
        bZ  = U[index].bZ;
        e   = U[index].e;
        p   = (IdealMHD2DConst::device_gamma - 1.0)
            * (e - 0.5 * (rho * (u * u + v * v + w * w))
            - 0.5 * (bX * bX + bY * bY + bZ * bZ));
        psi = U[index].psi; 
        
        dQ[index].rho = rho;
        dQ[index].u   = u;
        dQ[index].v   = v;
        dQ[index].w   = w;
        dQ[index].bX  = bX; 
        dQ[index].bY  = bY;
        dQ[index].bZ  = bZ;
        dQ[index].p   = p;
        dQ[index].psi = psi; 
    }
}

void CalculateQ::setPhysicalParameterX(
    const thrust::device_vector<ConservationParameter>& U, 
    thrust::device_vector<BasicParameter>& dQCenter
)
{

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (IdealMHD2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    getBasicParamter_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(dQCenter.data()), 
        mPIInfo.localSizeX, IdealMHD2DConst::ny
    );
    cudaDeviceSynchronize();
}

void CalculateQ::setPhysicalParameterY(
    const thrust::device_vector<ConservationParameter>& U, 
    thrust::device_vector<BasicParameter>& dQCenter
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (IdealMHD2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    getBasicParamter_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(dQCenter.data()), 
        mPIInfo.localSizeX, 1
    );
    cudaDeviceSynchronize();
}


void CalculateQ::calculateLeftQX(
    const thrust::device_vector<BasicParameter>& dQCenter, 
    thrust::device_vector<BasicParameter>& dQLeft
)
{ 
    muscl.getLeftQX(dQCenter, dQLeft);
}


void CalculateQ::calculateLeftQY(
    const thrust::device_vector<BasicParameter>& dQCenter, 
    thrust::device_vector<BasicParameter>& dQLeft
)
{ 
    muscl.getLeftQY(dQCenter, dQLeft);
}


void CalculateQ::calculateRightQX(
    const thrust::device_vector<BasicParameter>& dQCenter, 
    thrust::device_vector<BasicParameter>& dQRight
)
{ 
    muscl.getRightQX(dQCenter, dQRight);
}


void CalculateQ::calculateRightQY(
    const thrust::device_vector<BasicParameter>& dQCenter, 
    thrust::device_vector<BasicParameter>& dQRight
)
{ 
    muscl.getRightQY(dQCenter, dQRight);
}


