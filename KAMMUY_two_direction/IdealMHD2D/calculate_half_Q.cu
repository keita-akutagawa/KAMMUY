#include "calculate_half_Q.hpp"


CalculateHalfQ::CalculateHalfQ()
{
}


__global__ void getBasicParamter_kernel(
    const ConservationParameter* U, 
    BasicParameter* dQ, 
    int shiftForNeighbor
)
{

    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < IdealMHD2DConst::device_nx - 1 && j < IdealMHD2DConst::device_ny - 1) {

        double rho, u, v, w, bX, bY, bZ, e, p; 
        unsigned long long index = j + i * IdealMHD2DConst::device_ny;

        rho = U[index].rho;
        u   = U[index].rhoU / rho;
        v   = U[index].rhoV / rho;
        w   = U[index].rhoW / rho;
        bX  = 0.5 * (U[index].bX + U[index + shiftForNeighbor].bX); // flux計算でx, y方向使いまわすため
        bY  = U[index].bY;
        bZ  = U[index].bZ;
        e   = U[index].e;
        p   = (IdealMHD2DConst::device_gamma - 1.0)
            * (e - 0.5 * (rho * (u * u + v * v + w * w))
            - 0.5 * (bX * bX + bY * bY + bZ * bZ));
        
        dQ[index].rho = rho;
        dQ[index].u   = u;
        dQ[index].v   = v;
        dQ[index].w   = w;
        dQ[index].bX  = bX; //HLLDではBxは中心のものを使うため
        dQ[index].bY  = bY;
        dQ[index].bZ  = bZ;
        dQ[index].p   = p;
    }
}


void CalculateHalfQ::setPhysicalParameterX(
    const thrust::device_vector<ConservationParameter>& U, 
    thrust::device_vector<BasicParameter>& dQCenter
)
{

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((IdealMHD2DConst::nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (IdealMHD2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    getBasicParamter_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(dQCenter.data()), 
        IdealMHD2DConst::ny
    );
    cudaDeviceSynchronize();
}

void CalculateHalfQ::setPhysicalParameterY(
    const thrust::device_vector<ConservationParameter>& U, 
    thrust::device_vector<BasicParameter>& dQCenter
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((IdealMHD2DConst::nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (IdealMHD2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    getBasicParamter_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(dQCenter.data()), 
        1
    );
    cudaDeviceSynchronize();
}


void CalculateHalfQ::calculateLeftQX(
    const thrust::device_vector<BasicParameter>& dQCenter, 
    thrust::device_vector<BasicParameter>& dQLeft
)
{ 
    muscl.getLeftQX(dQCenter, dQLeft);
}


void CalculateHalfQ::calculateLeftQY(
    const thrust::device_vector<BasicParameter>& dQCenter, 
    thrust::device_vector<BasicParameter>& dQLeft
)
{ 
    muscl.getLeftQY(dQCenter, dQLeft);
}


void CalculateHalfQ::calculateRightQX(
    const thrust::device_vector<BasicParameter>& dQCenter, 
    thrust::device_vector<BasicParameter>& dQRight
)
{ 
    muscl.getRightQX(dQCenter, dQRight);
}


void CalculateHalfQ::calculateRightQY(
    const thrust::device_vector<BasicParameter>& dQCenter, 
    thrust::device_vector<BasicParameter>& dQRight
)
{ 
    muscl.getRightQY(dQCenter, dQRight);
}

