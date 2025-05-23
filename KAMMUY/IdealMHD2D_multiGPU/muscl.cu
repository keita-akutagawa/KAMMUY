#include "muscl.hpp"
#include <thrust/transform.h>
#include <thrust/tuple.h>


MUSCL::MUSCL(IdealMHD2DMPI::MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo)
{
}


__global__ void leftParameter_kernel(
    const BasicParameter* dQ, 
    BasicParameter* dQLeft, 
    int localSizeX, int shiftForNeighbor
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < localSizeX - 1) && (0 < j) &&  (j < IdealMHD2DConst::device_ny - 1)) {
        unsigned long long index = j + i * IdealMHD2DConst::device_ny;

        dQLeft[index].rho = dQ[index].rho + 0.5 * minmod(dQ[index].rho - dQ[index - shiftForNeighbor].rho, dQ[index + shiftForNeighbor].rho - dQ[index].rho);
        dQLeft[index].u   = dQ[index].u   + 0.5 * minmod(dQ[index].u   - dQ[index - shiftForNeighbor].u  , dQ[index + shiftForNeighbor].u   - dQ[index].u  );
        dQLeft[index].v   = dQ[index].v   + 0.5 * minmod(dQ[index].v   - dQ[index - shiftForNeighbor].v  , dQ[index + shiftForNeighbor].v   - dQ[index].v  );
        dQLeft[index].w   = dQ[index].w   + 0.5 * minmod(dQ[index].w   - dQ[index - shiftForNeighbor].w  , dQ[index + shiftForNeighbor].w   - dQ[index].w  );
        dQLeft[index].bX  = dQ[index].bX;
        dQLeft[index].bY  = dQ[index].bY  + 0.5 * minmod(dQ[index].bY  - dQ[index - shiftForNeighbor].bY , dQ[index + shiftForNeighbor].bY  - dQ[index].bY );
        dQLeft[index].bZ  = dQ[index].bZ  + 0.5 * minmod(dQ[index].bZ  - dQ[index - shiftForNeighbor].bZ , dQ[index + shiftForNeighbor].bZ  - dQ[index].bZ );
        dQLeft[index].p   = dQ[index].p   + 0.5 * minmod(dQ[index].p   - dQ[index - shiftForNeighbor].p  , dQ[index + shiftForNeighbor].p   - dQ[index].p  );
    }
}


void MUSCL::getLeftQX(
    const thrust::device_vector<BasicParameter>& dQ, 
    thrust::device_vector<BasicParameter>& dQLeft
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (IdealMHD2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    leftParameter_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(dQ.data()), 
        thrust::raw_pointer_cast(dQLeft.data()), 
        mPIInfo.localSizeX, IdealMHD2DConst::ny
    );
    cudaDeviceSynchronize();
}


void MUSCL::getLeftQY(
    const thrust::device_vector<BasicParameter>& dQ, 
    thrust::device_vector<BasicParameter>& dQLeft
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (IdealMHD2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    leftParameter_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(dQ.data()), 
        thrust::raw_pointer_cast(dQLeft.data()), 
        mPIInfo.localSizeX, 1 
    );
    cudaDeviceSynchronize();
}


__global__ void rightParameter_kernel(
    const BasicParameter* dQ, 
    BasicParameter* dQRight, 
    int localSizeX, int shiftForNeighbor
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localSizeX - 2 && j < IdealMHD2DConst::device_ny - 2) {
        unsigned long long index = j + i * IdealMHD2DConst::device_ny;

        dQRight[index].rho = dQ[index + shiftForNeighbor].rho - 0.5 * minmod(dQ[index + shiftForNeighbor].rho - dQ[index].rho, dQ[index + 2 * shiftForNeighbor].rho - dQ[index + shiftForNeighbor].rho);
        dQRight[index].u   = dQ[index + shiftForNeighbor].u   - 0.5 * minmod(dQ[index + shiftForNeighbor].u   - dQ[index].u  , dQ[index + 2 * shiftForNeighbor].u   - dQ[index + shiftForNeighbor].u  );
        dQRight[index].v   = dQ[index + shiftForNeighbor].v   - 0.5 * minmod(dQ[index + shiftForNeighbor].v   - dQ[index].v  , dQ[index + 2 * shiftForNeighbor].v   - dQ[index + shiftForNeighbor].v  );
        dQRight[index].w   = dQ[index + shiftForNeighbor].w   - 0.5 * minmod(dQ[index + shiftForNeighbor].w   - dQ[index].w  , dQ[index + 2 * shiftForNeighbor].w   - dQ[index + shiftForNeighbor].w  );
        dQRight[index].bX  = dQ[index].bX;
        dQRight[index].bY  = dQ[index + shiftForNeighbor].bY  - 0.5 * minmod(dQ[index + shiftForNeighbor].bY  - dQ[index].bY , dQ[index + 2 * shiftForNeighbor].bY  - dQ[index + shiftForNeighbor].bY );
        dQRight[index].bZ  = dQ[index + shiftForNeighbor].bZ  - 0.5 * minmod(dQ[index + shiftForNeighbor].bZ  - dQ[index].bZ , dQ[index + 2 * shiftForNeighbor].bZ  - dQ[index + shiftForNeighbor].bZ );
        dQRight[index].p   = dQ[index + shiftForNeighbor].p   - 0.5 * minmod(dQ[index + shiftForNeighbor].p   - dQ[index].p  , dQ[index + 2 * shiftForNeighbor].p   - dQ[index + shiftForNeighbor].p  );
    }
}


void MUSCL::getRightQX(
    const thrust::device_vector<BasicParameter>& dQ, 
    thrust::device_vector<BasicParameter>& dQRight
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (IdealMHD2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    rightParameter_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(dQ.data()), 
        thrust::raw_pointer_cast(dQRight.data()), 
        mPIInfo.localSizeX, IdealMHD2DConst::ny
    );
    cudaDeviceSynchronize();
}


void MUSCL::getRightQY(
    const thrust::device_vector<BasicParameter>& dQ, 
    thrust::device_vector<BasicParameter>& dQRight
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (IdealMHD2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    rightParameter_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(dQ.data()), 
        thrust::raw_pointer_cast(dQRight.data()), 
        mPIInfo.localSizeX, 1 
    );
    cudaDeviceSynchronize();
}

