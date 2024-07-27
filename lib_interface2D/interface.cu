#include "interface.hpp"
#include <cmath>


using namespace Interface2DConst;

Interface2D::Interface2D()
    :  interlockingFunctionX(interfaceLength), 
       interlockingFunctionY(interfaceLength), 
       host_interlockingFunctionX(interfaceLength), 
       host_interlockingFunctionY(interfaceLength)
{
    for(int i = 0; interfaceLength; i++) {
        host_interlockingFunctionX[i] = 0.5f * (
            1.0f + cos(Interface2DConst::PI * (i - 0) / (interfaceLength - i))
        );
        host_interlockingFunctionY[i] = 0.5f * (
            1.0f + cos(Interface2DConst::PI  * (i - 0) / (interfaceLength - i))
        );
    }

    interlockingFunctionX = host_interlockingFunctionX;
    interlockingFunctionY = host_interlockingFunctionY;
}


__global__ void sendMHDtoPIC_MagneticField_yDirection_kernel(
    const ConservationParameter* U, 
    MagneticField* B
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < PIC2DConst::device_nx && j < Interface2DConst::interfaceLength) {
        
    }
}

void Interface2D::sendMHDtoPIC_MagneticField_yDirection(
    const thrust::device_vector<ConservationParameter>& U, 
    thrust::device_vector<MagneticField>& B
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((PIC2DConst::nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (Interface2DConst::interfaceLength + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendMHDtoPIC_MagneticField_yDirection_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(B.data())
    );

    cudaDeviceSynchronize();
}



