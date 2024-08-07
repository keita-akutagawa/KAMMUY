#include "remove_noise.hpp"


using namespace IdealMHD2DConst;
using namespace PIC2DConst;
using namespace Interface2DConst;


InterfaceNoiseRemover2D::InterfaceNoiseRemover2D(
    int indexStartMHD, 
    int indexStartPIC, 
    int length, 
    int windowSizeForConvolution
)
    : indexOfInterfaceStartInMHD(indexStartMHD), 
      indexOfInterfaceStartInPIC(indexStartPIC), 
      interfaceLength(length), 
      indexOfInterfaceEndInMHD(indexStartMHD + length), 
      indexOfInterfaceEndInPIC(indexStartPIC + length), 

      windowSize(windowSizeForConvolution), 

      tmpB(PIC2DConst::nx_PIC * (interfaceLength + windowSize)), 
      tmpE(PIC2DConst::nx_PIC * (interfaceLength + windowSize)), 
      tmpCurrent(PIC2DConst::nx_PIC * (interfaceLength + windowSize)), 
      tmpZerothMoment(PIC2DConst::nx_PIC * (interfaceLength + windowSize)), 
      tmpFirstMoment(PIC2DConst::nx_PIC * (interfaceLength + windowSize))
{
}


__global__ void copyFields_kernel(
    const MagneticField* B, 
    const ElectricField* E, 
    const CurrentField* current, 
    MagneticField* tmpB, 
    ElectricField* tmpE, 
    CurrentField* tmpCurrent, 
    int indexOfInterfaceStartInPIC, 
    int interfaceLength, 
    int windowSize
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < PIC2DConst::device_nx_PIC && j < interfaceLength + windowSize) {
        int ny_PIC = PIC2DConst::device_ny_PIC;
        int indexPIC = indexOfInterfaceStartInPIC + j + i * ny_PIC;
        int indexForCopy = 

        tmpB[indexPIC] = B[indexPIC];
        tmpE[indexPIC] = E[indexPIC];
        tmpCurrent[indexPIC] = current[indexPIC];
    }
}


__global__ void convolveFields_kernel(
    MagneticField* B, 
    ElectricField* E, 
    CurrentField* current, 
    int indexOfInterfaceStartInMHD, 
    int indexOfInterfaceStartInPIC, 
    int interfaceLength
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < PIC2DConst::device_nx_PIC - 1 && 0 < j && j < interfaceLength - 1) {
        int ny_PIC = PIC2DConst::device_ny_PIC;
        int indexPIC = indexOfInterfaceStartInPIC + j + i * ny_PIC;
        int 


    }
}

void InterfaceNoiseRemover2D::convolveFields(
    thrust::device_vector<MagneticField>& B, 
    thrust::device_vector<ElectricField>& E, 
    thrust::device_vector<CurrentField>& current
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((IdealMHD2DConst::nx_MHD + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (interfaceLength + windowSize + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyFields_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(E.data()), 
        thrust::raw_pointer_cast(current.data()), 
        thrust::raw_pointer_cast(tmpB.data()), 
        thrust::raw_pointer_cast(tmpE.data()), 
        thrust::raw_pointer_cast(tmpCurrent.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize
    );

    cudaDeviceSynchronize();
    

    convolveFields_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(E.data()), 
        thrust::raw_pointer_cast(current.data()), 
        indexOfInterfaceStartInMHD, 
        indexOfInterfaceStartInPIC, 
        interfaceLength
    );

    cudaDeviceSynchronize();
}

