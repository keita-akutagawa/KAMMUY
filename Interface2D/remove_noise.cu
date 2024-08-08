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
      tmpFirstMoment(PIC2DConst::nx_PIC * (interfaceLength + windowSize)), 
      tmpU(IdealMHD2DConst::nx_MHD * (interfaceLength + windowSize))
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
        int indexForCopy = j + i * (interfaceLength + windowSize);

        tmpB[indexForCopy] = B[indexPIC];
        tmpE[indexForCopy] = E[indexPIC];
        tmpCurrent[indexForCopy] = current[indexPIC];
    }
}


__global__ void convolveFields_kernel(
    const MagneticField* tmpB, 
    const ElectricField* tmpE, 
    const CurrentField* tmpCurrent, 
    MagneticField* B, 
    ElectricField* E, 
    CurrentField* current, 
    int indexOfInterfaceStartInPIC, 
    int interfaceLength, 
    int windowSize
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (windowSize <= i && i < PIC2DConst::device_nx_PIC - windowSize && windowSize <= j && j < interfaceLength) {
        int ny_PIC = PIC2DConst::device_ny_PIC;
        int indexPIC = indexOfInterfaceStartInPIC + j + i * ny_PIC;
        int ny_Interface = interfaceLength + windowSize;
        int indexForCopy = j + i * ny_Interface;
        MagneticField convolvedB; 
        ElectricField convolvedE;
        CurrentField convolvedCurrent;

        for (int sizeX = -windowSize; sizeX <= windowSize; sizeX++) {
            for (int sizeY = -windowSize; sizeY <= windowSize; sizeY++) {
                convolvedB       = convolvedB       + 1.0 / pow(2.0 * windowSize + 1, 2) * tmpB[indexForCopy + sizeX * ny_Interface + sizeY];
                convolvedE       = convolvedE       + 1.0 / pow(2.0 * windowSize + 1, 2) * tmpE[indexForCopy + sizeX * ny_Interface + sizeY];
                convolvedCurrent = convolvedCurrent + 1.0 / pow(2.0 * windowSize + 1, 2) * tmpCurrent[indexForCopy + sizeX * ny_Interface + sizeY];
            }
        }
        
        B[indexPIC]       = convolvedB;
        E[indexPIC]       = convolvedE; 
        current[indexPIC] = convolvedCurrent;
    }
}

void InterfaceNoiseRemover2D::convolveFields(
    thrust::device_vector<MagneticField>& B, 
    thrust::device_vector<ElectricField>& E, 
    thrust::device_vector<CurrentField>& current
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((PIC2DConst::nx_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
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
        thrust::raw_pointer_cast(tmpB.data()), 
        thrust::raw_pointer_cast(tmpE.data()), 
        thrust::raw_pointer_cast(tmpCurrent.data()), 
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(E.data()), 
        thrust::raw_pointer_cast(current.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize
    );

    cudaDeviceSynchronize();
}



__global__ void copyMoments_kernel(
    const ZerothMoment* zerothMomentSpecies, 
    const FirstMoment* firstMomentSpecies, 
    ZerothMoment* tmpZerothMoment, 
    FirstMoment* tmpFirstMoment, 
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
        int indexForCopy = j + i * (interfaceLength + windowSize);

        tmpZerothMoment[indexForCopy] = zerothMomentSpecies[indexPIC];
        tmpFirstMoment [indexForCopy] = firstMomentSpecies [indexPIC];
    }
}


__global__ void convolveMoments_kernel(
    const ZerothMoment* tmpZerothMoment, 
    const FirstMoment* tmpFirstMoment, 
    ZerothMoment* zerothMomentSpecies, 
    FirstMoment* firstMomentSpecies, 
    int indexOfInterfaceStartInPIC, 
    int interfaceLength, 
    int windowSize
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (windowSize <= i && i < PIC2DConst::device_nx_PIC - windowSize && windowSize <= j && j < interfaceLength) {
        int ny_PIC = PIC2DConst::device_ny_PIC;
        int indexPIC = indexOfInterfaceStartInPIC + j + i * ny_PIC;
        int ny_Interface = interfaceLength + windowSize;
        int indexForCopy = j + i * ny_Interface;
        ZerothMoment convolvedZerothMoment;
        FirstMoment convolvedFirstMoment;

        for (int sizeX = -windowSize; sizeX <= windowSize; sizeX++) {
            for (int sizeY = -windowSize; sizeY <= windowSize; sizeY++) {
                convolvedZerothMoment = convolvedZerothMoment + 1.0 / pow(2.0 * windowSize + 1, 2) * tmpZerothMoment[indexForCopy + sizeX * ny_Interface + sizeY];
                convolvedFirstMoment  = convolvedFirstMoment  + 1.0 / pow(2.0 * windowSize + 1, 2) * tmpFirstMoment[indexForCopy + sizeX * ny_Interface + sizeY];
            }
        }

        zerothMomentSpecies[indexPIC] = convolvedZerothMoment;
        firstMomentSpecies[indexPIC]  = convolvedFirstMoment;
    }
}


void InterfaceNoiseRemover2D::convolveMoments(
    thrust::device_vector<ZerothMoment>& zerothMomentIon, 
    thrust::device_vector<ZerothMoment>& zerothMomentElectron, 
    thrust::device_vector<FirstMoment>& firstMomentIon, 
    thrust::device_vector<FirstMoment>& firstMomentElectron
)
{
    dim3 threadsPerBlockForIon(16, 16);
    dim3 blocksPerGridForIon((PIC2DConst::nx_PIC + threadsPerBlockForIon.x - 1) / threadsPerBlockForIon.x,
                       (interfaceLength + windowSize + threadsPerBlockForIon.y - 1) / threadsPerBlockForIon.y);

    copyMoments_kernel<<<blocksPerGridForIon, threadsPerBlockForIon>>>(
        thrust::raw_pointer_cast(zerothMomentIon.data()), 
        thrust::raw_pointer_cast(firstMomentIon.data()), 
        thrust::raw_pointer_cast(tmpZerothMoment.data()), 
        thrust::raw_pointer_cast(tmpFirstMoment.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize
    );

    cudaDeviceSynchronize();

    convolveMoments_kernel<<<blocksPerGridForIon, threadsPerBlockForIon>>>(
        thrust::raw_pointer_cast(tmpZerothMoment.data()), 
        thrust::raw_pointer_cast(tmpFirstMoment.data()), 
        thrust::raw_pointer_cast(zerothMomentIon.data()), 
        thrust::raw_pointer_cast(firstMomentIon.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize
    );

    cudaDeviceSynchronize();


    dim3 threadsPerBlockForElectron(16, 16);
    dim3 blocksPerGridForElectron((PIC2DConst::nx_PIC + threadsPerBlockForElectron.x - 1) / threadsPerBlockForElectron.x,
                       (interfaceLength + windowSize + threadsPerBlockForElectron.y - 1) / threadsPerBlockForElectron.y);

    copyMoments_kernel<<<blocksPerGridForElectron, threadsPerBlockForElectron>>>(
        thrust::raw_pointer_cast(zerothMomentElectron.data()), 
        thrust::raw_pointer_cast(firstMomentElectron.data()), 
        thrust::raw_pointer_cast(tmpZerothMoment.data()), 
        thrust::raw_pointer_cast(tmpFirstMoment.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize
    );

    cudaDeviceSynchronize();

    convolveMoments_kernel<<<blocksPerGridForElectron, threadsPerBlockForElectron>>>(
        thrust::raw_pointer_cast(tmpZerothMoment.data()), 
        thrust::raw_pointer_cast(tmpFirstMoment.data()), 
        thrust::raw_pointer_cast(zerothMomentElectron.data()), 
        thrust::raw_pointer_cast(firstMomentElectron.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize
    );

    cudaDeviceSynchronize();
}



__global__ void copyU_kernel(
    const ConservationParameter* U, 
    ConservationParameter* tmpU, 
    int indexOfInterfaceStartInMHD, 
    int interfaceLength, 
    int windowSize
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < IdealMHD2DConst::device_nx_MHD && j < interfaceLength + windowSize) {
        int ny_MHD = IdealMHD2DConst::device_ny_MHD;
        int indexMHD = indexOfInterfaceStartInMHD + j - windowSize + i * ny_MHD;
        int indexForCopy = j + i * (interfaceLength + windowSize);

        tmpU[indexForCopy] = U[indexMHD];
    }
}


__global__ void convolveU_kernel(
    const ConservationParameter* tmpU, 
    ConservationParameter* U, 
    int indexOfInterfaceStartInMHD, 
    int interfaceLength, 
    int windowSize
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (windowSize <= i && i < IdealMHD2DConst::device_nx_MHD - windowSize && windowSize <= j && j < interfaceLength) {
        int ny_MHD = IdealMHD2DConst::device_ny_MHD;
        int indexMHD = indexOfInterfaceStartInMHD + j - windowSize + i * ny_MHD;
        int ny_Interface = interfaceLength + windowSize;
        int indexForCopy = j + i * ny_Interface;
        ConservationParameter convolvedU;

        for (int sizeX = -windowSize; sizeX <= windowSize; sizeX++) {
            for (int sizeY = -windowSize; sizeY <= windowSize; sizeY++) {
                convolvedU = convolvedU + 1.0 / pow(2.0 * windowSize + 1, 2) * tmpU[indexForCopy + sizeX * ny_Interface + sizeY];
            }
        }

        U[indexMHD] = convolvedU;

        if (j == windowSize) {
            for (int tmp = 1; tmp <= windowSize; tmp++) {
                U[indexMHD - tmp] = U[indexMHD];
            }
        }
        if (j == interfaceLength - 1) {
            for (int tmp = 1; tmp <= windowSize; tmp++) {
                U[indexMHD + tmp] = U[indexMHD];
            }
        }
        if (i == windowSize) {
            for (int tmp = 1; tmp <= windowSize; tmp++) {
                U[indexMHD - tmp * ny_MHD] = U[indexMHD];
            }
        }
        if (i == IdealMHD2DConst::device_nx_MHD - windowSize - 1) {
            for (int tmp = 1; tmp <= windowSize; tmp++) {
                U[indexMHD + tmp * ny_MHD] = U[indexMHD];
            }
        }
    }
}


void InterfaceNoiseRemover2D::convolveU(
    thrust::device_vector<ConservationParameter>& U 
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((IdealMHD2DConst::nx_MHD + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (interfaceLength + windowSize + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyU_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(tmpU.data()),
        indexOfInterfaceStartInMHD, 
        interfaceLength, 
        windowSize
    );

    cudaDeviceSynchronize();

    convolveU_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpU.data()), 
        thrust::raw_pointer_cast(U.data()), 
        indexOfInterfaceStartInMHD, 
        interfaceLength, 
        windowSize
    );

    cudaDeviceSynchronize();
}
