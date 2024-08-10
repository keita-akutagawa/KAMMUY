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

      tmpB(PIC2DConst::nx_PIC * interfaceLength), 
      tmpE(PIC2DConst::nx_PIC * interfaceLength), 
      tmpCurrent(PIC2DConst::nx_PIC * interfaceLength), 
      tmpZerothMoment(PIC2DConst::nx_PIC * interfaceLength), 
      tmpFirstMoment(PIC2DConst::nx_PIC * interfaceLength), 
      tmpU(IdealMHD2DConst::nx_MHD * interfaceLength)
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

    if (i < PIC2DConst::device_nx_PIC && j < interfaceLength) {
        int ny_PIC = PIC2DConst::device_ny_PIC;
        int indexPIC = indexOfInterfaceStartInPIC + j + i * ny_PIC;
        int ny_Interface = interfaceLength;
        int indexForCopy = j + i * ny_Interface;

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

    if (i < PIC2DConst::device_nx_PIC && j < interfaceLength) {
        int ny_PIC = PIC2DConst::device_ny_PIC;
        int indexPIC = indexOfInterfaceStartInPIC + j + i * ny_PIC;
        int ny_Interface = interfaceLength;
        int indexForCopy = j + i * ny_Interface;
        MagneticField convolvedB; 
        ElectricField convolvedE;
        CurrentField convolvedCurrent;
        int windowSizeX = min(min(i, PIC2DConst::device_nx_PIC - 1 - i), windowSize);
        int windowSizeY = min(min(j, interfaceLength - 1 - j), windowSize);

        for (int sizeX = -windowSizeX; sizeX <= windowSizeX; sizeX++) {
            for (int sizeY = -windowSizeY; sizeY <= windowSizeY; sizeY++) {
                convolvedB       = convolvedB + 1.0 / (2.0 * windowSizeX + 1.0) / (2.0 * windowSizeY + 1.0)
                                 * tmpB[indexForCopy + sizeX * ny_Interface + sizeY];
                convolvedE       = convolvedE + 1.0 / (2.0 * windowSizeX + 1.0) / (2.0 * windowSizeY + 1.0)
                                 * tmpE[indexForCopy + sizeX * ny_Interface + sizeY];
                convolvedCurrent = convolvedCurrent + 1.0 / (2.0 * windowSizeX + 1.0) / (2.0 * windowSizeY + 1.0)
                                 * tmpCurrent[indexForCopy + sizeX * ny_Interface + sizeY];
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

    if (i < PIC2DConst::device_nx_PIC && j < interfaceLength) {
        int ny_PIC = PIC2DConst::device_ny_PIC;
        int indexPIC = indexOfInterfaceStartInPIC + j + i * ny_PIC;
        int ny_Interface = interfaceLength;
        int indexForCopy = j + i * ny_Interface;

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

    if (i < PIC2DConst::device_nx_PIC && j < interfaceLength) {
        int ny_PIC = PIC2DConst::device_ny_PIC;
        int indexPIC = indexOfInterfaceStartInPIC + j + i * ny_PIC;
        int ny_Interface = interfaceLength;
        int indexForCopy = j + i * ny_Interface;
        ZerothMoment convolvedZerothMoment;
        FirstMoment convolvedFirstMoment;
        int windowSizeX = min(min(i, PIC2DConst::device_nx_PIC - 1 - i), windowSize);
        int windowSizeY = min(min(j, interfaceLength - 1 - j), windowSize);

        for (int sizeX = -windowSize; sizeX <= windowSize; sizeX++) {
            for (int sizeY = -windowSize; sizeY <= windowSize; sizeY++) {
                convolvedZerothMoment = convolvedZerothMoment + 1.0 / (2.0 * windowSizeX + 1.0) / (2.0 * windowSizeY + 1.0)
                                      * tmpZerothMoment[indexForCopy + sizeX * ny_Interface + sizeY];
                convolvedFirstMoment  = convolvedFirstMoment + 1.0 / (2.0 * windowSizeX + 1.0) / (2.0 * windowSizeY + 1.0)
                                      * tmpFirstMoment[indexForCopy + sizeX * ny_Interface + sizeY];
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

    if (i < IdealMHD2DConst::device_nx_MHD && j < interfaceLength) {
        int ny_MHD = IdealMHD2DConst::device_ny_MHD;
        int indexMHD = indexOfInterfaceStartInMHD + j + i * ny_MHD;
        int ny_Interface = interfaceLength;
        int indexForCopy = j + i * ny_Interface;

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

    if (i < IdealMHD2DConst::device_nx_MHD && j < interfaceLength) {
        int ny_MHD = IdealMHD2DConst::device_ny_MHD;
        int indexMHD = indexOfInterfaceStartInMHD + j + i * ny_MHD;
        int ny_Interface = interfaceLength;
        int indexForCopy = j + i * ny_Interface;
        ConservationParameter convolvedU;
        int windowSizeX = min(min(i, IdealMHD2DConst::device_nx_MHD - 1 - i), windowSize);
        int windowSizeY = min(min(j, interfaceLength - 1 - j), windowSize);

        for (int sizeX = -windowSizeX; sizeX <= windowSizeX; sizeX++) {
            for (int sizeY = -windowSizeY; sizeY <= windowSizeY; sizeY++) {
                convolvedU = convolvedU + 1.0 / (2.0 * windowSizeX + 1.0) / (2.0 * windowSizeY + 1.0)
                           * tmpU[indexForCopy + sizeX * ny_Interface + sizeY];
            }
        }

        U[indexMHD] = convolvedU;
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
