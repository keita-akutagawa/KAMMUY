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

      tmpB(PIC2DConst::nx_PIC * length), 
      tmpE(PIC2DConst::nx_PIC * length), 
      tmpCurrent(PIC2DConst::nx_PIC * length), 
      tmpZerothMoment(PIC2DConst::nx_PIC * length), 
      tmpFirstMoment(PIC2DConst::nx_PIC * length), 
      tmpU(IdealMHD2DConst::nx_MHD * length)
{
}


template <typename FieldType>
__global__ void copyFields_kernel(
    const FieldType* field, 
    FieldType* tmpField, 
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

        tmpField[indexForCopy] = field[indexPIC];
    }
}


template <typename FieldType>
__global__ void convolveFields_kernel(
    const FieldType* tmpField, 
    FieldType* field, 
    int indexOfInterfaceStartInPIC, 
    int interfaceLength, 
    int windowSize
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < PIC2DConst::device_nx_PIC && windowSize <= j && j < interfaceLength - windowSize) {
        int ny_PIC = PIC2DConst::device_ny_PIC;
        int indexPIC = indexOfInterfaceStartInPIC + j + i * ny_PIC;
        int ny_Interface = interfaceLength;
        int indexForCopy = j + i * ny_Interface;
        FieldType convolvedField; 
        int windowSizeX = min(min(i, PIC2DConst::device_nx_PIC - 1 - i), windowSize);
        int windowSizeY = min(min(j, interfaceLength - 1 - j), windowSize);

        for (int sizeX = -windowSizeX; sizeX <= windowSizeX; sizeX++) {
            for (int sizeY = -windowSizeY; sizeY <= windowSizeY; sizeY++) {
                convolvedField = convolvedField + 1.0 / (2.0 * windowSizeX + 1.0) / (2.0 * windowSizeY + 1.0)
                           * tmpField[indexForCopy + sizeX * ny_Interface + sizeY];
            }
        }
        
        field[indexPIC] = convolvedField;

        if (j == windowSize) {
            for (int tmp = 1; tmp <= windowSize; tmp++) {
                field[indexPIC - tmp] = convolvedField;
            }
        }

        if (j == interfaceLength - windowSize - 1) {
            for (int tmp = 1; tmp <= windowSize; tmp++) {
                field[indexPIC + tmp] = convolvedField;
            }
        }
    }
}


void InterfaceNoiseRemover2D::convolve_magneticField(
    thrust::device_vector<MagneticField>& B
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((PIC2DConst::nx_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (interfaceLength + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyFields_kernel<MagneticField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(tmpB.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize
    );

    cudaDeviceSynchronize();
    

    convolveFields_kernel<MagneticField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpB.data()), 
        thrust::raw_pointer_cast(B.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize
    );

    cudaDeviceSynchronize();
}


void InterfaceNoiseRemover2D::convolve_electricField(
    thrust::device_vector<ElectricField>& E
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((PIC2DConst::nx_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (interfaceLength + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyFields_kernel<ElectricField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(E.data()), 
        thrust::raw_pointer_cast(tmpE.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize
    );

    cudaDeviceSynchronize();
    

    convolveFields_kernel<ElectricField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpE.data()), 
        thrust::raw_pointer_cast(E.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize
    );

    cudaDeviceSynchronize();
}


void InterfaceNoiseRemover2D::convolve_currentField(
    thrust::device_vector<CurrentField>& current
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((PIC2DConst::nx_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (interfaceLength + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyFields_kernel<CurrentField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(current.data()), 
        thrust::raw_pointer_cast(tmpCurrent.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize
    );

    cudaDeviceSynchronize();
    

    convolveFields_kernel<CurrentField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpCurrent.data()), 
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

    if (i < PIC2DConst::device_nx_PIC && windowSize <= j && j < interfaceLength - windowSize) {
        int ny_PIC = PIC2DConst::device_ny_PIC;
        int indexPIC = indexOfInterfaceStartInPIC + j + i * ny_PIC;
        int ny_Interface = interfaceLength;
        int indexForCopy = j + i * ny_Interface;
        ZerothMoment convolvedZerothMoment;
        FirstMoment convolvedFirstMoment;
        int windowSizeX = min(min(i, PIC2DConst::device_nx_PIC - 1 - i), windowSize);
        int windowSizeY = min(min(j, interfaceLength - 1 - j), windowSize);

        for (int sizeX = -windowSizeX; sizeX <= windowSizeX; sizeX++) {
            for (int sizeY = -windowSizeY; sizeY <= windowSizeY; sizeY++) {
                convolvedZerothMoment = convolvedZerothMoment + 1.0 / (2.0 * windowSizeX + 1.0) / (2.0 * windowSizeY + 1.0)
                                      * tmpZerothMoment[indexForCopy + sizeX * ny_Interface + sizeY];
                convolvedFirstMoment  = convolvedFirstMoment + 1.0 / (2.0 * windowSizeX + 1.0) / (2.0 * windowSizeY + 1.0)
                                      * tmpFirstMoment[indexForCopy + sizeX * ny_Interface + sizeY];
            }
        }

        zerothMomentSpecies[indexPIC] = convolvedZerothMoment;
        firstMomentSpecies[indexPIC]  = convolvedFirstMoment;

        if (j == windowSize) {
            for (int tmp = 1; tmp <= windowSize; tmp++) {
                zerothMomentSpecies[indexPIC - tmp] = convolvedZerothMoment;
                firstMomentSpecies[indexPIC - tmp]  = convolvedFirstMoment;
            }
        }

        if (j == interfaceLength - windowSize - 1) {
            for (int tmp = 1; tmp <= windowSize; tmp++) {
                zerothMomentSpecies[indexPIC + tmp] = convolvedZerothMoment;
                firstMomentSpecies[indexPIC + tmp]  = convolvedFirstMoment;
            }
        }
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
                       (interfaceLength + threadsPerBlockForIon.y - 1) / threadsPerBlockForIon.y);

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
                       (interfaceLength + threadsPerBlockForElectron.y - 1) / threadsPerBlockForElectron.y);

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

    if (i < IdealMHD2DConst::device_nx_MHD && windowSize <= j && j < interfaceLength - windowSize) {
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

        if (j == windowSize) {
            for (int tmp = 1; tmp <= windowSize; tmp++) {
                U[indexMHD - tmp] = convolvedU;
            }
        }

        if (j == interfaceLength - windowSize - 1) {
            for (int tmp = 1; tmp <= windowSize; tmp++) {
                U[indexMHD + tmp] = convolvedU;
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
                       (interfaceLength + threadsPerBlock.y - 1) / threadsPerBlock.y);

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
