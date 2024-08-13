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



//////////////////////////////////////////////////
// Lower side
//////////////////////////////////////////////////

template <typename FieldType>
__global__ void copyFields_lower_kernel(
    const FieldType* field, 
    FieldType* tmpField, 
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
        int ny_Interface = interfaceLength + windowSize;
        int indexForCopy = j + i * ny_Interface;

        tmpField[indexForCopy] = field[indexPIC];
    }
}


template <typename FieldType>
__global__ void convolveFields_lower_kernel(
    const FieldType* tmpField, 
    FieldType* field, 
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
        int ny_Interface = interfaceLength + windowSize;
        int indexForCopy = j + i * ny_Interface;
        FieldType convolvedField; 
        int windowSizeX = min(min(i, PIC2DConst::device_nx_PIC - 1 - i), windowSize);
        int windowSizeY = min(min(j, interfaceLength + windowSize - 1 - j), windowSize);

        for (int sizeX = -windowSizeX; sizeX <= windowSizeX; sizeX++) {
            for (int sizeY = -windowSizeY; sizeY <= windowSizeY; sizeY++) {
                convolvedField = convolvedField + 1.0 / (2.0 * windowSizeX + 1.0) / (2.0 * windowSizeY + 1.0)
                           * tmpField[indexForCopy + sizeX * ny_Interface + sizeY];
            }
        }
        
        field[indexPIC] = convolvedField;
    }
}


__global__ void copyU_lower_kernel(
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
        int ny_Interface = interfaceLength + windowSize;
        int indexForCopy = j + i * ny_Interface;

        tmpU[indexForCopy] = U[indexMHD];
    }
}


__global__ void convolveU_lower_kernel(
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
        int ny_Interface = interfaceLength + windowSize;
        int indexForCopy = j + i * ny_Interface;
        ConservationParameter convolvedU;
        int windowSizeX = min(min(i, IdealMHD2DConst::device_nx_MHD - 1 - i), windowSize);
        int windowSizeY = min(min(j + windowSize, interfaceLength - 1 - j), windowSize);

        for (int sizeX = -windowSizeX; sizeX <= windowSizeX; sizeX++) {
            for (int sizeY = -windowSizeY; sizeY <= windowSizeY; sizeY++) {
                convolvedU = convolvedU + 1.0 / (2.0 * windowSizeX + 1.0) / (2.0 * windowSizeY + 1.0)
                           * tmpU[indexForCopy + sizeX * ny_Interface + sizeY];
            }
        }

        U[indexMHD] = convolvedU;
    }
}


void InterfaceNoiseRemover2D::convolve_lower_magneticField(
    thrust::device_vector<MagneticField>& B
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((PIC2DConst::nx_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (interfaceLength + windowSize + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyFields_lower_kernel<MagneticField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(tmpB.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize
    );

    cudaDeviceSynchronize();
    

    convolveFields_lower_kernel<MagneticField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpB.data()), 
        thrust::raw_pointer_cast(B.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize
    );

    cudaDeviceSynchronize();
}


void InterfaceNoiseRemover2D::convolve_lower_electricField(
    thrust::device_vector<ElectricField>& E
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((PIC2DConst::nx_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (interfaceLength + windowSize + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyFields_lower_kernel<ElectricField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(E.data()), 
        thrust::raw_pointer_cast(tmpE.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize
    );

    cudaDeviceSynchronize();
    

    convolveFields_lower_kernel<ElectricField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpE.data()), 
        thrust::raw_pointer_cast(E.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize
    );

    cudaDeviceSynchronize();
}


void InterfaceNoiseRemover2D::convolve_lower_currentField(
    thrust::device_vector<CurrentField>& current
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((PIC2DConst::nx_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (interfaceLength + windowSize + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyFields_lower_kernel<CurrentField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(current.data()), 
        thrust::raw_pointer_cast(tmpCurrent.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize
    );

    cudaDeviceSynchronize();
    

    convolveFields_lower_kernel<CurrentField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpCurrent.data()), 
        thrust::raw_pointer_cast(current.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize
    );

    cudaDeviceSynchronize();
}


void InterfaceNoiseRemover2D::convolveMoments_lower(
    thrust::device_vector<ZerothMoment>& zerothMomentIon, 
    thrust::device_vector<ZerothMoment>& zerothMomentElectron, 
    thrust::device_vector<FirstMoment>& firstMomentIon, 
    thrust::device_vector<FirstMoment>& firstMomentElectron
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((PIC2DConst::nx_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (interfaceLength + windowSize + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyFields_lower_kernel<ZerothMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(zerothMomentIon.data()), 
        thrust::raw_pointer_cast(tmpZerothMoment.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize
    );

    cudaDeviceSynchronize();

    convolveFields_lower_kernel<ZerothMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpZerothMoment.data()), 
        thrust::raw_pointer_cast(zerothMomentIon.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize
    );

    cudaDeviceSynchronize();


    copyFields_lower_kernel<ZerothMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(zerothMomentElectron.data()), 
        thrust::raw_pointer_cast(tmpZerothMoment.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize
    );

    cudaDeviceSynchronize();

    convolveFields_lower_kernel<ZerothMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpZerothMoment.data()), 
        thrust::raw_pointer_cast(zerothMomentElectron.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize
    );
    
    cudaDeviceSynchronize();

    //////////

    copyFields_lower_kernel<FirstMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(firstMomentIon.data()), 
        thrust::raw_pointer_cast(tmpFirstMoment.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize
    );

    cudaDeviceSynchronize();

    convolveFields_lower_kernel<FirstMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpFirstMoment.data()), 
        thrust::raw_pointer_cast(firstMomentIon.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize
    );
    
    cudaDeviceSynchronize();


    copyFields_lower_kernel<FirstMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(firstMomentElectron.data()), 
        thrust::raw_pointer_cast(tmpFirstMoment.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize
    );

    cudaDeviceSynchronize();

    convolveFields_lower_kernel<FirstMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpFirstMoment.data()), 
        thrust::raw_pointer_cast(firstMomentElectron.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize
    );
    
    cudaDeviceSynchronize();
}


void InterfaceNoiseRemover2D::convolveU_lower(
    thrust::device_vector<ConservationParameter>& U 
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((IdealMHD2DConst::nx_MHD + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (interfaceLength + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyU_lower_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(tmpU.data()),
        indexOfInterfaceStartInMHD, 
        interfaceLength, 
        windowSize
    );

    cudaDeviceSynchronize();

    convolveU_lower_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpU.data()), 
        thrust::raw_pointer_cast(U.data()), 
        indexOfInterfaceStartInMHD, 
        interfaceLength, 
        windowSize
    );

    cudaDeviceSynchronize();
}



//////////////////////////////////////////////////
// Upper side
//////////////////////////////////////////////////

template <typename FieldType>
__global__ void copyFields_upper_kernel(
    const FieldType* field, 
    FieldType* tmpField, 
    int indexOfInterfaceStartInPIC, 
    int interfaceLength, 
    int windowSize
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < PIC2DConst::device_nx_PIC && j < interfaceLength + windowSize) {
        int ny_PIC = PIC2DConst::device_ny_PIC;
        int indexPIC = indexOfInterfaceStartInPIC + j - windowSize + i * ny_PIC;
        int ny_Interface = interfaceLength + windowSize;
        int indexForCopy = j + i * ny_Interface;

        tmpField[indexForCopy] = field[indexPIC];
    }
}


template <typename FieldType>
__global__ void convolveFields_upper_kernel(
    const FieldType* tmpField, 
    FieldType* field, 
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
        int ny_Interface = interfaceLength + windowSize;
        int indexForCopy = j + i * ny_Interface;
        FieldType convolvedField; 
        int windowSizeX = min(min(i, PIC2DConst::device_nx_PIC - 1 - i), windowSize);
        int windowSizeY = min(min(j + windowSize, interfaceLength - 1 - j), windowSize);

        for (int sizeX = -windowSizeX; sizeX <= windowSizeX; sizeX++) {
            for (int sizeY = -windowSizeY; sizeY <= windowSizeY; sizeY++) {
                convolvedField = convolvedField + 1.0 / (2.0 * windowSizeX + 1.0) / (2.0 * windowSizeY + 1.0)
                           * tmpField[indexForCopy + sizeX * ny_Interface + sizeY];
            }
        }
        
        field[indexPIC] = convolvedField;
    }
}


__global__ void copyU_upper_kernel(
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
        int indexMHD = indexOfInterfaceStartInMHD + j + i * ny_MHD;
        int ny_Interface = interfaceLength + windowSize;
        int indexForCopy = j + i * ny_Interface;

        tmpU[indexForCopy] = U[indexMHD];
    }
}


__global__ void convolveU_upper_kernel(
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
        int ny_Interface = interfaceLength + windowSize;
        int indexForCopy = j + i * ny_Interface;
        ConservationParameter convolvedU;
        int windowSizeX = min(min(i, IdealMHD2DConst::device_nx_MHD - 1 - i), windowSize);
        int windowSizeY = min(min(j, interfaceLength + windowSize - 1 - j), windowSize);

        for (int sizeX = -windowSizeX; sizeX <= windowSizeX; sizeX++) {
            for (int sizeY = -windowSizeY; sizeY <= windowSizeY; sizeY++) {
                convolvedU = convolvedU + 1.0 / (2.0 * windowSizeX + 1.0) / (2.0 * windowSizeY + 1.0)
                           * tmpU[indexForCopy + sizeX * ny_Interface + sizeY];
            }
        }

        U[indexMHD] = convolvedU;
    }
}


void InterfaceNoiseRemover2D::convolve_upper_magneticField(
    thrust::device_vector<MagneticField>& B
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((PIC2DConst::nx_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (interfaceLength + windowSize + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyFields_upper_kernel<MagneticField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(tmpB.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize
    );

    cudaDeviceSynchronize();
    

    convolveFields_upper_kernel<MagneticField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpB.data()), 
        thrust::raw_pointer_cast(B.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize
    );

    cudaDeviceSynchronize();
}


void InterfaceNoiseRemover2D::convolve_upper_electricField(
    thrust::device_vector<ElectricField>& E
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((PIC2DConst::nx_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (interfaceLength + windowSize + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyFields_upper_kernel<ElectricField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(E.data()), 
        thrust::raw_pointer_cast(tmpE.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize
    );

    cudaDeviceSynchronize();
    

    convolveFields_upper_kernel<ElectricField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpE.data()), 
        thrust::raw_pointer_cast(E.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize
    );

    cudaDeviceSynchronize();
}


void InterfaceNoiseRemover2D::convolve_upper_currentField(
    thrust::device_vector<CurrentField>& current
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((PIC2DConst::nx_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (interfaceLength + windowSize + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyFields_upper_kernel<CurrentField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(current.data()), 
        thrust::raw_pointer_cast(tmpCurrent.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize
    );

    cudaDeviceSynchronize();
    

    convolveFields_upper_kernel<CurrentField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpCurrent.data()), 
        thrust::raw_pointer_cast(current.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize
    );

    cudaDeviceSynchronize();
}


void InterfaceNoiseRemover2D::convolveMoments_upper(
    thrust::device_vector<ZerothMoment>& zerothMomentIon, 
    thrust::device_vector<ZerothMoment>& zerothMomentElectron, 
    thrust::device_vector<FirstMoment>& firstMomentIon, 
    thrust::device_vector<FirstMoment>& firstMomentElectron
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((PIC2DConst::nx_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (interfaceLength + windowSize + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyFields_upper_kernel<ZerothMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(zerothMomentIon.data()), 
        thrust::raw_pointer_cast(tmpZerothMoment.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize
    );

    cudaDeviceSynchronize();

    convolveFields_upper_kernel<ZerothMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpZerothMoment.data()), 
        thrust::raw_pointer_cast(zerothMomentIon.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize
    );

    cudaDeviceSynchronize();


    copyFields_upper_kernel<ZerothMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(zerothMomentElectron.data()), 
        thrust::raw_pointer_cast(tmpZerothMoment.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize
    );

    cudaDeviceSynchronize();

    convolveFields_upper_kernel<ZerothMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpZerothMoment.data()), 
        thrust::raw_pointer_cast(zerothMomentElectron.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize
    );
    
    cudaDeviceSynchronize();

    //////////

    copyFields_upper_kernel<FirstMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(firstMomentIon.data()), 
        thrust::raw_pointer_cast(tmpFirstMoment.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize
    );

    cudaDeviceSynchronize();

    convolveFields_upper_kernel<FirstMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpFirstMoment.data()), 
        thrust::raw_pointer_cast(firstMomentIon.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize
    );
    
    cudaDeviceSynchronize();


    copyFields_upper_kernel<FirstMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(firstMomentElectron.data()), 
        thrust::raw_pointer_cast(tmpFirstMoment.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize
    );

    cudaDeviceSynchronize();

    convolveFields_upper_kernel<FirstMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpFirstMoment.data()), 
        thrust::raw_pointer_cast(firstMomentElectron.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize
    );
    
    cudaDeviceSynchronize();
}


void InterfaceNoiseRemover2D::convolveU_upper(
    thrust::device_vector<ConservationParameter>& U 
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((IdealMHD2DConst::nx_MHD + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (interfaceLength + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyU_upper_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(tmpU.data()),
        indexOfInterfaceStartInMHD, 
        interfaceLength, 
        windowSize
    );

    cudaDeviceSynchronize();

    convolveU_upper_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpU.data()), 
        thrust::raw_pointer_cast(U.data()), 
        indexOfInterfaceStartInMHD, 
        interfaceLength, 
        windowSize
    );

    cudaDeviceSynchronize();
}

