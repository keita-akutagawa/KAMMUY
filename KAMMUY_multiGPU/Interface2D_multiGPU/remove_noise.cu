#include "remove_noise.hpp"


using namespace IdealMHD2DConst;
using namespace PIC2DConst;
using namespace Interface2DConst;


InterfaceNoiseRemover2D::InterfaceNoiseRemover2D(
    int indexOfInterfaceStartInMHD, 
    int indexOfInterfaceStartInPIC, 
    int interfaceLength, 
    int windowSizeForConvolution, 
    int nx_Interface, int ny_Interface
)
    : indexOfInterfaceStartInMHD(indexOfInterfaceStartInMHD), 
      indexOfInterfaceStartInPIC(indexOfInterfaceStartInPIC), 
      interfaceLength(interfaceLength), 
      windowSize(windowSizeForConvolution), 
      nx_Interface(nx_Interface), 
      ny_Interface(ny_Interface), 

      tmpB(nx_Interface * ny_Interface), 
      tmpE(nx_Interface * ny_Interface), 
      tmpCurrent(nx_Interface * ny_Interface), 
      tmpZerothMoment(nx_Interface * ny_Interface), 
      tmpFirstMoment(nx_Interface * ny_Interface), 
      tmpU(nx_Interface * ny_Interface)
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
    int windowSize, 
    int nx_Interface, int ny_Interface
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nx_Interface && j < ny_Interface) {
        int ny_PIC = PIC2DConst::device_ny_PIC;
        int indexPIC = indexOfInterfaceStartInPIC + j + i * ny_PIC;
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
    int windowSize, 
    int nx_Interface, int ny_Interface
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nx_Interface && windowSize <= j && j < ny_Interface - windowSize) {
        int ny_PIC = PIC2DConst::device_ny_PIC;
        int indexPIC = indexOfInterfaceStartInPIC + j + i * ny_PIC;
        int indexForCopy = j + i * ny_Interface;
        FieldType convolvedField; 
        int windowSizeX = min(min(i, nx_Interface - 1 - i), windowSize);
        int windowSizeY = min(min(j, ny_Interface - 1 - j), windowSize);

        for (int sizeX = -windowSizeX; sizeX <= windowSizeX; sizeX++) {
            for (int sizeY = -windowSizeY; sizeY <= windowSizeY; sizeY++) {
                convolvedField = convolvedField + (1.0 / (2.0 * windowSizeX + 1.0) / (2.0 * windowSizeY + 1.0))
                           * tmpField[indexForCopy + sizeX * ny_Interface + sizeY];
            }
        }
        
        field[indexPIC] = convolvedField;

        if (j == windowSize) {
            for (int tmp = 1; tmp <= windowSize; tmp++) {
                field[indexPIC - tmp] = convolvedField;
            }
        }
    }
}


__global__ void copyU_lower_kernel(
    const ConservationParameter* U, 
    ConservationParameter* tmpU, 
    int indexOfInterfaceStartInMHD, 
    int interfaceLength, 
    int windowSize, 
    int nx_Interface, int ny_Interface
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < IdealMHD2DConst::device_nx_MHD && j < ny_Interface) {
        int ny_MHD = IdealMHD2DConst::device_ny_MHD;
        int indexMHD = indexOfInterfaceStartInMHD
                     + j - (ny_Interface - interfaceLength) + i * ny_MHD;
        int indexForCopy = j + i * ny_Interface;

        tmpU[indexForCopy] = U[indexMHD];
    }
}


__global__ void convolveU_lower_kernel(
    const ConservationParameter* tmpU, 
    ConservationParameter* U, 
    int indexOfInterfaceStartInMHD, 
    int interfaceLength, 
    int windowSize, 
    int nx_Interface, int ny_Interface
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nx_Interface && windowSize <= j && j < ny_Interface - windowSize) {
        int ny_MHD = IdealMHD2DConst::device_ny_MHD;
        int indexMHD = indexOfInterfaceStartInMHD
                     + j - (ny_Interface - interfaceLength) + i * ny_MHD;
        int indexForCopy = j + i * ny_Interface;
        ConservationParameter convolvedU;
        int windowSizeX = min(min(i, nx_Interface - 1 - i), windowSize);
        int windowSizeY = min(min(j, ny_Interface - 1 - j), windowSize);

        for (int sizeX = -windowSizeX; sizeX <= windowSizeX; sizeX++) {
            for (int sizeY = -windowSizeY; sizeY <= windowSizeY; sizeY++) {
                convolvedU = convolvedU + (1.0 / (2.0 * windowSizeX + 1.0) / (2.0 * windowSizeY + 1.0))
                           * tmpU[indexForCopy + sizeX * ny_Interface + sizeY];
            }
        }

        U[indexMHD] = convolvedU;

        if (j == ny_Interface - windowSize - 1) {
            for (int tmp = 1; tmp <= windowSize; tmp++) {
                U[indexMHD + tmp] = convolvedU;
            }
        }
    }
}


void InterfaceNoiseRemover2D::convolve_lower_magneticField(
    thrust::device_vector<MagneticField>& B
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx_Interface + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny_Interface + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyFields_lower_kernel<MagneticField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(tmpB.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize, 
        nx_Interface, ny_Interface
    );

    cudaDeviceSynchronize();
    

    convolveFields_lower_kernel<MagneticField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpB.data()), 
        thrust::raw_pointer_cast(B.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize, 
        nx_Interface, ny_Interface
    );

    cudaDeviceSynchronize();
}


void InterfaceNoiseRemover2D::convolve_lower_electricField(
    thrust::device_vector<ElectricField>& E
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx_Interface + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny_Interface + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyFields_lower_kernel<ElectricField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(E.data()), 
        thrust::raw_pointer_cast(tmpE.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize, 
        nx_Interface, ny_Interface
    );

    cudaDeviceSynchronize();
    

    convolveFields_lower_kernel<ElectricField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpE.data()), 
        thrust::raw_pointer_cast(E.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize, 
        nx_Interface, ny_Interface
    );

    cudaDeviceSynchronize();
}


void InterfaceNoiseRemover2D::convolve_lower_currentField(
    thrust::device_vector<CurrentField>& current
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx_Interface + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny_Interface + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyFields_lower_kernel<CurrentField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(current.data()), 
        thrust::raw_pointer_cast(tmpCurrent.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize, 
        nx_Interface, ny_Interface
    );

    cudaDeviceSynchronize();
    

    convolveFields_lower_kernel<CurrentField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpCurrent.data()), 
        thrust::raw_pointer_cast(current.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize, 
        nx_Interface, ny_Interface
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
    dim3 blocksPerGrid((nx_Interface + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny_Interface + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyFields_lower_kernel<ZerothMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(zerothMomentIon.data()), 
        thrust::raw_pointer_cast(tmpZerothMoment.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize, 
        nx_Interface, ny_Interface
    );

    cudaDeviceSynchronize();

    convolveFields_lower_kernel<ZerothMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpZerothMoment.data()), 
        thrust::raw_pointer_cast(zerothMomentIon.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize, 
        nx_Interface, ny_Interface
    );

    cudaDeviceSynchronize();


    copyFields_lower_kernel<ZerothMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(zerothMomentElectron.data()), 
        thrust::raw_pointer_cast(tmpZerothMoment.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize, 
        nx_Interface, ny_Interface
    );

    cudaDeviceSynchronize();

    convolveFields_lower_kernel<ZerothMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpZerothMoment.data()), 
        thrust::raw_pointer_cast(zerothMomentElectron.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize, 
        nx_Interface, ny_Interface
    );
    
    cudaDeviceSynchronize();

    //////////

    copyFields_lower_kernel<FirstMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(firstMomentIon.data()), 
        thrust::raw_pointer_cast(tmpFirstMoment.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize, 
        nx_Interface, ny_Interface
    );

    cudaDeviceSynchronize();

    convolveFields_lower_kernel<FirstMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpFirstMoment.data()), 
        thrust::raw_pointer_cast(firstMomentIon.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize, 
        nx_Interface, ny_Interface
    );
    
    cudaDeviceSynchronize();


    copyFields_lower_kernel<FirstMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(firstMomentElectron.data()), 
        thrust::raw_pointer_cast(tmpFirstMoment.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize, 
        nx_Interface, ny_Interface
    );

    cudaDeviceSynchronize();

    convolveFields_lower_kernel<FirstMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpFirstMoment.data()), 
        thrust::raw_pointer_cast(firstMomentElectron.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize, 
        nx_Interface, ny_Interface
    );
    
    cudaDeviceSynchronize();
}


void InterfaceNoiseRemover2D::convolveU_lower(
    thrust::device_vector<ConservationParameter>& U 
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx_Interface + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny_Interface + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyU_lower_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(tmpU.data()),
        indexOfInterfaceStartInMHD, 
        interfaceLength, 
        windowSize, 
        nx_Interface, ny_Interface
    );

    cudaDeviceSynchronize();

    convolveU_lower_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpU.data()), 
        thrust::raw_pointer_cast(U.data()), 
        indexOfInterfaceStartInMHD, 
        interfaceLength, 
        windowSize, 
        nx_Interface, ny_Interface
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
    int windowSize, 
    int nx_Interface, int ny_Interface
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nx_Interface && j < ny_Interface) {
        int ny_PIC = PIC2DConst::device_ny_PIC;
        int indexPIC = indexOfInterfaceStartInPIC
                     + j - (ny_Interface - interfaceLength) + i * ny_PIC;
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
    int windowSize, 
    int nx_Interface, int ny_Interface
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nx_Interface && windowSize <= j && j < ny_Interface - windowSize) {
        int ny_PIC = PIC2DConst::device_ny_PIC;
        int indexPIC = indexOfInterfaceStartInPIC
                     + j - (ny_Interface - interfaceLength) + i * ny_PIC;
        int indexForCopy = j + i * ny_Interface;
        FieldType convolvedField; 
        int windowSizeX = min(min(i, nx_Interface - 1 - i), windowSize);
        int windowSizeY = min(min(j, ny_Interface - 1 - j), windowSize);

        for (int sizeX = -windowSizeX; sizeX <= windowSizeX; sizeX++) {
            for (int sizeY = -windowSizeY; sizeY <= windowSizeY; sizeY++) {
                convolvedField = convolvedField + (1.0 / (2.0 * windowSizeX + 1.0) / (2.0 * windowSizeY + 1.0))
                               * tmpField[indexForCopy + sizeX * ny_Interface + sizeY];
            }
        }
        
        field[indexPIC] = convolvedField;

        if (j == ny_Interface - windowSize - 1) {
            for (int tmp = 1; tmp <= windowSize; tmp++) {
                field[indexPIC + tmp] = convolvedField;
            }
        }
    }
}


__global__ void copyU_upper_kernel(
    const ConservationParameter* U, 
    ConservationParameter* tmpU, 
    int indexOfInterfaceStartInMHD, 
    int interfaceLength, 
    int windowSize, 
    int nx_Interface, int ny_Interface
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nx_Interface && j < ny_Interface) {
        int ny_MHD = IdealMHD2DConst::device_ny_MHD;
        int indexMHD = indexOfInterfaceStartInMHD + j + i * ny_MHD;
        int indexForCopy = j + i * ny_Interface;

        tmpU[indexForCopy] = U[indexMHD];
    }
}


__global__ void convolveU_upper_kernel(
    const ConservationParameter* tmpU, 
    ConservationParameter* U, 
    int indexOfInterfaceStartInMHD, 
    int interfaceLength, 
    int windowSize, 
    int nx_Interface, int ny_Interface
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nx_Interface && windowSize <= j && j < ny_Interface - windowSize) {
        int ny_MHD = IdealMHD2DConst::device_ny_MHD;
        int indexMHD = indexOfInterfaceStartInMHD + j + i * ny_MHD;
        int indexForCopy = j + i * ny_Interface;
        ConservationParameter convolvedU;
        int windowSizeX = min(min(i, nx_Interface - 1 - i), windowSize);
        int windowSizeY = min(min(j, ny_Interface - 1 - j), windowSize);

        for (int sizeX = -windowSizeX; sizeX <= windowSizeX; sizeX++) {
            for (int sizeY = -windowSizeY; sizeY <= windowSizeY; sizeY++) {
                convolvedU = convolvedU + (1.0 / (2.0 * windowSizeX + 1.0) / (2.0 * windowSizeY + 1.0))
                           * tmpU[indexForCopy + sizeX * ny_Interface + sizeY];
            }
        }

        U[indexMHD] = convolvedU;

        if (j == windowSize) {
            for (int tmp = 1; tmp <= windowSize; tmp++) {
                U[indexMHD - tmp] = convolvedU;
            }
        }
    }
}


void InterfaceNoiseRemover2D::convolve_upper_magneticField(
    thrust::device_vector<MagneticField>& B
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx_Interface + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny_Interface + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyFields_upper_kernel<MagneticField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(tmpB.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize, 
        nx_Interface, ny_Interface
    );

    cudaDeviceSynchronize();
    

    convolveFields_upper_kernel<MagneticField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpB.data()), 
        thrust::raw_pointer_cast(B.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize, 
        nx_Interface, ny_Interface
    );

    cudaDeviceSynchronize();
}


void InterfaceNoiseRemover2D::convolve_upper_electricField(
    thrust::device_vector<ElectricField>& E
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx_Interface + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny_Interface + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyFields_upper_kernel<ElectricField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(E.data()), 
        thrust::raw_pointer_cast(tmpE.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize, 
        nx_Interface, ny_Interface
    );

    cudaDeviceSynchronize();
    

    convolveFields_upper_kernel<ElectricField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpE.data()), 
        thrust::raw_pointer_cast(E.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize, 
        nx_Interface, ny_Interface
    );

    cudaDeviceSynchronize();
}


void InterfaceNoiseRemover2D::convolve_upper_currentField(
    thrust::device_vector<CurrentField>& current
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx_Interface + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny_Interface + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyFields_upper_kernel<CurrentField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(current.data()), 
        thrust::raw_pointer_cast(tmpCurrent.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize, 
        nx_Interface, ny_Interface
    );

    cudaDeviceSynchronize();
    

    convolveFields_upper_kernel<CurrentField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpCurrent.data()), 
        thrust::raw_pointer_cast(current.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize, 
        nx_Interface, ny_Interface
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
    dim3 blocksPerGrid((nx_Interface + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny_Interface + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyFields_upper_kernel<ZerothMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(zerothMomentIon.data()), 
        thrust::raw_pointer_cast(tmpZerothMoment.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize, 
        nx_Interface, ny_Interface
    );

    cudaDeviceSynchronize();

    convolveFields_upper_kernel<ZerothMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpZerothMoment.data()), 
        thrust::raw_pointer_cast(zerothMomentIon.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize, 
        nx_Interface, ny_Interface
    );

    cudaDeviceSynchronize();


    copyFields_upper_kernel<ZerothMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(zerothMomentElectron.data()), 
        thrust::raw_pointer_cast(tmpZerothMoment.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize, 
        nx_Interface, ny_Interface
    );

    cudaDeviceSynchronize();

    convolveFields_upper_kernel<ZerothMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpZerothMoment.data()), 
        thrust::raw_pointer_cast(zerothMomentElectron.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize, 
        nx_Interface, ny_Interface
    );
    
    cudaDeviceSynchronize();

    //////////

    copyFields_upper_kernel<FirstMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(firstMomentIon.data()), 
        thrust::raw_pointer_cast(tmpFirstMoment.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize, 
        nx_Interface, ny_Interface
    );

    cudaDeviceSynchronize();

    convolveFields_upper_kernel<FirstMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpFirstMoment.data()), 
        thrust::raw_pointer_cast(firstMomentIon.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize, 
        nx_Interface, ny_Interface
    );
    
    cudaDeviceSynchronize();


    copyFields_upper_kernel<FirstMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(firstMomentElectron.data()), 
        thrust::raw_pointer_cast(tmpFirstMoment.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize, 
        nx_Interface, ny_Interface
    );

    cudaDeviceSynchronize();

    convolveFields_upper_kernel<FirstMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpFirstMoment.data()), 
        thrust::raw_pointer_cast(firstMomentElectron.data()), 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSize, 
        nx_Interface, ny_Interface
    );
    
    cudaDeviceSynchronize();
}


void InterfaceNoiseRemover2D::convolveU_upper(
    thrust::device_vector<ConservationParameter>& U 
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx_Interface + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny_Interface + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyU_upper_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(tmpU.data()),
        indexOfInterfaceStartInMHD, 
        interfaceLength, 
        windowSize, 
        nx_Interface, ny_Interface
    );

    cudaDeviceSynchronize();

    convolveU_upper_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpU.data()), 
        thrust::raw_pointer_cast(U.data()), 
        indexOfInterfaceStartInMHD, 
        interfaceLength, 
        windowSize, 
        nx_Interface, ny_Interface
    );

    cudaDeviceSynchronize();
}

