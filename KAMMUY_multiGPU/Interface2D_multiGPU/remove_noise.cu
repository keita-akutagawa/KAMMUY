#include "remove_noise.hpp"


InterfaceNoiseRemover2D::InterfaceNoiseRemover2D(
    IdealMHD2DMPI::MPIInfo& mPIInfoMHD, 
    PIC2DMPI::MPIInfo& mPIInfoPIC, 
    int indexOfInterfaceStartInMHD_lower, 
    int indexOfInterfaceStartInPIC_lower, 
    int indexOfInterfaceStartInMHD_upper, 
    int indexOfInterfaceStartInPIC_upper, 
    int interfaceLength, 
    int windowSizeForConvolution, 
    int nxInterface, int nyInterface
)
    : mPIInfoMHD(mPIInfoMHD), 
      mPIInfoPIC(mPIInfoPIC), 

      indexOfInterfaceStartInMHD_lower(indexOfInterfaceStartInMHD_lower), 
      indexOfInterfaceStartInPIC_lower(indexOfInterfaceStartInPIC_lower), 
      indexOfInterfaceStartInMHD_upper(indexOfInterfaceStartInMHD_upper), 
      indexOfInterfaceStartInPIC_upper(indexOfInterfaceStartInPIC_upper), 
      interfaceLength(interfaceLength), 
      windowSize(windowSizeForConvolution), 
      nxInterface(nxInterface), 
      nyInterface(nyInterface), 

      tmpB(nxInterface * nyInterface), 
      tmpE(nxInterface * nyInterface), 
      tmpCurrent(nxInterface * nyInterface), 
      tmpZerothMoment(nxInterface * nyInterface), 
      tmpFirstMoment(nxInterface * nyInterface), 
      tmpU(nxInterface * nyInterface)
{

    cudaMalloc(&device_mPIInfoMHD, sizeof(IdealMHD2DMPI::MPIInfo));
    cudaMemcpy(device_mPIInfoMHD, &mPIInfoMHD, sizeof(IdealMHD2DMPI::MPIInfo), cudaMemcpyHostToDevice);
    cudaMalloc(&device_mPIInfoPIC, sizeof(PIC2DMPI::MPIInfo));
    cudaMemcpy(device_mPIInfoPIC, &mPIInfoPIC, sizeof(PIC2DMPI::MPIInfo), cudaMemcpyHostToDevice);
    
}


template <typename FieldType>
__global__ void copyFields_kernel(
    const FieldType* field, 
    FieldType* tmpField, 
    int indexOfInterfaceStartInPIC, 
    int interfaceLength, 
    int windowSize, 
    int nxInterface, int nyInterface, 
    int localSizeXPIC, int localSizeYPIC, 
    int localSizeXMHD, int localSizeYMHD, 
    bool isLower, bool isUpper
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nxInterface && j < nyInterface) {
        int indexForCopy = j + i * nyInterface;
        int indexPIC;
        if (isLower) {
            indexPIC = indexOfInterfaceStartInPIC + j + i * localSizeYPIC;
        }
        if (isUpper) {
            indexPIC = indexOfInterfaceStartInPIC
                     + j - (nyInterface - interfaceLength) + i * localSizeYPIC;
        }

        tmpField[indexForCopy] = field[indexPIC];
    }
}


template <typename FieldType>
__global__ void convolveFields_kernel(
    const FieldType* tmpField, 
    FieldType* field, 
    int indexOfInterfaceStartInPIC, 
    int interfaceLength, 
    int windowSize, 
    int nxInterface, int nyInterface, 
    int localSizeXPIC, int localSizeYPIC, 
    int localSizeXMHD, int localSizeYMHD, 
    bool isLower, bool isUpper
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nxInterface && windowSize <= j && j < nyInterface - windowSize) {
        int indexForCopy = j + i * nyInterface;
        int indexPIC;
        if (isLower) {
            indexPIC = indexOfInterfaceStartInPIC + j + i * localSizeYPIC;
        }
        if (isUpper) {
            indexPIC = indexOfInterfaceStartInPIC
                     + j - (nyInterface - interfaceLength) + i * localSizeYPIC;
        }
        
        FieldType convolvedField; 
        int windowSizeX = min(min(i, nxInterface - 1 - i), windowSize);
        int windowSizeY = min(min(j, nyInterface - 1 - j), windowSize);

        for (int sizeX = -windowSizeX; sizeX <= windowSizeX; sizeX++) {
            for (int sizeY = -windowSizeY; sizeY <= windowSizeY; sizeY++) {
                convolvedField = convolvedField + (1.0 / (2.0 * windowSizeX + 1.0) / (2.0 * windowSizeY + 1.0))
                               * tmpField[indexForCopy + sizeX * nyInterface + sizeY];
            }
        }
        
        field[indexPIC] = convolvedField;

        if (isLower) {
            if (j == windowSize) {
                for (int tmp = 1; tmp <= windowSize; tmp++) {
                    field[indexPIC - tmp] = convolvedField;
                }
            }
        }
        if (isUpper) {
            if (j == nyInterface - windowSize - 1) {
                for (int tmp = 1; tmp <= windowSize; tmp++) {
                    field[indexPIC + tmp] = convolvedField;
                }
            }
        }
    }
}


void InterfaceNoiseRemover2D::convolve_magneticField(
    thrust::device_vector<MagneticField>& B, 
    bool isLower, bool isUpper
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nxInterface + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (nyInterface + threadsPerBlock.y - 1) / threadsPerBlock.y);

    if (isLower) {
        copyFields_kernel<MagneticField><<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(B.data()), 
            thrust::raw_pointer_cast(tmpB.data()), 
            indexOfInterfaceStartInPIC_lower, 
            interfaceLength, 
            windowSize, 
            nxInterface, nyInterface, 
            mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY, 
            mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY, 
            isLower, isUpper
        );
        cudaDeviceSynchronize();
    
        convolveFields_kernel<MagneticField><<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(tmpB.data()), 
            thrust::raw_pointer_cast(B.data()), 
            indexOfInterfaceStartInPIC_lower, 
            interfaceLength, 
            windowSize, 
            nxInterface, nyInterface, 
            mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY, 
            mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY, 
            isLower, isUpper
        );
        cudaDeviceSynchronize();
    }

    if (isUpper) {
        copyFields_kernel<MagneticField><<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(B.data()), 
            thrust::raw_pointer_cast(tmpB.data()), 
            indexOfInterfaceStartInPIC_upper, 
            interfaceLength, 
            windowSize, 
            nxInterface, nyInterface, 
            mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY, 
            mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY, 
            isLower, isUpper
        );
        cudaDeviceSynchronize();
    
        convolveFields_kernel<MagneticField><<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(tmpB.data()), 
            thrust::raw_pointer_cast(B.data()), 
            indexOfInterfaceStartInPIC_upper, 
            interfaceLength, 
            windowSize, 
            nxInterface, nyInterface, 
            mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY, 
            mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY, 
            isLower, isUpper
        );
        cudaDeviceSynchronize();
    }
}


void InterfaceNoiseRemover2D::convolve_electricField(
    thrust::device_vector<ElectricField>& E, 
    bool isLower, bool isUpper
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nxInterface + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (nyInterface + threadsPerBlock.y - 1) / threadsPerBlock.y);

    if (isLower) {
        copyFields_kernel<ElectricField><<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(E.data()), 
            thrust::raw_pointer_cast(tmpE.data()), 
            indexOfInterfaceStartInPIC_lower, 
            interfaceLength, 
            windowSize, 
            nxInterface, nyInterface, 
            mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY, 
            mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY, 
            isLower, isUpper
        );
        cudaDeviceSynchronize();

        convolveFields_kernel<ElectricField><<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(tmpE.data()), 
            thrust::raw_pointer_cast(E.data()), 
            indexOfInterfaceStartInPIC_lower, 
            interfaceLength, 
            windowSize, 
            nxInterface, nyInterface, 
            mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY, 
            mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY, 
            isLower, isUpper
        );
        cudaDeviceSynchronize();
    }

    if (isUpper) {
        copyFields_kernel<ElectricField><<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(E.data()), 
            thrust::raw_pointer_cast(tmpE.data()), 
            indexOfInterfaceStartInPIC_upper, 
            interfaceLength, 
            windowSize, 
            nxInterface, nyInterface, 
            mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY, 
            mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY, 
            isLower, isUpper
        );
        cudaDeviceSynchronize();

        convolveFields_kernel<ElectricField><<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(tmpE.data()), 
            thrust::raw_pointer_cast(E.data()), 
            indexOfInterfaceStartInPIC_upper, 
            interfaceLength, 
            windowSize, 
            nxInterface, nyInterface, 
            mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY, 
            mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY, 
            isLower, isUpper
        );
        cudaDeviceSynchronize();
    }
}


void InterfaceNoiseRemover2D::convolve_currentField(
    thrust::device_vector<CurrentField>& current, 
    bool isLower, bool isUpper
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nxInterface + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (nyInterface + threadsPerBlock.y - 1) / threadsPerBlock.y);

    if (isLower) {
        copyFields_kernel<CurrentField><<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(current.data()), 
            thrust::raw_pointer_cast(tmpCurrent.data()), 
            indexOfInterfaceStartInPIC_lower, 
            interfaceLength, 
            windowSize, 
            nxInterface, nyInterface, 
            mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY, 
            mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY, 
            isLower, isUpper
        );
        cudaDeviceSynchronize();
        
        convolveFields_kernel<CurrentField><<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(tmpCurrent.data()), 
            thrust::raw_pointer_cast(current.data()), 
            indexOfInterfaceStartInPIC_lower, 
            interfaceLength, 
            windowSize, 
            nxInterface, nyInterface, 
            mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY, 
            mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY, 
            isLower, isUpper
        );
        cudaDeviceSynchronize();
    }

    if (isUpper) {
        copyFields_kernel<CurrentField><<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(current.data()), 
            thrust::raw_pointer_cast(tmpCurrent.data()), 
            indexOfInterfaceStartInPIC_upper, 
            interfaceLength, 
            windowSize, 
            nxInterface, nyInterface, 
            mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY, 
            mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY, 
            isLower, isUpper
        );
        cudaDeviceSynchronize();
        
        convolveFields_kernel<CurrentField><<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(tmpCurrent.data()), 
            thrust::raw_pointer_cast(current.data()), 
            indexOfInterfaceStartInPIC_upper, 
            interfaceLength, 
            windowSize, 
            nxInterface, nyInterface, 
            mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY, 
            mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY, 
            isLower, isUpper
        );
        cudaDeviceSynchronize();
    }
}


void InterfaceNoiseRemover2D::convolveMoments(
    thrust::device_vector<ZerothMoment>& zerothMomentIon, 
    thrust::device_vector<ZerothMoment>& zerothMomentElectron, 
    thrust::device_vector<FirstMoment>& firstMomentIon, 
    thrust::device_vector<FirstMoment>& firstMomentElectron, 
    bool isLower, bool isUpper
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nxInterface + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (nyInterface + threadsPerBlock.y - 1) / threadsPerBlock.y);

    if (isLower) {
        copyFields_kernel<ZerothMoment><<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(zerothMomentIon.data()), 
            thrust::raw_pointer_cast(tmpZerothMoment.data()), 
            indexOfInterfaceStartInPIC_lower, 
            interfaceLength, 
            windowSize, 
            nxInterface, nyInterface, 
            mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY, 
            mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY, 
            isLower, isUpper
        );
        cudaDeviceSynchronize();

        convolveFields_kernel<ZerothMoment><<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(tmpZerothMoment.data()), 
            thrust::raw_pointer_cast(zerothMomentIon.data()), 
            indexOfInterfaceStartInPIC_lower, 
            interfaceLength, 
            windowSize, 
            nxInterface, nyInterface, 
            mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY, 
            mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY, 
            isLower, isUpper
        );
        cudaDeviceSynchronize();

        copyFields_kernel<FirstMoment><<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(firstMomentIon.data()), 
            thrust::raw_pointer_cast(tmpFirstMoment.data()), 
            indexOfInterfaceStartInPIC_lower, 
            interfaceLength, 
            windowSize, 
            nxInterface, nyInterface, 
            mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY, 
            mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY, 
            isLower, isUpper
        );
        cudaDeviceSynchronize();

        convolveFields_kernel<FirstMoment><<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(tmpFirstMoment.data()), 
            thrust::raw_pointer_cast(firstMomentIon.data()), 
            indexOfInterfaceStartInPIC_lower, 
            interfaceLength, 
            windowSize, 
            nxInterface, nyInterface, 
            mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY, 
            mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY, 
            isLower, isUpper
        );
        cudaDeviceSynchronize();

        copyFields_kernel<ZerothMoment><<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(zerothMomentElectron.data()), 
            thrust::raw_pointer_cast(tmpZerothMoment.data()), 
            indexOfInterfaceStartInPIC_lower, 
            interfaceLength, 
            windowSize, 
            nxInterface, nyInterface, 
            mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY, 
            mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY, 
            isLower, isUpper
        );
        cudaDeviceSynchronize();

        convolveFields_kernel<ZerothMoment><<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(tmpZerothMoment.data()), 
            thrust::raw_pointer_cast(zerothMomentElectron.data()), 
            indexOfInterfaceStartInPIC_lower, 
            interfaceLength, 
            windowSize, 
            nxInterface, nyInterface, 
            mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY, 
            mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY, 
            isLower, isUpper
        );
        cudaDeviceSynchronize();

        copyFields_kernel<FirstMoment><<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(firstMomentElectron.data()), 
            thrust::raw_pointer_cast(tmpFirstMoment.data()), 
            indexOfInterfaceStartInPIC_lower, 
            interfaceLength, 
            windowSize, 
            nxInterface, nyInterface, 
            mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY, 
            mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY, 
            isLower, isUpper
        );
        cudaDeviceSynchronize();

        convolveFields_kernel<FirstMoment><<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(tmpFirstMoment.data()), 
            thrust::raw_pointer_cast(firstMomentElectron.data()), 
            indexOfInterfaceStartInPIC_lower, 
            interfaceLength, 
            windowSize, 
            nxInterface, nyInterface, 
            mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY, 
            mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY, 
            isLower, isUpper
        );
        cudaDeviceSynchronize();
    }

    if (isUpper) {
        copyFields_kernel<ZerothMoment><<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(zerothMomentIon.data()), 
            thrust::raw_pointer_cast(tmpZerothMoment.data()), 
            indexOfInterfaceStartInPIC_upper, 
            interfaceLength, 
            windowSize, 
            nxInterface, nyInterface, 
            mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY, 
            mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY, 
            isLower, isUpper
        );
        cudaDeviceSynchronize();

        convolveFields_kernel<ZerothMoment><<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(tmpZerothMoment.data()), 
            thrust::raw_pointer_cast(zerothMomentIon.data()), 
            indexOfInterfaceStartInPIC_upper, 
            interfaceLength, 
            windowSize, 
            nxInterface, nyInterface, 
            mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY, 
            mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY, 
            isLower, isUpper
        );
        cudaDeviceSynchronize();

        copyFields_kernel<FirstMoment><<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(firstMomentIon.data()), 
            thrust::raw_pointer_cast(tmpFirstMoment.data()), 
            indexOfInterfaceStartInPIC_upper, 
            interfaceLength, 
            windowSize, 
            nxInterface, nyInterface, 
            mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY, 
            mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY, 
            isLower, isUpper
        );
        cudaDeviceSynchronize();

        convolveFields_kernel<FirstMoment><<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(tmpFirstMoment.data()), 
            thrust::raw_pointer_cast(firstMomentIon.data()), 
            indexOfInterfaceStartInPIC_upper, 
            interfaceLength, 
            windowSize, 
            nxInterface, nyInterface, 
            mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY, 
            mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY, 
            isLower, isUpper
        );
        cudaDeviceSynchronize();

        copyFields_kernel<ZerothMoment><<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(zerothMomentElectron.data()), 
            thrust::raw_pointer_cast(tmpZerothMoment.data()), 
            indexOfInterfaceStartInPIC_upper, 
            interfaceLength, 
            windowSize, 
            nxInterface, nyInterface, 
            mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY, 
            mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY, 
            isLower, isUpper
        );
        cudaDeviceSynchronize();

        convolveFields_kernel<ZerothMoment><<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(tmpZerothMoment.data()), 
            thrust::raw_pointer_cast(zerothMomentElectron.data()), 
            indexOfInterfaceStartInPIC_upper, 
            interfaceLength, 
            windowSize, 
            nxInterface, nyInterface, 
            mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY, 
            mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY, 
            isLower, isUpper
        );
        cudaDeviceSynchronize();

        copyFields_kernel<FirstMoment><<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(firstMomentElectron.data()), 
            thrust::raw_pointer_cast(tmpFirstMoment.data()), 
            indexOfInterfaceStartInPIC_upper, 
            interfaceLength, 
            windowSize, 
            nxInterface, nyInterface, 
            mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY, 
            mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY, 
            isLower, isUpper
        );
        cudaDeviceSynchronize();

        convolveFields_kernel<FirstMoment><<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(tmpFirstMoment.data()), 
            thrust::raw_pointer_cast(firstMomentElectron.data()), 
            indexOfInterfaceStartInPIC_upper, 
            interfaceLength, 
            windowSize, 
            nxInterface, nyInterface, 
            mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY, 
            mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY, 
            isLower, isUpper
        );
        cudaDeviceSynchronize();
    }
}



__global__ void copyU_kernel(
    const ConservationParameter* U, 
    ConservationParameter* tmpU, 
    int indexOfInterfaceStartInMHD, 
    int interfaceLength, 
    int windowSize, 
    int nxInterface, int nyInterface, 
    int localSizeXPIC, int localSizeYPIC, 
    int localSizeXMHD, int localSizeYMHD, 
    bool isLower, bool isUpper
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nxInterface && j < nyInterface) {
        int indexForCopy = j + i * nyInterface;
        int indexMHD;
        if (isLower) {
            indexMHD = indexOfInterfaceStartInMHD
                     + j - (nyInterface - interfaceLength) + i * localSizeYMHD;
        }
        if (isUpper) {
            indexMHD = indexOfInterfaceStartInMHD + j + i * localSizeYMHD;
        }

        tmpU[indexForCopy] = U[indexMHD];
    }
}


__global__ void convolveU_kernel(
    const ConservationParameter* tmpU, 
    ConservationParameter* U, 
    int indexOfInterfaceStartInMHD, 
    int interfaceLength, 
    int windowSize, 
    int nxInterface, int nyInterface, 
    int localSizeXPIC, int localSizeYPIC, 
    int localSizeXMHD, int localSizeYMHD, 
    bool isLower, bool isUpper
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nxInterface && windowSize <= j && j < nyInterface - windowSize) {
        int indexForCopy = j + i * nyInterface;
        int indexMHD;
        if (isLower) {
            indexMHD = indexOfInterfaceStartInMHD
                     + j - (nyInterface - interfaceLength) + i * localSizeYMHD;
        }
        if (isUpper) {
            indexMHD = indexOfInterfaceStartInMHD + j + i * localSizeYMHD;
        }
        
        ConservationParameter convolvedU;
        int windowSizeX = min(min(i, nxInterface - 1 - i), windowSize);
        int windowSizeY = min(min(j, nyInterface - 1 - j), windowSize);

        for (int sizeX = -windowSizeX; sizeX <= windowSizeX; sizeX++) {
            for (int sizeY = -windowSizeY; sizeY <= windowSizeY; sizeY++) {
                convolvedU = convolvedU + (1.0 / (2.0 * windowSizeX + 1.0) / (2.0 * windowSizeY + 1.0))
                           * tmpU[indexForCopy + sizeX * nyInterface + sizeY];
            }
        }

        U[indexMHD] = convolvedU;

        if (isLower) {
            if (j == windowSize) {
                for (int tmp = 1; tmp <= windowSize; tmp++) {
                    U[indexMHD + tmp] = convolvedU;
                }
            }
        }
        if (isUpper) {
            if (j == windowSize) {
                for (int tmp = 1; tmp <= windowSize; tmp++) {
                    U[indexMHD - tmp] = convolvedU;
                }
            }
        }
    }
}


void InterfaceNoiseRemover2D::convolveU(
    thrust::device_vector<ConservationParameter>& U , 
    bool isLower, bool isUpper
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nxInterface + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (nyInterface + threadsPerBlock.y - 1) / threadsPerBlock.y);

    if (isLower) {
        copyU_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(U.data()), 
            thrust::raw_pointer_cast(tmpU.data()),
            indexOfInterfaceStartInMHD_lower, 
            interfaceLength, 
            windowSize, 
            nxInterface, nyInterface, 
            mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY, 
            mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY, 
            isLower, isUpper
        );
        cudaDeviceSynchronize();

        convolveU_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(tmpU.data()), 
            thrust::raw_pointer_cast(U.data()), 
            indexOfInterfaceStartInMHD_lower, 
            interfaceLength, 
            windowSize, 
            nxInterface, nyInterface, 
            mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY, 
            mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY, 
            isLower, isUpper
        );
        cudaDeviceSynchronize();
    }

    if (isUpper) {
        copyU_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(U.data()), 
            thrust::raw_pointer_cast(tmpU.data()),
            indexOfInterfaceStartInMHD_upper, 
            interfaceLength, 
            windowSize, 
            nxInterface, nyInterface, 
            mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY, 
            mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY, 
            isLower, isUpper
        );
        cudaDeviceSynchronize();

        convolveU_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(tmpU.data()), 
            thrust::raw_pointer_cast(U.data()), 
            indexOfInterfaceStartInMHD_upper, 
            interfaceLength, 
            windowSize, 
            nxInterface, nyInterface, 
            mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY, 
            mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY, 
            isLower, isUpper
        );
        cudaDeviceSynchronize();
    }
}

