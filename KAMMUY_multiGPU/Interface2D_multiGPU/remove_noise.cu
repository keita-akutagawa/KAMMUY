#include "remove_noise.hpp"


InterfaceNoiseRemover2D::InterfaceNoiseRemover2D(
    IdealMHD2DMPI::MPIInfo& mPIInfoMHD, 
    PIC2DMPI::MPIInfo& mPIInfoPIC, 
    int indexOfInterfaceStartInMHD, 
    int indexOfInterfaceStartInPIC, 
    int nxInterface, int nyInterface
)
    : mPIInfoMHD(mPIInfoMHD), 
      mPIInfoPIC(mPIInfoPIC), 

      indexOfInterfaceStartInMHD(indexOfInterfaceStartInMHD), 
      indexOfInterfaceStartInPIC(indexOfInterfaceStartInPIC), 
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
__global__ void copyFieldsPIC_kernel(
    const FieldType* field, 
    FieldType* tmpField, 
    int indexOfInterfaceStartInPIC, 
    int nxInterface, int nyInterface, 
    int localSizeXPIC, int localSizeYPIC
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nxInterface && j < nyInterface) {
        int indexForCopy = j + i * nyInterface;
        int indexPIC = indexOfInterfaceStartInPIC + j + i * localSizeYPIC;

        tmpField[indexForCopy] = field[indexPIC];
    }
}


template <typename FieldType>
__global__ void convolveFields_kernel(
    const FieldType* tmpField, 
    FieldType* field, 
    int indexOfInterfaceStartInPIC, 
    int nxInterface, int nyInterface, 
    int localSizeXPIC, int localSizeYPIC
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < nxInterface - 1 && 0 < j && j < nyInterface - 1) {
        int indexForCopy = j + i * nyInterface;
        int indexPIC = indexOfInterfaceStartInPIC + j + i * localSizeYPIC;
        
        FieldType convolvedField; 

        convolvedField = 0.5 * tmpField[indexForCopy] + 0.25 * (tmpField[indexForCopy + nyInterface] + tmpField[indexForCopy - nyInterface]);
        convolvedField = 0.5 * tmpField[indexForCopy] + 0.25 * (tmpField[indexForCopy + 1] + tmpField[indexForCopy - 1]);
        
        field[indexPIC] = convolvedField;
    }
}


void InterfaceNoiseRemover2D::convolve_magneticField(
    thrust::device_vector<MagneticField>& B
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nxInterface + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (nyInterface + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyFieldsPIC_kernel<MagneticField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(tmpB.data()), 
        indexOfInterfaceStartInPIC, 
        nxInterface, nyInterface, 
        mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY
    );
    cudaDeviceSynchronize();
    
    convolveFields_kernel<MagneticField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpB.data()), 
        thrust::raw_pointer_cast(B.data()), 
        indexOfInterfaceStartInPIC, 
        nxInterface, nyInterface, 
        mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY
    );
    cudaDeviceSynchronize();

}


void InterfaceNoiseRemover2D::convolve_electricField(
    thrust::device_vector<ElectricField>& E
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nxInterface + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (nyInterface + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyFieldsPIC_kernel<ElectricField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(E.data()), 
        thrust::raw_pointer_cast(tmpE.data()), 
        indexOfInterfaceStartInPIC, 
        nxInterface, nyInterface, 
        mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY
    );
    cudaDeviceSynchronize();

    convolveFields_kernel<ElectricField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpE.data()), 
        thrust::raw_pointer_cast(E.data()), 
        indexOfInterfaceStartInPIC, 
        nxInterface, nyInterface, 
        mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY
    );
    cudaDeviceSynchronize();

}


void InterfaceNoiseRemover2D::convolve_currentField(
    thrust::device_vector<CurrentField>& current
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nxInterface + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (nyInterface + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyFieldsPIC_kernel<CurrentField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(current.data()), 
        thrust::raw_pointer_cast(tmpCurrent.data()), 
        indexOfInterfaceStartInPIC, 
        nxInterface, nyInterface, 
        mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY
    );
    cudaDeviceSynchronize();
    
    convolveFields_kernel<CurrentField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpCurrent.data()), 
        thrust::raw_pointer_cast(current.data()), 
        indexOfInterfaceStartInPIC, 
        nxInterface, nyInterface, 
        mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY
    );
    cudaDeviceSynchronize();

}


void InterfaceNoiseRemover2D::convolveMomentsOfOneSpecies(
    thrust::device_vector<ZerothMoment>& zerothMomentOfOneSpecies, 
    thrust::device_vector<FirstMoment>& firstMomentOfOneSpecies
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nxInterface + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (nyInterface + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyFieldsPIC_kernel<ZerothMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(zerothMomentOfOneSpecies.data()), 
        thrust::raw_pointer_cast(tmpZerothMoment.data()), 
        indexOfInterfaceStartInPIC, 
        nxInterface, nyInterface, 
        mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY
    );
    cudaDeviceSynchronize();

    convolveFields_kernel<ZerothMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpZerothMoment.data()), 
        thrust::raw_pointer_cast(zerothMomentOfOneSpecies.data()), 
        indexOfInterfaceStartInPIC, 
        nxInterface, nyInterface, 
        mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY
    );
    cudaDeviceSynchronize();

    copyFieldsPIC_kernel<FirstMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(firstMomentOfOneSpecies.data()), 
        thrust::raw_pointer_cast(tmpFirstMoment.data()), 
        indexOfInterfaceStartInPIC, 
        nxInterface, nyInterface, 
        mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY
    );
    cudaDeviceSynchronize();

    convolveFields_kernel<FirstMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpFirstMoment.data()), 
        thrust::raw_pointer_cast(firstMomentOfOneSpecies.data()), 
        indexOfInterfaceStartInPIC, 
        nxInterface, nyInterface, 
        mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY
    );
    cudaDeviceSynchronize();

}


void InterfaceNoiseRemover2D::convolveMoments(
    thrust::device_vector<ZerothMoment>& zerothMomentIon, 
    thrust::device_vector<ZerothMoment>& zerothMomentElectron, 
    thrust::device_vector<FirstMoment>& firstMomentIon, 
    thrust::device_vector<FirstMoment>& firstMomentElectron
)
{
    convolveMomentsOfOneSpecies(
        zerothMomentIon, firstMomentIon
    );
    convolveMomentsOfOneSpecies(
        zerothMomentElectron, firstMomentElectron
    );
}



__global__ void copyU_kernel(
    const ConservationParameter* U, 
    ConservationParameter* tmpU, 
    int indexOfInterfaceStartInMHD, 
    int nxInterface, int nyInterface, 
    int localSizeXMHD, int localSizeYMHD
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nxInterface && j < nyInterface) {
        int indexForCopy = j + i * nyInterface;
        int indexMHD = indexOfInterfaceStartInMHD + j + i * localSizeYMHD;

        tmpU[indexForCopy] = U[indexMHD];
    }
}


__global__ void convolveU_kernel(
    const ConservationParameter* tmpU, 
    ConservationParameter* U, 
    int indexOfInterfaceStartInMHD, 
    int nxInterface, int nyInterface, 
    int localSizeXMHD, int localSizeYMHD
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < nxInterface - 1 && 0 < j && j < nyInterface - 1) {
        int indexForCopy = j + i * nyInterface;
        int indexMHD = indexOfInterfaceStartInMHD + j + i * localSizeYMHD;
        
        ConservationParameter convolvedU;

        convolvedU = 0.5 * tmpU[indexForCopy] + 0.25 * (tmpU[indexForCopy + nyInterface] + tmpU[indexForCopy - nyInterface]);
        convolvedU = 0.5 * tmpU[indexForCopy] + 0.25 * (tmpU[indexForCopy + 1] + tmpU[indexForCopy - 1]);
        
        U[indexMHD] = convolvedU;
    }
}


void InterfaceNoiseRemover2D::convolveU(
    thrust::device_vector<ConservationParameter>& U
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nxInterface + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (nyInterface + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyU_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(tmpU.data()),
        indexOfInterfaceStartInMHD, 
        nxInterface, nyInterface, 
        mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY
    );
    cudaDeviceSynchronize();

    convolveU_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpU.data()), 
        thrust::raw_pointer_cast(U.data()), 
        indexOfInterfaceStartInMHD, 
        nxInterface, nyInterface, 
        mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY
    );
    cudaDeviceSynchronize();
}

