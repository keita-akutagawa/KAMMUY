#include "remove_noise.hpp"


InterfaceNoiseRemover2D::InterfaceNoiseRemover2D(
    IdealMHD2DMPI::MPIInfo& mPIInfoMHD, 
    PIC2DMPI::MPIInfo& mPIInfoPIC, 
    int indexOfConvolutionStartInMHD, 
    int indexOfConvolutionStartInPIC, 
    int localSizeXConvolution, int localSizeYConvolution
)
    : mPIInfoMHD(mPIInfoMHD), 
      mPIInfoPIC(mPIInfoPIC), 

      indexOfConvolutionStartInMHD(indexOfConvolutionStartInMHD), 
      indexOfConvolutionStartInPIC(indexOfConvolutionStartInPIC), 
      localSizeXConvolution(localSizeXConvolution), 
      localSizeYConvolution(localSizeYConvolution), 

      tmpB(localSizeXConvolution * localSizeYConvolution), 
      tmpE(localSizeXConvolution * localSizeYConvolution), 
      tmpCurrent(localSizeXConvolution * localSizeYConvolution), 
      tmpZerothMoment(localSizeXConvolution * localSizeYConvolution), 
      tmpFirstMoment(localSizeXConvolution * localSizeYConvolution), 
      tmpU(localSizeXConvolution * localSizeYConvolution)
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
    int indexOfConvolutionStartInPIC, 
    int localSizeXConvolution, int localSizeYConvolution, 
    int localSizeXPIC, int localSizeYPIC
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localSizeXConvolution && j < localSizeYConvolution) {
        int indexForCopy = j + i * localSizeYConvolution;
        int indexPIC = indexOfConvolutionStartInPIC + j + i * localSizeYPIC;

        tmpField[indexForCopy] = field[indexPIC];
    }
}


template <typename FieldType>
__global__ void convolveFields_kernel(
    const FieldType* tmpField, 
    FieldType* field, 
    int indexOfConvolutionStartInPIC, 
    int localSizeXConvolution, int localSizeYConvolution, 
    int localSizeXPIC, int localSizeYPIC
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < localSizeXConvolution - 1 && 0 < j && j < localSizeYConvolution - 1) {
        int indexForCopy = j + i * localSizeYConvolution;
        int indexPIC = indexOfConvolutionStartInPIC + j + i * localSizeYPIC;
        
        FieldType convolvedField; 

        convolvedField = 0.25 * tmpField[indexForCopy]
                       + 0.125 * (tmpField[indexForCopy + 1]
                                + tmpField[indexForCopy + localSizeYConvolution]
                                + tmpField[indexForCopy - 1]
                                + tmpField[indexForCopy - localSizeYConvolution])
                       + 0.0625 * (tmpField[indexForCopy + localSizeYConvolution + 1]
                                 + tmpField[indexForCopy + localSizeYConvolution - 1]
                                 + tmpField[indexForCopy - localSizeYConvolution - 1]
                                 + tmpField[indexForCopy - localSizeYConvolution + 1]);
        
        field[indexPIC] = convolvedField;

        if (j == 1) {
            field[indexPIC - 1] = field[indexPIC];
        }
        if (j == localSizeYConvolution - 2) {
            field[indexPIC + 1] = field[indexPIC];
        }
    }
}


void InterfaceNoiseRemover2D::convolve_magneticField(
    thrust::device_vector<MagneticField>& B
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((localSizeXConvolution + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (localSizeYConvolution + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyFieldsPIC_kernel<MagneticField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(tmpB.data()), 
        indexOfConvolutionStartInPIC, 
        localSizeXConvolution, localSizeYConvolution, 
        mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY
    );
    cudaDeviceSynchronize();
    
    convolveFields_kernel<MagneticField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpB.data()), 
        thrust::raw_pointer_cast(B.data()), 
        indexOfConvolutionStartInPIC, 
        localSizeXConvolution, localSizeYConvolution, 
        mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY
    );
    cudaDeviceSynchronize();

}


void InterfaceNoiseRemover2D::convolve_electricField(
    thrust::device_vector<ElectricField>& E
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((localSizeXConvolution + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (localSizeYConvolution + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyFieldsPIC_kernel<ElectricField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(E.data()), 
        thrust::raw_pointer_cast(tmpE.data()), 
        indexOfConvolutionStartInPIC, 
        localSizeXConvolution, localSizeYConvolution, 
        mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY
    );
    cudaDeviceSynchronize();

    convolveFields_kernel<ElectricField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpE.data()), 
        thrust::raw_pointer_cast(E.data()), 
        indexOfConvolutionStartInPIC, 
        localSizeXConvolution, localSizeYConvolution, 
        mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY
    );
    cudaDeviceSynchronize();

}


void InterfaceNoiseRemover2D::convolve_currentField(
    thrust::device_vector<CurrentField>& current
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((localSizeXConvolution + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (localSizeYConvolution + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyFieldsPIC_kernel<CurrentField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(current.data()), 
        thrust::raw_pointer_cast(tmpCurrent.data()), 
        indexOfConvolutionStartInPIC, 
        localSizeXConvolution, localSizeYConvolution, 
        mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY
    );
    cudaDeviceSynchronize();
    
    convolveFields_kernel<CurrentField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpCurrent.data()), 
        thrust::raw_pointer_cast(current.data()), 
        indexOfConvolutionStartInPIC, 
        localSizeXConvolution, localSizeYConvolution, 
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
    dim3 blocksPerGrid((localSizeXConvolution + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (localSizeYConvolution + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyFieldsPIC_kernel<ZerothMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(zerothMomentOfOneSpecies.data()), 
        thrust::raw_pointer_cast(tmpZerothMoment.data()), 
        indexOfConvolutionStartInPIC, 
        localSizeXConvolution, localSizeYConvolution, 
        mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY
    );
    cudaDeviceSynchronize();

    convolveFields_kernel<ZerothMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpZerothMoment.data()), 
        thrust::raw_pointer_cast(zerothMomentOfOneSpecies.data()), 
        indexOfConvolutionStartInPIC, 
        localSizeXConvolution, localSizeYConvolution, 
        mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY
    );
    cudaDeviceSynchronize();

    copyFieldsPIC_kernel<FirstMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(firstMomentOfOneSpecies.data()), 
        thrust::raw_pointer_cast(tmpFirstMoment.data()), 
        indexOfConvolutionStartInPIC, 
        localSizeXConvolution, localSizeYConvolution, 
        mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY
    );
    cudaDeviceSynchronize();

    convolveFields_kernel<FirstMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpFirstMoment.data()), 
        thrust::raw_pointer_cast(firstMomentOfOneSpecies.data()), 
        indexOfConvolutionStartInPIC, 
        localSizeXConvolution, localSizeYConvolution, 
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
    int indexOfConvolutionStartInMHD, 
    int localSizeXConvolution, int localSizeYConvolution, 
    int localSizeXMHD, int localSizeYMHD
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localSizeXConvolution && j < localSizeYConvolution) {
        int indexForCopy = j + i * localSizeYConvolution;
        int indexMHD = indexOfConvolutionStartInMHD + j + i * localSizeYMHD;

        tmpU[indexForCopy] = U[indexMHD];
    }
}


__global__ void convolveU_kernel(
    const ConservationParameter* tmpU, 
    ConservationParameter* U, 
    int indexOfConvolutionStartInMHD, 
    int localSizeXConvolution, int localSizeYConvolution, 
    int localSizeXMHD, int localSizeYMHD
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < localSizeXConvolution - 1 && 0 < j && j < localSizeYConvolution - 1) {
        int indexForCopy = j + i * localSizeYConvolution;
        int indexMHD = indexOfConvolutionStartInMHD + j + i * localSizeYMHD;
        
        ConservationParameter convolvedU;

        convolvedU = 0.25 * tmpU[indexForCopy]
                   + 0.125 * (tmpU[indexForCopy + 1]
                            + tmpU[indexForCopy + localSizeYConvolution]
                            + tmpU[indexForCopy - 1]
                            + tmpU[indexForCopy - localSizeYConvolution])
                   + 0.0625 * (tmpU[indexForCopy + localSizeYConvolution + 1]
                             + tmpU[indexForCopy + localSizeYConvolution - 1]
                             + tmpU[indexForCopy - localSizeYConvolution - 1]
                             + tmpU[indexForCopy - localSizeYConvolution + 1]);
        
        U[indexMHD] = convolvedU;

        if (j == 1) {
            U[indexMHD - 1] = U[indexMHD];
        }
        if (j == localSizeYConvolution - 2) {
            U[indexMHD + 1] = U[indexMHD];
        }
    }
}


void InterfaceNoiseRemover2D::convolveU(
    thrust::device_vector<ConservationParameter>& U
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((localSizeXConvolution + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (localSizeYConvolution + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyU_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(tmpU.data()),
        indexOfConvolutionStartInMHD, 
        localSizeXConvolution, localSizeYConvolution, 
        mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY
    );
    cudaDeviceSynchronize();

    convolveU_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpU.data()), 
        thrust::raw_pointer_cast(U.data()), 
        indexOfConvolutionStartInMHD, 
        localSizeXConvolution, localSizeYConvolution, 
        mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY
    );
    cudaDeviceSynchronize();
}

