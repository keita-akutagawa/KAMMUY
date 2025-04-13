#include "remove_noise.hpp"


InterfaceNoiseRemover2D::InterfaceNoiseRemover2D(
    IdealMHD2DMPI::MPIInfo& mPIInfoMHD, 
    PIC2DMPI::MPIInfo& mPIInfoPIC
)
    : mPIInfoMHD(mPIInfoMHD), 
      mPIInfoPIC(mPIInfoPIC), 

      //tmpB(localSizeXConvolutionPIC * localSizeYConvolutionPIC), 
      //tmpE(localSizeXConvolutionPIC * localSizeYConvolutionPIC), 
      //tmpCurrent(localSizeXConvolutionPIC * localSizeYConvolutionPIC), 
      //tmpZerothMoment(localSizeXConvolutionPIC * localSizeYConvolutionPIC), 
      //tmpFirstMoment(localSizeXConvolutionPIC * localSizeYConvolutionPIC), 
      tmpU(mPIInfoMHD.localSizeX * IdealMHD2DConst::ny)
{

    cudaMalloc(&device_mPIInfoMHD, sizeof(IdealMHD2DMPI::MPIInfo));
    cudaMemcpy(device_mPIInfoMHD, &mPIInfoMHD, sizeof(IdealMHD2DMPI::MPIInfo), cudaMemcpyHostToDevice);
    cudaMalloc(&device_mPIInfoPIC, sizeof(PIC2DMPI::MPIInfo));
    cudaMemcpy(device_mPIInfoPIC, &mPIInfoPIC, sizeof(PIC2DMPI::MPIInfo), cudaMemcpyHostToDevice);
    
}


/*
template <typename FieldType>
__global__ void copyFieldsPIC_kernel(
    const FieldType* field, 
    FieldType* tmpField, 
    int indexOfConvolutionStartInPIC, 
    int localSizeXConvolutionPIC, int localSizeYConvolutionPIC, 
    int localSizeXPIC
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localSizeXConvolutionPIC && j < localSizeYConvolutionPIC) {
        int indexForCopy = j + i * localSizeYConvolutionPIC;
        int indexPIC = indexOfConvolutionStartInPIC + j + i * PIC2DConst::device_ny;

        tmpField[indexForCopy] = field[indexPIC];
    }
}


template <typename FieldType>
__global__ void convolveFields_kernel(
    const FieldType* tmpField, 
    FieldType* field, 
    int indexOfConvolutionStartInPIC, 
    int localSizeXConvolutionPIC, int localSizeYConvolutionPIC, 
    int localSizeXPIC
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (2 <= i && i <= localSizeXConvolutionPIC - 3 && 2 <= j && j <= localSizeYConvolutionPIC - 3) {
        int indexForCopy = j + i * localSizeYConvolutionPIC;
        int indexPIC = indexOfConvolutionStartInPIC + j + i * PIC2DConst::device_ny;
        
        FieldType convolvedField; 

        for (int windowX = -1; windowX <= 1; windowX++) {
            for (int windowY = -1; windowY <= 1; windowY++) {
                int localIndex; 
                localIndex = indexForCopy + windowY + windowX * localSizeYConvolutionPIC; 
                convolvedField = convolvedField + tmpField[localIndex];
            }
        }
        convolvedField = 1.0 / 9.0 * convolvedField; 
        
        field[indexPIC] = convolvedField;
    }
}


void InterfaceNoiseRemover2D::convolve_magneticField(
    thrust::device_vector<MagneticField>& B
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((localSizeXConvolutionPIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (localSizeYConvolutionPIC + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyFieldsPIC_kernel<MagneticField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(tmpB.data()), 
        indexOfConvolutionStartInPIC, 
        localSizeXConvolutionPIC, localSizeYConvolutionPIC, 
        mPIInfoPIC.localSizeX
    );
    cudaDeviceSynchronize();
    
    convolveFields_kernel<MagneticField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpB.data()), 
        thrust::raw_pointer_cast(B.data()), 
        indexOfConvolutionStartInPIC, 
        localSizeXConvolutionPIC, localSizeYConvolutionPIC, 
        mPIInfoPIC.localSizeX
    );
    cudaDeviceSynchronize();

}


void InterfaceNoiseRemover2D::convolve_electricField(
    thrust::device_vector<ElectricField>& E
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((localSizeXConvolutionPIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (localSizeYConvolutionPIC + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyFieldsPIC_kernel<ElectricField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(E.data()), 
        thrust::raw_pointer_cast(tmpE.data()), 
        indexOfConvolutionStartInPIC, 
        localSizeXConvolutionPIC, localSizeYConvolutionPIC, 
        mPIInfoPIC.localSizeX
    );
    cudaDeviceSynchronize();

    convolveFields_kernel<ElectricField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpE.data()), 
        thrust::raw_pointer_cast(E.data()), 
        indexOfConvolutionStartInPIC, 
        localSizeXConvolutionPIC, localSizeYConvolutionPIC, 
        mPIInfoPIC.localSizeX
    );
    cudaDeviceSynchronize();

}


void InterfaceNoiseRemover2D::convolve_currentField(
    thrust::device_vector<CurrentField>& current
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((localSizeXConvolutionPIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (localSizeYConvolutionPIC + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyFieldsPIC_kernel<CurrentField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(current.data()), 
        thrust::raw_pointer_cast(tmpCurrent.data()), 
        indexOfConvolutionStartInPIC, 
        localSizeXConvolutionPIC, localSizeYConvolutionPIC, 
        mPIInfoPIC.localSizeX
    );
    cudaDeviceSynchronize();
    
    convolveFields_kernel<CurrentField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpCurrent.data()), 
        thrust::raw_pointer_cast(current.data()), 
        indexOfConvolutionStartInPIC, 
        localSizeXConvolutionPIC, localSizeYConvolutionPIC, 
        mPIInfoPIC.localSizeX
    );
    cudaDeviceSynchronize();

}


void InterfaceNoiseRemover2D::convolveMomentsOfOneSpecies(
    thrust::device_vector<ZerothMoment>& zerothMomentOfOneSpecies, 
    thrust::device_vector<FirstMoment>& firstMomentOfOneSpecies
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((localSizeXConvolutionPIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (localSizeYConvolutionPIC + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyFieldsPIC_kernel<ZerothMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(zerothMomentOfOneSpecies.data()), 
        thrust::raw_pointer_cast(tmpZerothMoment.data()), 
        indexOfConvolutionStartInPIC, 
        localSizeXConvolutionPIC, localSizeYConvolutionPIC, 
        mPIInfoPIC.localSizeX
    );
    cudaDeviceSynchronize();

    convolveFields_kernel<ZerothMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpZerothMoment.data()), 
        thrust::raw_pointer_cast(zerothMomentOfOneSpecies.data()), 
        indexOfConvolutionStartInPIC, 
        localSizeXConvolutionPIC, localSizeYConvolutionPIC, 
        mPIInfoPIC.localSizeX
    );
    cudaDeviceSynchronize();

    copyFieldsPIC_kernel<FirstMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(firstMomentOfOneSpecies.data()), 
        thrust::raw_pointer_cast(tmpFirstMoment.data()), 
        indexOfConvolutionStartInPIC, 
        localSizeXConvolutionPIC, localSizeYConvolutionPIC, 
        mPIInfoPIC.localSizeX
    );
    cudaDeviceSynchronize();

    convolveFields_kernel<FirstMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpFirstMoment.data()), 
        thrust::raw_pointer_cast(firstMomentOfOneSpecies.data()), 
        indexOfConvolutionStartInPIC, 
        localSizeXConvolutionPIC, localSizeYConvolutionPIC, 
        mPIInfoPIC.localSizeX
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
*/


/*
__global__ void copyU_kernel(
    const ConservationParameter* U, 
    ConservationParameter* tmpU, 
    int indexOfConvolutionStartInMHD, 
    int localSizeXConvolutionMHD, int localSizeYConvolutionMHD, 
    int localSizeXMHD
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localSizeXConvolutionMHD && j < localSizeYConvolutionMHD) {
        int indexForCopy = j + i * localSizeYConvolutionMHD;
        int indexMHD = indexOfConvolutionStartInMHD + j + i * IdealMHD2DConst::device_ny;

        tmpU[indexForCopy] = U[indexMHD];
    }
}


__global__ void convolveU_kernel(
    const ConservationParameter* tmpU, 
    ConservationParameter* U, 
    int indexOfConvolutionStartInMHD, 
    int localSizeXConvolutionMHD, int localSizeYConvolutionMHD, 
    int localSizeXMHD
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (2 <= i && i <= localSizeXConvolutionMHD - 3 && 2 <= j && j <= localSizeYConvolutionMHD - 3) {
        int indexForCopy = j + i * localSizeYConvolutionMHD;
        int indexMHD = indexOfConvolutionStartInMHD + j + i * IdealMHD2DConst::device_ny;

        ConservationParameter convolvedU; 

        for (int windowX = -1; windowX <= 1; windowX++) {
            for (int windowY = -1; windowY <= 1; windowY++) {
                int localIndex; 
                localIndex = indexForCopy + windowY + windowX * localSizeYConvolutionMHD; 
                convolvedU = convolvedU + tmpU[localIndex];
            }
        }
        convolvedU = 1.0 / 9.0 * convolvedU; 

        U[indexMHD] = convolvedU;
    }
}
*/

__global__ void convolveU_kernel(
    const ConservationParameter* tmpU, 
    ConservationParameter* U, 
    int localSizeXMHD
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < localSizeXMHD - 1 && 0 < j && j < IdealMHD2DConst::device_ny - 1) {
        int indexMHD = j + i * IdealMHD2DConst::device_ny;

        ConservationParameter convolvedU; 

        for (int windowX = -1; windowX <= 1; windowX++) {
            for (int windowY = -1; windowY <= 1; windowY++) {
                int localIndex = indexMHD + windowY + windowX * IdealMHD2DConst::device_ny; 

                //if (windowX == 0 && windowY == 0) {
                //    convolvedU += 4.0 * tmpU[localIndex];
                //} else if (windowX == 0 || windowY == 0) {
                //    convolvedU += 2.0 * tmpU[localIndex]; 
                //} else {
                //    convolvedU += 1.0 * tmpU[localIndex];
                //}
                convolvedU += tmpU[localIndex];
            }
        }
        //convolvedU = convolvedU / 16.0; 
        convolvedU = convolvedU / 9.0; 

        U[indexMHD] = convolvedU;
    }
}


void InterfaceNoiseRemover2D::convolveU(
    thrust::device_vector<ConservationParameter>& U
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfoMHD.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (IdealMHD2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    //copyU_kernel<<<blocksPerGrid, threadsPerBlock>>>(
    //    thrust::raw_pointer_cast(U.data()), 
    //    thrust::raw_pointer_cast(tmpU.data()),
    //    indexOfConvolutionStartInMHD, 
    //    localSizeXConvolutionMHD, localSizeYConvolutionMHD, 
    //    mPIInfoMHD.localSizeX
    //);
    //cudaDeviceSynchronize();

    thrust::copy(U.begin(), U.end(), tmpU.begin());

    convolveU_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpU.data()), 
        thrust::raw_pointer_cast(U.data()), 
        mPIInfoMHD.localSizeX
    );
    cudaDeviceSynchronize();
}

