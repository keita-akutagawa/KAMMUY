#include "remove_noise.hpp"


InterfaceNoiseRemover2D::InterfaceNoiseRemover2D(
    IdealMHD2DMPI::MPIInfo& mPIInfoMHD, 
    PIC2DMPI::MPIInfo& mPIInfoPIC, 
    int indexOfConvolutionStartInMHD, 
    int indexOfConvolutionStartInPIC, 
    int localSizeXConvolutionMHD, int localSizeYConvolutionMHD, 
    int localSizeXConvolutionPIC, int localSizeYConvolutionPIC
)
    : mPIInfoMHD(mPIInfoMHD), 
      mPIInfoPIC(mPIInfoPIC), 

      indexOfConvolutionStartInMHD(indexOfConvolutionStartInMHD), 
      indexOfConvolutionStartInPIC(indexOfConvolutionStartInPIC), 
      localSizeXConvolutionMHD(localSizeXConvolutionMHD), 
      localSizeYConvolutionMHD(localSizeYConvolutionMHD), 
      localSizeXConvolutionPIC(localSizeXConvolutionPIC), 
      localSizeYConvolutionPIC(localSizeYConvolutionPIC), 

      tmpB(localSizeXConvolutionPIC * localSizeYConvolutionPIC), 
      tmpE(localSizeXConvolutionPIC * localSizeYConvolutionPIC), 
      tmpCurrent(localSizeXConvolutionPIC * localSizeYConvolutionPIC), 
      tmpZerothMoment(localSizeXConvolutionPIC * localSizeYConvolutionPIC), 
      tmpFirstMoment(localSizeXConvolutionPIC * localSizeYConvolutionPIC), 
      tmpU(localSizeXConvolutionMHD * localSizeYConvolutionMHD)
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
    int localSizeXConvolutionPIC, int localSizeYConvolutionPIC, 
    int localSizeXPIC, int localSizeYPIC
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localSizeXConvolutionPIC && j < localSizeYConvolutionPIC) {
        int indexForCopy = j + i * localSizeYConvolutionPIC;
        int indexPIC = indexOfConvolutionStartInPIC + j + i * localSizeYPIC;

        tmpField[indexForCopy] = field[indexPIC];
    }
}


template <typename FieldType>
__global__ void convolveFields_kernel(
    const FieldType* tmpField, 
    FieldType* field, 
    int indexOfConvolutionStartInPIC, 
    int localSizeXConvolutionPIC, int localSizeYConvolutionPIC, 
    int localSizeXPIC, int localSizeYPIC
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (2 <= i && i <= localSizeXConvolutionPIC - 3 && 2 <= j && j <= localSizeYConvolutionPIC - 3) {
        int indexForCopy = j + i * localSizeYConvolutionPIC;
        int indexPIC = indexOfConvolutionStartInPIC + j + i * localSizeYPIC;
        
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
        mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY
    );
    cudaDeviceSynchronize();
    
    convolveFields_kernel<MagneticField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpB.data()), 
        thrust::raw_pointer_cast(B.data()), 
        indexOfConvolutionStartInPIC, 
        localSizeXConvolutionPIC, localSizeYConvolutionPIC, 
        mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY
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
        mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY
    );
    cudaDeviceSynchronize();

    convolveFields_kernel<ElectricField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpE.data()), 
        thrust::raw_pointer_cast(E.data()), 
        indexOfConvolutionStartInPIC, 
        localSizeXConvolutionPIC, localSizeYConvolutionPIC, 
        mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY
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
        mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY
    );
    cudaDeviceSynchronize();
    
    convolveFields_kernel<CurrentField><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpCurrent.data()), 
        thrust::raw_pointer_cast(current.data()), 
        indexOfConvolutionStartInPIC, 
        localSizeXConvolutionPIC, localSizeYConvolutionPIC, 
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
    dim3 blocksPerGrid((localSizeXConvolutionPIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (localSizeYConvolutionPIC + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyFieldsPIC_kernel<ZerothMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(zerothMomentOfOneSpecies.data()), 
        thrust::raw_pointer_cast(tmpZerothMoment.data()), 
        indexOfConvolutionStartInPIC, 
        localSizeXConvolutionPIC, localSizeYConvolutionPIC, 
        mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY
    );
    cudaDeviceSynchronize();

    convolveFields_kernel<ZerothMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpZerothMoment.data()), 
        thrust::raw_pointer_cast(zerothMomentOfOneSpecies.data()), 
        indexOfConvolutionStartInPIC, 
        localSizeXConvolutionPIC, localSizeYConvolutionPIC, 
        mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY
    );
    cudaDeviceSynchronize();

    copyFieldsPIC_kernel<FirstMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(firstMomentOfOneSpecies.data()), 
        thrust::raw_pointer_cast(tmpFirstMoment.data()), 
        indexOfConvolutionStartInPIC, 
        localSizeXConvolutionPIC, localSizeYConvolutionPIC, 
        mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY
    );
    cudaDeviceSynchronize();

    convolveFields_kernel<FirstMoment><<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpFirstMoment.data()), 
        thrust::raw_pointer_cast(firstMomentOfOneSpecies.data()), 
        indexOfConvolutionStartInPIC, 
        localSizeXConvolutionPIC, localSizeYConvolutionPIC, 
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
    int localSizeXConvolutionMHD, int localSizeYConvolutionMHD, 
    int localSizeXMHD, int localSizeYMHD
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localSizeXConvolutionMHD && j < localSizeYConvolutionMHD) {
        int indexForCopy = j + i * localSizeYConvolutionMHD;
        int indexMHD = indexOfConvolutionStartInMHD + j + i * localSizeYMHD;

        tmpU[indexForCopy] = U[indexMHD];
    }
}


__global__ void convolveU_kernel(
    const ConservationParameter* tmpU, 
    ConservationParameter* U, 
    int indexOfConvolutionStartInMHD, 
    int localSizeXConvolutionMHD, int localSizeYConvolutionMHD, 
    int localSizeXMHD, int localSizeYMHD
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (2 <= i && i <= localSizeXConvolutionMHD - 3 && 2 <= j && j <= localSizeYConvolutionMHD - 3) {
        int indexForCopy = j + i * localSizeYConvolutionMHD;
        int indexMHD = indexOfConvolutionStartInMHD + j + i * localSizeYMHD;

        /*
        double window_rho[9], window_rhoU[9], window_rhoV[9], window_rhoW[9]; 
        double window_bX[9], window_bY[9], window_bZ[9], window_e[9], window_psi[9]; 
        int count = 0;

        for (int di = -1; di <= 1; ++di) {
            for (int dj = -1; dj <= 1; ++dj) {
                int neighborIndex = indexForCopy + dj + di * localSizeYConvolutionMHD;

                window_rho[count]  = tmpU[neighborIndex].rho;
                window_rhoU[count] = tmpU[neighborIndex].rhoU;
                window_rhoV[count] = tmpU[neighborIndex].rhoV;
                window_rhoW[count] = tmpU[neighborIndex].rhoW;
                window_bX[count]   = tmpU[neighborIndex].bX;
                window_bY[count]   = tmpU[neighborIndex].bY;
                window_bZ[count]   = tmpU[neighborIndex].bZ;
                window_e[count]    = tmpU[neighborIndex].e;
                window_psi[count]  = tmpU[neighborIndex].psi; 

                count += 1;
            }
        }

        for (int m = 0; m < 9; ++m) {
            for (int n = m + 1; n < 9; ++n) {
                if (window_rho[m] > window_rho[n]) {
                    double tmp;
                    tmp = window_rho[m];
                    window_rho[m] = window_rho[n];
                    window_rho[n] = tmp;
                }
                if (window_rhoU[m] > window_rhoU[n]) {
                    double tmp;
                    tmp = window_rhoU[m];
                    window_rhoU[m] = window_rhoU[n];
                    window_rhoU[n] = tmp;
                }
                if (window_rhoV[m] > window_rhoV[n]) {
                    double tmp;
                    tmp = window_rhoV[m];
                    window_rhoV[m] = window_rhoV[n];
                    window_rhoV[n] = tmp;
                }
                if (window_rhoW[m] > window_rhoW[n]) {
                    double tmp;
                    tmp = window_rhoW[m];
                    window_rhoW[m] = window_rhoW[n];
                    window_rhoW[n] = tmp;
                }
                if (window_bX[m] > window_bX[n]) {
                    double tmp;
                    tmp = window_bX[m];
                    window_bX[m] = window_bX[n];
                    window_bX[n] = tmp;
                }
                if (window_bY[m] > window_bY[n]) {
                    double tmp;
                    tmp = window_bY[m];
                    window_bY[m] = window_bY[n];
                    window_bY[n] = tmp;
                }
                if (window_bZ[m] > window_bZ[n]) {
                    double tmp;
                    tmp = window_bZ[m];
                    window_bZ[m] = window_bZ[n];
                    window_bZ[n] = tmp;
                }
                if (window_e[m] > window_e[n]) {
                    double tmp;
                    tmp = window_e[m];
                    window_e[m] = window_e[n];
                    window_e[n] = tmp;
                }
                if (window_psi[m] > window_psi[n]) {
                    double tmp;
                    tmp = window_psi[m];
                    window_psi[m] = window_psi[n];
                    window_psi[n] = tmp;
                }
            }
        }

        ConservationParameter convolvedU;

        convolvedU.rho  = window_rho[4]; 
        convolvedU.rhoU = window_rhoU[4]; 
        convolvedU.rhoV = window_rhoV[4]; 
        convolvedU.rhoW = window_rhoW[4]; 
        convolvedU.bX   = window_bX[4]; 
        convolvedU.bY   = window_bY[4]; 
        convolvedU.bZ   = window_bZ[4]; 
        convolvedU.e    = window_e[4]; 
        convolvedU.psi  = window_psi[4]; 
        */

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


void InterfaceNoiseRemover2D::convolveU(
    thrust::device_vector<ConservationParameter>& U
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((localSizeXConvolutionMHD + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (localSizeYConvolutionMHD + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyU_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(tmpU.data()),
        indexOfConvolutionStartInMHD, 
        localSizeXConvolutionMHD, localSizeYConvolutionMHD, 
        mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY
    );
    cudaDeviceSynchronize();

    convolveU_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpU.data()), 
        thrust::raw_pointer_cast(U.data()), 
        indexOfConvolutionStartInMHD, 
        localSizeXConvolutionMHD, localSizeYConvolutionMHD, 
        mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY
    );
    cudaDeviceSynchronize();
}

