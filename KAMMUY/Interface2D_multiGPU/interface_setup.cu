#include "interface.hpp"


__global__ void calculateSubU_kernel(
    const ConservationParameter* UPast, 
    const ConservationParameter* UNext, 
    ConservationParameter* USub, 
    double mixingRatio, 
    const int localSizeXMHD
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localSizeXMHD && j < IdealMHD2DConst::device_ny) {
        int index = j + i * IdealMHD2DConst::device_ny;

        USub[index] = mixingRatio * UPast[index] + (1.0 - mixingRatio) * UNext[index];
    }
}

thrust::device_vector<ConservationParameter>& Interface2D::calculateAndGetSubU(
    const thrust::device_vector<ConservationParameter>& UPast, 
    const thrust::device_vector<ConservationParameter>& UNext, 
    double mixingRatio
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfoMHD.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (IdealMHD2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    calculateSubU_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(UPast.data()), 
        thrust::raw_pointer_cast(UNext.data()), 
        thrust::raw_pointer_cast(USub.data()), 
        mixingRatio, 
        mPIInfoMHD.localSizeX
    );
    cudaDeviceSynchronize();

    return USub;
}


void Interface2D::resetTimeAveragedPICParameters()
{
    thrust::fill(
        B_timeAve.begin(), 
        B_timeAve.end(), 
        MagneticField()
    );

    thrust::fill(
        zerothMomentIon_timeAve.begin(), 
        zerothMomentIon_timeAve.end(), 
        ZerothMoment()
    );
    thrust::fill(
        zerothMomentElectron_timeAve.begin(), 
        zerothMomentElectron_timeAve.end(), 
        ZerothMoment()
    );

    thrust::fill(
        firstMomentIon_timeAve.begin(), 
        firstMomentIon_timeAve.end(), 
        FirstMoment()
    );
    thrust::fill(
        firstMomentElectron_timeAve.begin(), 
        firstMomentElectron_timeAve.end(), 
        FirstMoment()
    );

    thrust::fill(
        secondMomentIon_timeAve.begin(), 
        secondMomentIon_timeAve.end(), 
        SecondMoment()
    );
    thrust::fill(
        secondMomentElectron_timeAve.begin(), 
        secondMomentElectron_timeAve.end(), 
        SecondMoment()
    );
}


void Interface2D::sumUpTimeAveragedPICParameters(
    const thrust::device_vector<MagneticField>& B, 
    const thrust::device_vector<ZerothMoment>& zerothMomentIon, 
    const thrust::device_vector<ZerothMoment>& zerothMomentElectron, 
    const thrust::device_vector<FirstMoment>& firstMomentIon, 
    const thrust::device_vector<FirstMoment>& firstMomentElectron, 
    const thrust::device_vector<SecondMoment>& secondMomentIon, 
    const thrust::device_vector<SecondMoment>& secondMomentElectron
)
{
    thrust::transform(
        B_timeAve.begin(), B_timeAve.end(), 
        B.begin(), 
        B_timeAve.begin(), 
        thrust::plus<MagneticField>()
    );

    thrust::transform(
        zerothMomentIon_timeAve.begin(), zerothMomentIon_timeAve.end(), 
        zerothMomentIon.begin(), 
        zerothMomentIon_timeAve.begin(), 
        thrust::plus<ZerothMoment>()
    );
    thrust::transform(
        zerothMomentElectron_timeAve.begin(), zerothMomentElectron_timeAve.end(), 
        zerothMomentElectron.begin(), 
        zerothMomentElectron_timeAve.begin(), 
        thrust::plus<ZerothMoment>()
    );
    thrust::transform(
        firstMomentIon_timeAve.begin(), firstMomentIon_timeAve.end(), 
        firstMomentIon.begin(), 
        firstMomentIon_timeAve.begin(), 
        thrust::plus<FirstMoment>()
    );
    thrust::transform(
        firstMomentElectron_timeAve.begin(), firstMomentElectron_timeAve.end(), 
        firstMomentElectron.begin(), 
        firstMomentElectron_timeAve.begin(), 
        thrust::plus<FirstMoment>()
    );
    thrust::transform(
        secondMomentIon_timeAve.begin(), secondMomentIon_timeAve.end(), 
        secondMomentIon.begin(), 
        secondMomentIon_timeAve.begin(), 
        thrust::plus<SecondMoment>()
    );
    thrust::transform(
        secondMomentElectron_timeAve.begin(), secondMomentElectron_timeAve.end(), 
        secondMomentElectron.begin(), 
        secondMomentElectron_timeAve.begin(), 
        thrust::plus<SecondMoment>()
    );
}


template <typename FieldType>
__device__ FieldType getAveragedFieldForPICtoMHD(
    const FieldType* field, 
    int bufferPIC, 
    int i, int j
)
{
    FieldType convolvedField; 

    int validCount = 0; 

    for (int windowX = -Interface2DConst::device_gridSizeRatio / 2; windowX <= Interface2DConst::device_gridSizeRatio / 2; windowX++) {
        for (int windowY = -Interface2DConst::device_gridSizeRatio / 2; windowY <= Interface2DConst::device_gridSizeRatio / 2; windowY++) {
            int localI = i * Interface2DConst::device_gridSizeRatio + windowX; 
            int localJ = j * Interface2DConst::device_gridSizeRatio + windowY;

            if (0 <= localI && localI < PIC2DConst::device_nx && 0 <= localJ && localJ < PIC2DConst::device_ny) {
                int localIndex = localJ + (localI + bufferPIC) * PIC2DConst::device_ny; 
                convolvedField += field[localIndex];
                validCount++; 
            }
        }
    }
    convolvedField = convolvedField / validCount; 

    return convolvedField;
}


__global__ void averagingParametersForPICtoMHD_kernel(
    const MagneticField* B_timeAve, 
    const ZerothMoment* zerothMomentIon_timeAve, 
    const ZerothMoment* zerothMomentElectron_timeAve, 
    const FirstMoment* firstMomentIon_timeAve, 
    const FirstMoment* firstMomentElectron_timeAve, 
    const SecondMoment* secondMomentIon_timeAve, 
    const SecondMoment* secondMomentElectron_timeAve, 
    MagneticField* B_PICtoMHD, 
    ZerothMoment* zerothMomentIon_PICtoMHD, 
    ZerothMoment* zerothMomentElectron_PICtoMHD, 
    FirstMoment* firstMomentIon_PICtoMHD, 
    FirstMoment* firstMomentElectron_PICtoMHD, 
    SecondMoment* secondMomentIon_PICtoMHD, 
    SecondMoment* secondMomentElectron_PICtoMHD, 
    const int localNxMHD, const int bufferPIC, const int bufferMHD
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localNxMHD && j < PIC2DConst::device_ny / Interface2DConst::device_gridSizeRatio) {
        MagneticField averagedB; 
        ZerothMoment averagedZerothMomentIon, averagedZerothMomentElectron; 
        FirstMoment averagedFirstMomentIon, averagedFirstMomentElectron; 
        SecondMoment averagedSecondMomentIon, averagedSecondMomentElectron; 
        
        averagedB                    = getAveragedFieldForPICtoMHD(B_timeAve, bufferPIC, i, j);
        averagedZerothMomentIon      = getAveragedFieldForPICtoMHD(zerothMomentIon_timeAve, bufferPIC, i, j);
        averagedZerothMomentElectron = getAveragedFieldForPICtoMHD(zerothMomentElectron_timeAve, bufferPIC, i, j);
        averagedFirstMomentIon       = getAveragedFieldForPICtoMHD(firstMomentIon_timeAve, bufferPIC, i, j);
        averagedFirstMomentElectron  = getAveragedFieldForPICtoMHD(firstMomentElectron_timeAve, bufferPIC, i, j);
        averagedSecondMomentIon      = getAveragedFieldForPICtoMHD(secondMomentIon_timeAve, bufferPIC, i, j);
        averagedSecondMomentElectron = getAveragedFieldForPICtoMHD(secondMomentElectron_timeAve, bufferPIC, i, j);

        int indexPICtoMHD = j + i * PIC2DConst::device_ny / Interface2DConst::device_gridSizeRatio;

        B_PICtoMHD[indexPICtoMHD]                    = averagedB; 
        zerothMomentIon_PICtoMHD[indexPICtoMHD]      = averagedZerothMomentIon; 
        zerothMomentElectron_PICtoMHD[indexPICtoMHD] = averagedZerothMomentElectron; 
        firstMomentIon_PICtoMHD[indexPICtoMHD]       = averagedFirstMomentIon; 
        firstMomentElectron_PICtoMHD[indexPICtoMHD]  = averagedFirstMomentElectron; 
        secondMomentIon_PICtoMHD[indexPICtoMHD]      = averagedSecondMomentIon; 
        secondMomentElectron_PICtoMHD[indexPICtoMHD] = averagedSecondMomentElectron; 
    }
}


__global__ void calculateTimeAveragedPICParameters_kernel(
    MagneticField* B_timeAve, 
    ZerothMoment* zerothMomentIon_timeAve, 
    ZerothMoment* zerothMomentElectron_timeAve, 
    FirstMoment* firstMomentIon_timeAve, 
    FirstMoment* firstMomentElectron_timeAve, 
    SecondMoment* secondMomentIon_timeAve, 
    SecondMoment* secondMomentElectron_timeAve, 
    int count, 
    int localSizeXPIC
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localSizeXPIC && j < PIC2DConst::device_ny) {
        int index = j + i * PIC2DConst::device_ny;

        B_timeAve[index]                    = B_timeAve[index] / count; 
        zerothMomentIon_timeAve[index]      = zerothMomentIon_timeAve[index] / count; 
        zerothMomentElectron_timeAve[index] = zerothMomentElectron_timeAve[index] / count; 
        firstMomentIon_timeAve[index]       = firstMomentIon_timeAve[index] / count; 
        firstMomentElectron_timeAve[index]  = firstMomentElectron_timeAve[index] / count; 
        secondMomentIon_timeAve[index]      = secondMomentIon_timeAve[index] / count; 
        secondMomentElectron_timeAve[index] = secondMomentElectron_timeAve[index] / count; 
    }
}


void Interface2D::calculateTimeAveragedPICParameters(int count)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfoPIC.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (PIC2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    calculateTimeAveragedPICParameters_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(B_timeAve.data()), 
        thrust::raw_pointer_cast(zerothMomentIon_timeAve.data()), 
        thrust::raw_pointer_cast(zerothMomentElectron_timeAve.data()), 
        thrust::raw_pointer_cast(firstMomentIon_timeAve.data()), 
        thrust::raw_pointer_cast(firstMomentElectron_timeAve.data()), 
        thrust::raw_pointer_cast(secondMomentIon_timeAve.data()), 
        thrust::raw_pointer_cast(secondMomentElectron_timeAve.data()), 
        count, 
        mPIInfoPIC.localSizeX
    );
    cudaDeviceSynchronize();

}


void Interface2D::setParametersForPICtoMHD()
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfoMHD.localNx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (PIC2DConst::ny / Interface2DConst::gridSizeRatio + threadsPerBlock.y - 1) / threadsPerBlock.y);

    averagingParametersForPICtoMHD_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(B_timeAve.data()), 
        thrust::raw_pointer_cast(zerothMomentIon_timeAve.data()), 
        thrust::raw_pointer_cast(zerothMomentElectron_timeAve.data()), 
        thrust::raw_pointer_cast(firstMomentIon_timeAve.data()), 
        thrust::raw_pointer_cast(firstMomentElectron_timeAve.data()), 
        thrust::raw_pointer_cast(secondMomentIon_timeAve.data()), 
        thrust::raw_pointer_cast(secondMomentElectron_timeAve.data()), 
        thrust::raw_pointer_cast(B_PICtoMHD.data()), 
        thrust::raw_pointer_cast(zerothMomentIon_PICtoMHD.data()), 
        thrust::raw_pointer_cast(zerothMomentElectron_PICtoMHD.data()), 
        thrust::raw_pointer_cast(firstMomentIon_PICtoMHD.data()), 
        thrust::raw_pointer_cast(firstMomentElectron_PICtoMHD.data()), 
        thrust::raw_pointer_cast(secondMomentIon_PICtoMHD.data()), 
        thrust::raw_pointer_cast(secondMomentElectron_PICtoMHD.data()), 
        mPIInfoMHD.localNx, mPIInfoPIC.buffer, mPIInfoMHD.buffer
    );
    cudaDeviceSynchronize();
}


void Interface2D::calculateUHalf(
    const thrust::device_vector<ConservationParameter>& UPast, 
    const thrust::device_vector<ConservationParameter>& UNext 
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfoMHD.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (IdealMHD2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    calculateSubU_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(UPast.data()), 
        thrust::raw_pointer_cast(UNext.data()), 
        thrust::raw_pointer_cast(UHalf.data()), 
        0.5, 
        mPIInfoMHD.localSizeX
    );
    cudaDeviceSynchronize();
}


thrust::device_vector<ConservationParameter>& Interface2D::getUHalfRef()
{
    return UHalf;
}


