#include "interface.hpp"


__global__ void calculateSubU_kernel(
    const ConservationParameter* UPast, 
    const ConservationParameter* UNext, 
    ConservationParameter* USub, 
    double mixingRatio
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < IdealMHD2DConst::device_nx && j < IdealMHD2DConst::device_ny) {
        unsigned long long index = j + i * IdealMHD2DConst::device_ny;

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
        mixingRatio
    );
    cudaDeviceSynchronize();

    return USub;
}



template <typename FieldType>
__device__ FieldType getAveragedFieldForPICtoMHD(
    const FieldType* field, 
    unsigned long long i, unsigned long long j
)
{
    FieldType convolvedField; 

    const int R = Interface2DConst::device_gridSizeRatio / 2;
    const double sigma = Interface2DConst::device_gridSizeRatio / 2.0;
    const double twoSigma2 = 2.0 * sigma * sigma;

    double weightSum = 0.0;
    for (int dx = -R; dx <= R; dx++) {
        for (int dy = -R; dy <= R; dy++) {
            int localI = i * Interface2DConst::device_gridSizeRatio + dx;
            int localJ = j * Interface2DConst::device_gridSizeRatio + dy;

            if (0 <= localI && localI < PIC2DConst::device_nx &&
                0 <= localJ && localJ < PIC2DConst::device_ny)
            {
                double distance2 = double(dx * dx + dy * dy);
                double weight = 1.0;

                unsigned long long index = localJ + localI * PIC2DConst::device_ny;
                convolvedField += field[index] * weight;
                weightSum += weight;
            }
        }
    }
    convolvedField = convolvedField / weightSum;

    return convolvedField;
}


__global__ void averagingParametersForPICtoMHD_kernel(
    const MagneticField* B, 
    const ZerothMoment* zerothMomentIon, 
    const ZerothMoment* zerothMomentElectron, 
    const FirstMoment* firstMomentIon, 
    const FirstMoment* firstMomentElectron, 
    const SecondMoment* secondMomentIon, 
    const SecondMoment* secondMomentElectron, 
    MagneticField* B_PICtoMHD, 
    ZerothMoment* zerothMomentIon_PICtoMHD, 
    ZerothMoment* zerothMomentElectron_PICtoMHD, 
    FirstMoment* firstMomentIon_PICtoMHD, 
    FirstMoment* firstMomentElectron_PICtoMHD, 
    SecondMoment* secondMomentIon_PICtoMHD, 
    SecondMoment* secondMomentElectron_PICtoMHD, 
    const int localNxMHD, const int bufferMHD
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localNxMHD && j < PIC2DConst::device_ny / Interface2DConst::device_gridSizeRatio) {
        MagneticField averagedB; 
        ZerothMoment averagedZerothMomentIon, averagedZerothMomentElectron; 
        FirstMoment averagedFirstMomentIon, averagedFirstMomentElectron; 
        SecondMoment averagedSecondMomentIon, averagedSecondMomentElectron; 
        
        averagedB                    = getAveragedFieldForPICtoMHD(B, i, j);
        averagedZerothMomentIon      = getAveragedFieldForPICtoMHD(zerothMomentIon, i, j);
        averagedZerothMomentElectron = getAveragedFieldForPICtoMHD(zerothMomentElectron, i, j);
        averagedFirstMomentIon       = getAveragedFieldForPICtoMHD(firstMomentIon, i, j);
        averagedFirstMomentElectron  = getAveragedFieldForPICtoMHD(firstMomentElectron, i, j);
        averagedSecondMomentIon      = getAveragedFieldForPICtoMHD(secondMomentIon, i, j);
        averagedSecondMomentElectron = getAveragedFieldForPICtoMHD(secondMomentElectron, i, j);

        unsigned long long indexPICtoMHD = j + i * PIC2DConst::device_ny / Interface2DConst::device_gridSizeRatio;

        B_PICtoMHD[indexPICtoMHD]                    = averagedB; 
        zerothMomentIon_PICtoMHD[indexPICtoMHD]      = averagedZerothMomentIon; 
        zerothMomentElectron_PICtoMHD[indexPICtoMHD] = averagedZerothMomentElectron; 
        firstMomentIon_PICtoMHD[indexPICtoMHD]       = averagedFirstMomentIon; 
        firstMomentElectron_PICtoMHD[indexPICtoMHD]  = averagedFirstMomentElectron; 
        secondMomentIon_PICtoMHD[indexPICtoMHD]      = averagedSecondMomentIon; 
        secondMomentElectron_PICtoMHD[indexPICtoMHD] = averagedSecondMomentElectron; 
    }
}


void Interface2D::setParametersForPICtoMHD(
    const thrust::device_vector<MagneticField>& B, 
    const thrust::device_vector<ZerothMoment>& zerothMomentIon, 
    const thrust::device_vector<ZerothMoment>& zerothMomentElectron, 
    const thrust::device_vector<FirstMoment>& firstMomentIon, 
    const thrust::device_vector<FirstMoment>& firstMomentElectron, 
    const thrust::device_vector<SecondMoment>& secondMomentIon, 
    const thrust::device_vector<SecondMoment>& secondMomentElectron
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((PIC2DConst::nx / Interface2DConst::gridSizeRatio + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (PIC2DConst::ny / Interface2DConst::gridSizeRatio + threadsPerBlock.y - 1) / threadsPerBlock.y);

    averagingParametersForPICtoMHD_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(zerothMomentIon.data()), 
        thrust::raw_pointer_cast(zerothMomentElectron.data()), 
        thrust::raw_pointer_cast(firstMomentIon.data()), 
        thrust::raw_pointer_cast(firstMomentElectron.data()), 
        thrust::raw_pointer_cast(secondMomentIon.data()), 
        thrust::raw_pointer_cast(secondMomentElectron.data()), 
        thrust::raw_pointer_cast(B_PICtoMHD.data()), 
        thrust::raw_pointer_cast(zerothMomentIon_PICtoMHD.data()), 
        thrust::raw_pointer_cast(zerothMomentElectron_PICtoMHD.data()), 
        thrust::raw_pointer_cast(firstMomentIon_PICtoMHD.data()), 
        thrust::raw_pointer_cast(firstMomentElectron_PICtoMHD.data()), 
        thrust::raw_pointer_cast(secondMomentIon_PICtoMHD.data()), 
        thrust::raw_pointer_cast(secondMomentElectron_PICtoMHD.data()), 
        mPIInfoMHD.localNx, mPIInfoMHD.buffer
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
        0.5
    );
    cudaDeviceSynchronize();
}


thrust::device_vector<ConservationParameter>& Interface2D::getUHalfRef()
{
    return UHalf;
}


