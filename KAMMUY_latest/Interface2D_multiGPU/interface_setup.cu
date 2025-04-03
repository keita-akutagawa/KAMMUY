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


//void Interface2D::setMoments(
//    const thrust::device_vector<Particle>& particlesIon, 
//    const thrust::device_vector<Particle>& particlesElectron
//)
//{
//    momentCalculator.calculateZerothMomentOfOneSpecies(
//        tmp_zerothMomentIon, particlesIon, mPIInfoPIC.existNumIonPerProcs
//    );
//    momentCalculator.calculateZerothMomentOfOneSpecies(
//        tmp_zerothMomentElectron, particlesElectron, mPIInfoPIC.existNumElectronPerProcs
//    );
//
//    momentCalculator.calculateFirstMomentOfOneSpecies(
//        tmp_firstMomentIon, particlesIon, mPIInfoPIC.existNumIonPerProcs
//    );
//    momentCalculator.calculateFirstMomentOfOneSpecies(
//        tmp_firstMomentElectron, particlesElectron, mPIInfoPIC.existNumElectronPerProcs
//    );
//}


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
}


void Interface2D::sumUpTimeAveragedPICParameters(
    const thrust::device_vector<MagneticField>& B, 
    const thrust::device_vector<ZerothMoment>& zerothMomentIon, 
    const thrust::device_vector<ZerothMoment>& zerothMomentElectron, 
    const thrust::device_vector<FirstMoment>& firstMomentIon, 
    const thrust::device_vector<FirstMoment>& firstMomentElectron
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
}


__global__ void averagingParametersForPICtoMHD_kernel(
    const MagneticField* tmp_B, 
    const ZerothMoment* tmp_zerothMomentIon, 
    const ZerothMoment* tmp_zerothMomentElectron, 
    const FirstMoment* tmp_firstMomentIon, 
    const FirstMoment* tmp_firstMomentElectron, 
    MagneticField* B_PICtoMHD, 
    ZerothMoment* zerothMomentIon_PICtoMHD, 
    ZerothMoment* zerothMomentElectron_PICtoMHD, 
    FirstMoment* firstMomentIon_PICtoMHD, 
    FirstMoment* firstMomentElectron_PICtoMHD, 
    const int localNxMHD, const int bufferPIC, const int bufferMHD
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localNxMHD && j < PIC2DConst::device_ny / Interface2DConst::device_gridSizeRatio) {
        
        MagneticField averagedB; 
        ZerothMoment averagedZerothMomentIon, averagedZerothMomentElecteron; 
        FirstMoment averagedFirstMomentIon, averagedFirstMomentElectron; 
        for (int windowX = 0; windowX < Interface2DConst::device_gridSizeRatio; windowX++) {
            for (int windowY = 0; windowY < Interface2DConst::device_gridSizeRatio; windowY++) {
                int indexPIC = (j * Interface2DConst::device_gridSizeRatio + windowY)
                             + (i * Interface2DConst::device_gridSizeRatio + bufferPIC + windowX) * PIC2DConst::device_ny; 
                
                averagedB                     = averagedB + tmp_B[indexPIC]; 
                averagedZerothMomentIon       = averagedZerothMomentIon + tmp_zerothMomentIon[indexPIC]; 
                averagedZerothMomentElecteron = averagedZerothMomentElecteron + tmp_zerothMomentElectron[indexPIC]; 
                averagedFirstMomentIon        = averagedFirstMomentIon + tmp_firstMomentIon[indexPIC]; 
                averagedFirstMomentElectron   = averagedFirstMomentElectron + tmp_firstMomentElectron[indexPIC];
            }
        }
        averagedB                     = 1.0 / pow(Interface2DConst::device_gridSizeRatio, 2) * averagedB;
        averagedZerothMomentIon       = 1.0 / pow(Interface2DConst::device_gridSizeRatio, 2) * averagedZerothMomentIon;
        averagedZerothMomentElecteron = 1.0 / pow(Interface2DConst::device_gridSizeRatio, 2) * averagedZerothMomentElecteron;
        averagedFirstMomentIon        = 1.0 / pow(Interface2DConst::device_gridSizeRatio, 2) * averagedFirstMomentIon;
        averagedFirstMomentElectron   = 1.0 / pow(Interface2DConst::device_gridSizeRatio, 2) * averagedFirstMomentElectron;
        

        int indexPICtoMHD = j + i * PIC2DConst::device_ny / Interface2DConst::device_gridSizeRatio;

        B_PICtoMHD[indexPICtoMHD] = averagedB; 
        zerothMomentIon_PICtoMHD[indexPICtoMHD] = averagedZerothMomentIon; 
        zerothMomentElectron_PICtoMHD[indexPICtoMHD] = averagedZerothMomentElecteron; 
        firstMomentIon_PICtoMHD[indexPICtoMHD] = averagedFirstMomentIon; 
        firstMomentElectron_PICtoMHD[indexPICtoMHD] = averagedFirstMomentElectron; 
    }
}


__global__ void calculateSubPICParameters_kernel(
    MagneticField* B, 
    ZerothMoment* zerothMomentIon, 
    ZerothMoment* zerothMomentElectron, 
    FirstMoment* firstMomentIon, 
    FirstMoment* firstMomentElectron, 
    int count, 
    int localSizeXPIC
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localSizeXPIC && j < PIC2DConst::device_ny) {
        int index = j + i * PIC2DConst::device_ny;

        B[index]                    = 1.0 / count * B[index]; 
        zerothMomentIon[index]      = 1.0 / count * zerothMomentIon[index]; 
        zerothMomentElectron[index] = 1.0 / count * zerothMomentElectron[index]; 
        firstMomentIon[index]       = 1.0 / count * firstMomentIon[index]; 
        firstMomentElectron[index]  = 1.0 / count * firstMomentElectron[index]; 
    }
}


void Interface2D::calculateTimeAveragedPICParameters(int count)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfoPIC.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (PIC2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    calculateSubPICParameters_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(B_timeAve.data()), 
        thrust::raw_pointer_cast(zerothMomentIon_timeAve.data()), 
        thrust::raw_pointer_cast(zerothMomentElectron_timeAve.data()), 
        thrust::raw_pointer_cast(firstMomentIon_timeAve.data()), 
        thrust::raw_pointer_cast(firstMomentElectron_timeAve.data()), 
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
        thrust::raw_pointer_cast(B_PICtoMHD.data()), 
        thrust::raw_pointer_cast(zerothMomentIon_PICtoMHD.data()), 
        thrust::raw_pointer_cast(zerothMomentElectron_PICtoMHD.data()), 
        thrust::raw_pointer_cast(firstMomentIon_PICtoMHD.data()), 
        thrust::raw_pointer_cast(firstMomentElectron_PICtoMHD.data()), 
        mPIInfoMHD.localNx, mPIInfoPIC.buffer, mPIInfoMHD.buffer
    );
    cudaDeviceSynchronize();
}


thrust::device_vector<ConservationParameter>& Interface2D::getUHalfRef()
{
    return UHalf;
}


