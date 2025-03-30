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


void Interface2D::setMoments(
    const thrust::device_vector<Particle>& particlesIon, 
    const thrust::device_vector<Particle>& particlesElectron
)
{
    momentCalculater.calculateZerothMomentOfOneSpecies(
        tmp_zerothMomentIon, particlesIon, mPIInfoPIC.existNumIonPerProcs
    );
    momentCalculater.calculateZerothMomentOfOneSpecies(
        tmp_zerothMomentElectron, particlesElectron, mPIInfoPIC.existNumElectronPerProcs
    );

    momentCalculater.calculateFirstMomentOfOneSpecies(
        tmp_firstMomentIon, particlesIon, mPIInfoPIC.existNumIonPerProcs
    );
    momentCalculater.calculateFirstMomentOfOneSpecies(
        tmp_firstMomentElectron, particlesElectron, mPIInfoPIC.existNumElectronPerProcs
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


void Interface2D::setParametersForPICtoMHD(
    const thrust::device_vector<MagneticField>& B, 
    const thrust::device_vector<Particle>& particlesIon, 
    const thrust::device_vector<Particle>& particlesElectron
)
{
    thrust::copy(B.begin(), B.end(), tmp_B.begin());

    setMoments(particlesIon, particlesElectron); 
    boundaryPIC.periodicBoundaryZerothMoment_x(tmp_zerothMomentIon); 
    boundaryPIC.freeBoundaryZerothMoment_y(tmp_zerothMomentIon); 
    boundaryPIC.periodicBoundaryZerothMoment_x(tmp_zerothMomentElectron); 
    boundaryPIC.freeBoundaryZerothMoment_y(tmp_zerothMomentElectron); 
    boundaryPIC.periodicBoundaryFirstMoment_x(tmp_firstMomentIon); 
    boundaryPIC.freeBoundaryFirstMoment_y(tmp_firstMomentIon); 
    boundaryPIC.periodicBoundaryFirstMoment_x(tmp_firstMomentElectron); 
    boundaryPIC.freeBoundaryFirstMoment_y(tmp_firstMomentElectron); 


    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfoMHD.localNx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (PIC2DConst::ny / Interface2DConst::gridSizeRatio + threadsPerBlock.y - 1) / threadsPerBlock.y);

    averagingParametersForPICtoMHD_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmp_B.data()), 
        thrust::raw_pointer_cast(tmp_zerothMomentIon.data()), 
        thrust::raw_pointer_cast(tmp_zerothMomentElectron.data()), 
        thrust::raw_pointer_cast(tmp_firstMomentIon.data()), 
        thrust::raw_pointer_cast(tmp_firstMomentElectron.data()), 
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


