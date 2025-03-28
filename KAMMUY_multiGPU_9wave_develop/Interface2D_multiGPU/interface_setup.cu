#include "interface.hpp"


__global__ void calculateSubU_kernel(
    const ConservationParameter* UPast, 
    const ConservationParameter* UNext, 
    ConservationParameter* USub, 
    double mixingRatio, 
    int localSizeXMHD, int localSizeYMHD
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localSizeXMHD && j < localSizeYMHD) {
        int index = j + i * localSizeYMHD;

        USub[index].rho  = mixingRatio * UPast[index].rho  + (1.0 - mixingRatio) * UNext[index].rho;
        USub[index].rhoU = mixingRatio * UPast[index].rhoU + (1.0 - mixingRatio) * UNext[index].rhoU;
        USub[index].rhoV = mixingRatio * UPast[index].rhoV + (1.0 - mixingRatio) * UNext[index].rhoV;
        USub[index].rhoW = mixingRatio * UPast[index].rhoW + (1.0 - mixingRatio) * UNext[index].rhoW;
        USub[index].bX   = mixingRatio * UPast[index].bX   + (1.0 - mixingRatio) * UNext[index].bX;
        USub[index].bY   = mixingRatio * UPast[index].bY   + (1.0 - mixingRatio) * UNext[index].bY;
        USub[index].bZ   = mixingRatio * UPast[index].bZ   + (1.0 - mixingRatio) * UNext[index].bZ;
        USub[index].e    = mixingRatio * UPast[index].e    + (1.0 - mixingRatio) * UNext[index].e;
        USub[index].psi  = mixingRatio * UPast[index].psi  + (1.0 - mixingRatio) * UNext[index].psi;
    }
}

thrust::device_vector<ConservationParameter>& Interface2D::calculateAndGetSubU(
    const thrust::device_vector<ConservationParameter>& UPast, 
    const thrust::device_vector<ConservationParameter>& UNext, 
    double mixingRatio
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((localSizeXMHD + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (localSizeYMHD + threadsPerBlock.y - 1) / threadsPerBlock.y);

    calculateSubU_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(UPast.data()), 
        thrust::raw_pointer_cast(UNext.data()), 
        thrust::raw_pointer_cast(USub.data()), 
        mixingRatio, 
        localSizeXMHD, localSizeYMHD
    );
    cudaDeviceSynchronize();

    return USub;
}


void Interface2D::resetTimeAveParameters()
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


void Interface2D::setMoments(
    const thrust::device_vector<Particle>& particlesIon, 
    const thrust::device_vector<Particle>& particlesElectron
)
{
    momentCalculater.calculateZerothMomentOfOneSpecies(
        zerothMomentIon, particlesIon, mPIInfoPIC.existNumIonPerProcs
    );
    momentCalculater.calculateZerothMomentOfOneSpecies(
        zerothMomentElectron, particlesElectron, mPIInfoPIC.existNumElectronPerProcs
    );

    momentCalculater.calculateFirstMomentOfOneSpecies(
        firstMomentIon, particlesIon, mPIInfoPIC.existNumIonPerProcs
    );
    momentCalculater.calculateFirstMomentOfOneSpecies(
        firstMomentElectron, particlesElectron, mPIInfoPIC.existNumElectronPerProcs
    );
}


void Interface2D::sumUpTimeAveParameters(
    const thrust::device_vector<MagneticField>& B, 
    const thrust::device_vector<Particle>& particlesIon, 
    const thrust::device_vector<Particle>& particlesElectron
)
{
    thrust::transform(
        B_timeAve.begin(), B_timeAve.end(), B.begin(), 
        B_timeAve.begin(), thrust::plus<MagneticField>()
    );

    setMoments(particlesIon, particlesElectron); 
    boundaryPIC.periodicBoundaryZerothMoment_x(zerothMomentIon); 
    boundaryPIC.freeBoundaryZerothMoment_y(zerothMomentIon); 
    boundaryPIC.periodicBoundaryZerothMoment_x(zerothMomentElectron); 
    boundaryPIC.freeBoundaryZerothMoment_y(zerothMomentElectron); 
    boundaryPIC.periodicBoundaryFirstMoment_x(firstMomentIon); 
    boundaryPIC.freeBoundaryFirstMoment_y(firstMomentIon); 
    boundaryPIC.periodicBoundaryFirstMoment_x(firstMomentElectron); 
    boundaryPIC.freeBoundaryFirstMoment_y(firstMomentElectron); 

    thrust::transform(
        zerothMomentIon_timeAve.begin(), zerothMomentIon_timeAve.end(), zerothMomentIon.begin(), 
        zerothMomentIon_timeAve.begin(), thrust::plus<ZerothMoment>()
    );
    thrust::transform(
        zerothMomentElectron_timeAve.begin(), zerothMomentElectron_timeAve.end(), zerothMomentElectron.begin(), 
        zerothMomentElectron_timeAve.begin(), thrust::plus<ZerothMoment>()
    );
    thrust::transform(
        firstMomentIon_timeAve.begin(), firstMomentIon_timeAve.end(), firstMomentIon.begin(), 
        firstMomentIon_timeAve.begin(), thrust::plus<FirstMoment>()
    );
    thrust::transform(
        firstMomentElectron_timeAve.begin(), firstMomentElectron_timeAve.end(), firstMomentElectron.begin(), 
        firstMomentElectron_timeAve.begin(), thrust::plus<FirstMoment>()
    );
}


__global__ void calculateTimeAveParameters_kernel(
    MagneticField* B_timeAve, 
    ZerothMoment* zerothMomentIon_timeAve, 
    ZerothMoment* zerothMomentElectron_timeAve, 
    FirstMoment* firstMomentIon_timeAve, 
    FirstMoment* firstMomentElectron_timeAve, 
    int substeps, 
    int localSizeXPIC, int localSizeYPIC
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localSizeXPIC && j < localSizeYPIC) {
        int index = j + i * localSizeYPIC;

        B_timeAve[index].bX                   /= static_cast<float>(substeps);
        B_timeAve[index].bY                   /= static_cast<float>(substeps);
        B_timeAve[index].bZ                   /= static_cast<float>(substeps);
        zerothMomentIon_timeAve[index].n      /= static_cast<float>(substeps);
        zerothMomentElectron_timeAve[index].n /= static_cast<float>(substeps);
        firstMomentIon_timeAve[index].x       /= static_cast<float>(substeps);
        firstMomentIon_timeAve[index].y       /= static_cast<float>(substeps);
        firstMomentIon_timeAve[index].z       /= static_cast<float>(substeps);
        firstMomentElectron_timeAve[index].x  /= static_cast<float>(substeps);
        firstMomentElectron_timeAve[index].y  /= static_cast<float>(substeps);
        firstMomentElectron_timeAve[index].z  /= static_cast<float>(substeps);
    }
}

void Interface2D::calculateTimeAveParameters(int substeps)
{

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((localSizeXPIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (localSizeYPIC + threadsPerBlock.y - 1) / threadsPerBlock.y);

    calculateTimeAveParameters_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(B_timeAve.data()), 
        thrust::raw_pointer_cast(zerothMomentIon_timeAve.data()), 
        thrust::raw_pointer_cast(zerothMomentElectron_timeAve.data()), 
        thrust::raw_pointer_cast(firstMomentIon_timeAve.data()), 
        thrust::raw_pointer_cast(firstMomentElectron_timeAve.data()), 
        substeps, 
        localSizeXPIC, localSizeYPIC
    );
    cudaDeviceSynchronize();

    boundaryPIC.periodicBoundaryB_x(B_timeAve);
    boundaryPIC.periodicBoundaryZerothMoment_x(zerothMomentIon_timeAve); 
    boundaryPIC.periodicBoundaryZerothMoment_x(zerothMomentElectron_timeAve); 
    boundaryPIC.periodicBoundaryFirstMoment_x(firstMomentIon_timeAve); 
    boundaryPIC.periodicBoundaryFirstMoment_x(firstMomentElectron_timeAve); 
    boundaryPIC.freeBoundaryB_y(B_timeAve);
    boundaryPIC.freeBoundaryZerothMoment_y(zerothMomentIon_timeAve); 
    boundaryPIC.freeBoundaryZerothMoment_y(zerothMomentElectron_timeAve); 
    boundaryPIC.freeBoundaryFirstMoment_y(firstMomentIon_timeAve); 
    boundaryPIC.freeBoundaryFirstMoment_y(firstMomentElectron_timeAve); 
}


thrust::device_vector<ConservationParameter>& Interface2D::getUHalfRef()
{
    return UHalf;
}


