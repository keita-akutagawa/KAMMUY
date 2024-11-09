#include "interface.hpp"


__global__ void setUHalf_kernel(
    const ConservationParameter* UPast, 
    const ConservationParameter* UNext, 
    ConservationParameter* UHalf
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < IdealMHD2DConst::device_nx && j < IdealMHD2DConst::device_ny) {
        int index = j + i * IdealMHD2DConst::device_ny;

        UHalf[index].rho  = 0.5 * (UPast[index].rho  + UNext[index].rho );
        UHalf[index].rhoU = 0.5 * (UPast[index].rhoU + UNext[index].rhoU);
        UHalf[index].rhoV = 0.5 * (UPast[index].rhoV + UNext[index].rhoV);
        UHalf[index].rhoW = 0.5 * (UPast[index].rhoW + UNext[index].rhoW);
        UHalf[index].bX   = 0.5 * (UPast[index].bX   + UNext[index].bX  );
        UHalf[index].bY   = 0.5 * (UPast[index].bY   + UNext[index].bY  );
        UHalf[index].bZ   = 0.5 * (UPast[index].bZ   + UNext[index].bZ  );
        UHalf[index].e    = 0.5 * (UPast[index].e    + UNext[index].e   );
    }
}


__global__ void sendPICtoMHD_kernel(
    const double* interlockingFunctionY, 
    const double* interlockingFunctionYHalf, 
    const ZerothMoment* zerothMomentIon, 
    const ZerothMoment* ZerothMomentElectron, 
    const FirstMoment* firstMomentIon, 
    const FirstMoment* firstMomentElectron, 
    const MagneticField* B, 
    ConservationParameter* U, 
    int indexOfInterfaceStartInMHD, 
    int indexOfInterfaceStartInPIC, 
    int interfaceLength
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < PIC2DConst::device_nx - 1 && 0 < j && j < interfaceLength - 1) {
        int indexPIC = indexOfInterfaceStartInPIC + j + i * PIC2DConst::device_ny;
        int indexMHD = indexOfInterfaceStartInMHD + j + i * IdealMHD2DConst::device_ny;
        double rhoMHD, uMHD, vMHD, wMHD, bXMHD, bYMHD, bZMHD, eMHD, pMHD;
        double bXCenterMHD, bYCenterMHD;
        double rhoPIC, uPIC, vPIC, wPIC, bXPIC, bYPIC, bZPIC;
        double bXCenterPIC, bYCenterPIC;
        double niMHD, neMHD, tiMHD, teMHD;
        int ny = IdealMHD2DConst::device_ny;
        int ny = PIC2DConst::device_ny;
        double mIon = PIC2DConst::device_mIon, mElectron = PIC2DConst::device_mElectron;

        //MHDのグリッドにPICを合わせる
        rhoMHD      = max(U[indexMHD].rho, IdealMHD2DConst::device_EPS);
        uMHD        = U[indexMHD].rhoU / (rhoMHD + IdealMHD2DConst::device_EPS);
        vMHD        = U[indexMHD].rhoV / (rhoMHD + IdealMHD2DConst::device_EPS);
        wMHD        = U[indexMHD].rhoW / (rhoMHD + IdealMHD2DConst::device_EPS);
        bXMHD       = U[indexMHD].bX;
        bYMHD       = U[indexMHD].bY;
        bZMHD       = U[indexMHD].bZ;
        eMHD        = U[indexMHD].e;
        bXCenterMHD = 0.5 * (U[indexMHD].bX + U[indexMHD - ny].bX);
        bYCenterMHD = 0.5 * (U[indexMHD].bY + U[indexMHD - 1].bY);
        pMHD        = (IdealMHD2DConst::device_gamma - 1.0)
                    * (eMHD - 0.5 * rhoMHD * (uMHD * uMHD + vMHD * vMHD + wMHD * wMHD)
                    - 0.5 * (bXCenterMHD * bXCenterMHD + bYCenterMHD * bYCenterMHD + bZMHD * bZMHD));
        pMHD        = max(pMHD, IdealMHD2DConst::device_EPS);

        //tiMHD, teMHDはMHDの情報のままにするために、この計算が必要。
        niMHD = rhoMHD / (mIon + mElectron);
        neMHD = niMHD;
        tiMHD = pMHD / 2.0 / niMHD;
        teMHD = pMHD / 2.0 / neMHD;
        
        rhoPIC      =  max(mIon * zerothMomentIon[indexPIC].n + mElectron * ZerothMomentElectron[indexPIC].n, PIC2DConst::device_EPS);
        uPIC        = (mIon * firstMomentIon[indexPIC].x  + mElectron * firstMomentElectron[indexPIC].x) / (rhoPIC + PIC2DConst::device_EPS);
        vPIC        = (mIon * firstMomentIon[indexPIC].y  + mElectron * firstMomentElectron[indexPIC].y) / (rhoPIC + PIC2DConst::device_EPS);
        wPIC        = (mIon * firstMomentIon[indexPIC].z  + mElectron * firstMomentElectron[indexPIC].z) / (rhoPIC + PIC2DConst::device_EPS);
        bXPIC       = 0.25 * (B[indexPIC].bX + B[indexPIC + ny].bX + B[indexPIC - 1].bX + B[indexPIC - 1 + ny].bX);
        bYPIC       = 0.25 * (B[indexPIC].bY + B[indexPIC + 1].bY + B[indexPIC - ny].bY + B[indexPIC + 1 - ny].bY);
        bZPIC       = 0.25 * (B[indexPIC].bZ + B[indexPIC - ny].bZ + B[indexPIC - 1].bZ + B[indexPIC - 1 - ny].bZ);
        bXCenterPIC = 0.5 * (B[indexPIC].bX + B[indexPIC - ny].bX);
        bYCenterPIC = 0.5 * (B[indexPIC].bY + B[indexPIC - 1].bY);

        rhoMHD       = interlockingFunctionY[j]     * rhoMHD       + (1.0 - interlockingFunctionY[j])     * rhoPIC;
        uMHD         = interlockingFunctionY[j]     * uMHD         + (1.0 - interlockingFunctionY[j])     * uPIC;
        vMHD         = interlockingFunctionY[j]     * vMHD         + (1.0 - interlockingFunctionY[j])     * vPIC;
        wMHD         = interlockingFunctionY[j]     * wMHD         + (1.0 - interlockingFunctionY[j])     * wPIC;
        bXMHD        = interlockingFunctionY[j]     * bXMHD        + (1.0 - interlockingFunctionY[j])     * bXPIC;
        bYMHD        = interlockingFunctionYHalf[j] * bYMHD        + (1.0 - interlockingFunctionYHalf[j]) * bYPIC;
        bZMHD        = interlockingFunctionY[j]     * bZMHD        + (1.0 - interlockingFunctionY[j])     * bZPIC;
        bXCenterMHD  = interlockingFunctionY[j]     * bXCenterMHD  + (1.0 - interlockingFunctionY[j])     * bXCenterPIC;
        bYCenterMHD  = interlockingFunctionY[j]     * bYCenterMHD  + (1.0 - interlockingFunctionY[j])     * bYCenterPIC;

        niMHD = rhoMHD / (mIon + mElectron);
        neMHD = niMHD;
        pMHD  = niMHD * tiMHD + neMHD * teMHD;


        U[indexMHD].rho  = rhoMHD;
        U[indexMHD].rhoU = rhoMHD * uMHD;
        U[indexMHD].rhoV = rhoMHD * vMHD;
        U[indexMHD].rhoW = rhoMHD * wMHD;
        U[indexMHD].bX   = bXMHD;
        U[indexMHD].bY   = bYMHD;
        U[indexMHD].bZ   = bZMHD;
        eMHD = pMHD / (IdealMHD2DConst::device_gamma - 1.0)
             + 0.5 * rhoMHD * (uMHD * uMHD + vMHD * vMHD + wMHD * wMHD)
             + 0.5 * (bXCenterMHD * bXCenterMHD + bYCenterMHD * bYCenterMHD + bZMHD * bZMHD);
        U[indexMHD].e = eMHD;
    }
}


//MHDのグリッドを整数格子点上に再配置してから使うこと
void Interface2D::sendPICtoMHD(
    const thrust::device_vector<ConservationParameter>& UPast, 
    const thrust::device_vector<ConservationParameter>& UNext
)
{
    dim3 threadsPerBlockForSetUHalf(16, 16);
    dim3 blocksPerGridForSetUHalf((IdealMHD2DConst::nx + threadsPerBlockForSetUHalf.x - 1) / threadsPerBlockForSetUHalf.x,
                                  (IdealMHD2DConst::ny + threadsPerBlockForSetUHalf.y - 1) / threadsPerBlockForSetUHalf.y);

    setUHalf_kernel<<<blocksPerGridForSetUHalf, threadsPerBlockForSetUHalf>>>(
        thrust::raw_pointer_cast(UPast.data()), 
        thrust::raw_pointer_cast(UNext.data()), 
        thrust::raw_pointer_cast(UHalf.data())
    );

    cudaDeviceSynchronize();


    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((PIC2DConst::nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (interfaceLength + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendPICtoMHD_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()), 
        thrust::raw_pointer_cast(interlockingFunctionYHalf.data()), 
        thrust::raw_pointer_cast(zerothMomentIon_timeAve.data()), 
        thrust::raw_pointer_cast(zerothMomentElectron_timeAve.data()), 
        thrust::raw_pointer_cast(firstMomentIon_timeAve.data()), 
        thrust::raw_pointer_cast(firstMomentElectron_timeAve.data()), 
        thrust::raw_pointer_cast(B_timeAve.data()), 
        thrust::raw_pointer_cast(UHalf.data()), 
        indexOfInterfaceStartInMHD, 
        indexOfInterfaceStartInPIC, 
        interfaceLength
    );

    cudaDeviceSynchronize();

}


thrust::device_vector<ConservationParameter>& Interface2D::getUHalfRef()
{
    return UHalf;
}

//////////////////////////////


__global__ void calculateSubU_kernel(
    const ConservationParameter* UPast, 
    const ConservationParameter* UNext, 
    ConservationParameter* USub, 
    double mixingRatio
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < IdealMHD2DConst::device_nx && j < IdealMHD2DConst::device_ny) {
        int index = j + i * IdealMHD2DConst::device_ny;

        USub[index].rho  = mixingRatio * UPast[index].rho  + (1.0 - mixingRatio) * UNext[index].rho;
        USub[index].rhoU = mixingRatio * UPast[index].rhoU + (1.0 - mixingRatio) * UNext[index].rhoU;
        USub[index].rhoV = mixingRatio * UPast[index].rhoV + (1.0 - mixingRatio) * UNext[index].rhoV;
        USub[index].rhoW = mixingRatio * UPast[index].rhoW + (1.0 - mixingRatio) * UNext[index].rhoW;
        USub[index].bX   = mixingRatio * UPast[index].bX   + (1.0 - mixingRatio) * UNext[index].bX;
        USub[index].bY   = mixingRatio * UPast[index].bY   + (1.0 - mixingRatio) * UNext[index].bY;
        USub[index].bZ   = mixingRatio * UPast[index].bZ   + (1.0 - mixingRatio) * UNext[index].bZ;
        USub[index].e    = mixingRatio * UPast[index].e    + (1.0 - mixingRatio) * UNext[index].e;
    }
}

thrust::device_vector<ConservationParameter>& Interface2D::calculateAndGetSubU(
    const thrust::device_vector<ConservationParameter>& UPast, 
    const thrust::device_vector<ConservationParameter>& UNext, 
    double mixingRatio
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((IdealMHD2DConst::nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
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
    int substeps 
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < PIC2DConst::device_nx && j < PIC2DConst::device_ny) {
        int index = j + i * PIC2DConst::device_ny;

        B_timeAve[index].bX                   /= static_cast<double>(substeps);
        B_timeAve[index].bY                   /= static_cast<double>(substeps);
        B_timeAve[index].bZ                   /= static_cast<double>(substeps);
        zerothMomentIon_timeAve[index].n      /= static_cast<double>(substeps);
        zerothMomentElectron_timeAve[index].n /= static_cast<double>(substeps);
        firstMomentIon_timeAve[index].x       /= static_cast<double>(substeps);
        firstMomentIon_timeAve[index].y       /= static_cast<double>(substeps);
        firstMomentIon_timeAve[index].z       /= static_cast<double>(substeps);
        firstMomentElectron_timeAve[index].x  /= static_cast<double>(substeps);
        firstMomentElectron_timeAve[index].y  /= static_cast<double>(substeps);
        firstMomentElectron_timeAve[index].z  /= static_cast<double>(substeps);
    }
}

void Interface2D::calculateTimeAveParameters(int substeps)
{
    interfaceNoiseRemover2D_Lower.convolve_lower_magneticField(B_timeAve);
    interfaceNoiseRemover2D_Upper.convolve_upper_magneticField(B_timeAve);
    interfaceNoiseRemover2D_Lower.convolveMoments_lower(
        zerothMomentIon_timeAve, zerothMomentElectron_timeAve, 
        firstMomentIon_timeAve, firstMomentElectron_timeAve
    );
    interfaceNoiseRemover2D_Upper.convolveMoments_upper(
        zerothMomentIon_timeAve, zerothMomentElectron_timeAve, 
        firstMomentIon_timeAve, firstMomentElectron_timeAve
    );

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((PIC2DConst::nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (PIC2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    calculateTimeAveParameters_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(B_timeAve.data()), 
        thrust::raw_pointer_cast(zerothMomentIon_timeAve.data()), 
        thrust::raw_pointer_cast(zerothMomentElectron_timeAve.data()), 
        thrust::raw_pointer_cast(firstMomentIon_timeAve.data()), 
        thrust::raw_pointer_cast(firstMomentElectron_timeAve.data()), 
        substeps
    );

    cudaDeviceSynchronize();
}

