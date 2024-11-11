#include "interface.hpp"


__global__ void setUHalf_kernel(
    const ConservationParameter* UPast, 
    const ConservationParameter* UNext, 
    ConservationParameter* UHalf, 
    int localSizeXMHD, int localSizeYMHD
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localSizeXMHD && j < localSizeYMHD) {
        int index = j + i * localSizeYMHD;

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
    int interfaceLength, 
    int localSizeXPIC, int localSizeYPIC, 
    int localSizeXMHD, int localSizeYMHD
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < localSizeXMHD - 1 && j < interfaceLength) {
        int indexPIC = indexOfInterfaceStartInPIC + j + i * localSizeYPIC;
        int indexMHD = indexOfInterfaceStartInMHD + j + i * localSizeYMHD;
        double rhoMHD, uMHD, vMHD, wMHD, bXMHD, bYMHD, bZMHD, eMHD, pMHD;
        double bXCenterMHD, bYCenterMHD;
        double rhoPIC, uPIC, vPIC, wPIC, bXPIC, bYPIC, bZPIC;
        double bXCenterPIC, bYCenterPIC;
        double niMHD, neMHD, tiMHD, teMHD;
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
        bXCenterMHD = 0.5 * (U[indexMHD].bX + U[indexMHD - localSizeYMHD].bX);
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
        bXPIC       = 0.25 * (B[indexPIC].bX + B[indexPIC + localSizeYPIC].bX + B[indexPIC - 1].bX + B[indexPIC - 1 + localSizeYPIC].bX);
        bYPIC       = 0.25 * (B[indexPIC].bY + B[indexPIC + 1].bY + B[indexPIC - localSizeYPIC].bY + B[indexPIC + 1 - localSizeYPIC].bY);
        bZPIC       = 0.25 * (B[indexPIC].bZ + B[indexPIC - localSizeYPIC].bZ + B[indexPIC - 1].bZ + B[indexPIC - 1 - localSizeYPIC].bZ);
        bXCenterPIC = 0.5 * (B[indexPIC].bX + B[indexPIC - localSizeYPIC].bX);
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
    dim3 blocksPerGridForSetUHalf((mPIInfoMHD.localSizeX + threadsPerBlockForSetUHalf.x - 1) / threadsPerBlockForSetUHalf.x,
                                  (mPIInfoMHD.localSizeY + threadsPerBlockForSetUHalf.y - 1) / threadsPerBlockForSetUHalf.y);

    setUHalf_kernel<<<blocksPerGridForSetUHalf, threadsPerBlockForSetUHalf>>>(
        thrust::raw_pointer_cast(UPast.data()), 
        thrust::raw_pointer_cast(UNext.data()), 
        thrust::raw_pointer_cast(UHalf.data()), 
        mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY
    );
    cudaDeviceSynchronize();


    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfoMHD.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
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
        interfaceLength, 
        mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY, 
        mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY
    );
    cudaDeviceSynchronize();

}


