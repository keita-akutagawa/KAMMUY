#include "interface.hpp"


__global__ void setUHalf_kernel(
    const ConservationParameter* UPast, 
    const ConservationParameter* UNext, 
    ConservationParameter* UHalf, 
    int localSizeXMHD
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localSizeXMHD && j < IdealMHD2DConst::device_ny) {
        int index = j + i * IdealMHD2DConst::device_ny;

        UHalf[index] = 0.5 * (UPast[index] + UNext[index]);
    }
}


__global__ void sendPICtoMHD_kernel(
    const double* interlockingFunctionY, 
    const ZerothMoment* zerothMomentIon, 
    const ZerothMoment* ZerothMomentElectron, 
    const FirstMoment* firstMomentIon, 
    const FirstMoment* firstMomentElectron, 
    const MagneticField* B, 
    ConservationParameter* U, 
    int indexOfInterfaceStartInMHD, 
    int localSizeXPIC
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < localSizeXPIC - 1 && 0 < j && j < PIC2DConst::device_ny - 1) {
        int indexPIC = j + i * PIC2DConst::device_ny;
        int indexMHD = indexOfInterfaceStartInMHD + j + i * IdealMHD2DConst::device_ny;
        double rhoMHD, uMHD, vMHD, wMHD, bXMHD, bYMHD, bZMHD, eMHD, pMHD;
        double rhoPIC, uPIC, vPIC, wPIC, bXPIC, bYPIC, bZPIC;
        double niMHD, neMHD, tiMHD, teMHD;
        double mIon = PIC2DConst::device_mIon, mElectron = PIC2DConst::device_mElectron;

        //MHDのグリッドにPICを合わせる
        rhoMHD      = U[indexMHD].rho;
        uMHD        = U[indexMHD].rhoU / rhoMHD;
        vMHD        = U[indexMHD].rhoV / rhoMHD;
        wMHD        = U[indexMHD].rhoW / rhoMHD;
        bXMHD       = U[indexMHD].bX;
        bYMHD       = U[indexMHD].bY;
        bZMHD       = U[indexMHD].bZ;
        eMHD        = U[indexMHD].e;
        pMHD        = (IdealMHD2DConst::device_gamma - 1.0)
                    * (eMHD - 0.5 * rhoMHD * (uMHD * uMHD + vMHD * vMHD + wMHD * wMHD)
                    - 0.5 * (bXMHD * bXMHD + bYMHD * bYMHD + bZMHD * bZMHD));

        //tiMHD, teMHDはMHDの情報のままにするために、この計算が必要。
        niMHD = rhoMHD / (mIon + mElectron);
        neMHD = niMHD;
        tiMHD = pMHD / 2.0 / niMHD;
        teMHD = pMHD / 2.0 / neMHD;
        
        rhoPIC =  mIon * zerothMomentIon[indexPIC].n + mElectron * ZerothMomentElectron[indexPIC].n;
        uPIC   = (mIon * firstMomentIon[indexPIC].x  + mElectron * firstMomentElectron[indexPIC].x) / rhoPIC;
        vPIC   = (mIon * firstMomentIon[indexPIC].y  + mElectron * firstMomentElectron[indexPIC].y) / rhoPIC;
        wPIC   = (mIon * firstMomentIon[indexPIC].z  + mElectron * firstMomentElectron[indexPIC].z) / rhoPIC;
        bXPIC  = 0.5f * (B[indexPIC].bX + B[indexPIC - PIC2DConst::device_ny].bX); 
        bYPIC  = 0.5f * (B[indexPIC].bY + B[indexPIC - 1].bY); 
        bZPIC  = B[indexPIC].bZ; 

        rhoMHD = interlockingFunctionY[indexPIC] * rhoMHD + (1.0 - interlockingFunctionY[indexPIC]) * rhoPIC;
        uMHD   = interlockingFunctionY[indexPIC] * uMHD   + (1.0 - interlockingFunctionY[indexPIC]) * uPIC;
        vMHD   = interlockingFunctionY[indexPIC] * vMHD   + (1.0 - interlockingFunctionY[indexPIC]) * vPIC;
        wMHD   = interlockingFunctionY[indexPIC] * wMHD   + (1.0 - interlockingFunctionY[indexPIC]) * wPIC;
        bXMHD  = interlockingFunctionY[indexPIC] * bXMHD  + (1.0 - interlockingFunctionY[indexPIC]) * bXPIC;
        bYMHD  = interlockingFunctionY[indexPIC] * bYMHD  + (1.0 - interlockingFunctionY[indexPIC]) * bYPIC;
        bZMHD  = interlockingFunctionY[indexPIC] * bZMHD  + (1.0 - interlockingFunctionY[indexPIC]) * bZPIC;

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
             + 0.5 * (bXMHD * bXMHD + bYMHD * bYMHD + bZMHD * bZMHD);
        U[indexMHD].e = eMHD;
    }
}


//MHDのグリッドを整数格子点上に再配置してから使うこと
void Interface2D::sendPICtoMHD(
    thrust::device_vector<ConservationParameter>& U
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfoPIC.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (PIC2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendPICtoMHD_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()), 
        thrust::raw_pointer_cast(zerothMomentIon_timeAve.data()), 
        thrust::raw_pointer_cast(zerothMomentElectron_timeAve.data()), 
        thrust::raw_pointer_cast(firstMomentIon_timeAve.data()), 
        thrust::raw_pointer_cast(firstMomentElectron_timeAve.data()), 
        thrust::raw_pointer_cast(B_timeAve.data()), 
        thrust::raw_pointer_cast(U.data()), 
        indexOfInterfaceStartInMHD, 
        mPIInfoPIC.localSizeX
    );
}


