#include "interface.hpp"


__global__ void sendPICtoMHD_kernel(
    const double* interlockingFunctionY, 
    const MagneticField* B_PICtoMHD, 
    const ZerothMoment* zerothMomentIon_PICtoMHD, 
    const ZerothMoment* ZerothMomentElectron_PICtoMHD, 
    const FirstMoment* firstMomentIon_PICtoMHD, 
    const FirstMoment* firstMomentElectron_PICtoMHD, 
    ConservationParameter* U, 
    const int indexOfInterfaceStartInMHD, 
    const int localNxMHD, const int bufferPIC, const int bufferMHD
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localNxMHD && j < PIC2DConst::device_ny / Interface2DConst::device_gridSizeRatio) {
        int indexMHD = indexOfInterfaceStartInMHD + j  + (i + bufferMHD) * IdealMHD2DConst::device_ny;
        int indexPICtoMHD = j + i * PIC2DConst::device_ny / Interface2DConst::device_gridSizeRatio;
        double rhoMHD, uMHD, vMHD, wMHD, bXMHD, bYMHD, bZMHD, eMHD, pMHD;
        double rhoPICtoMHD, uPICtoMHD, vPICtoMHD, wPICtoMHD, bXPICtoMHD, bYPICtoMHD, bZPICtoMHD;
        double niMHD, neMHD, tiMHD, teMHD;
        double mIon = PIC2DConst::device_mIon, mElectron = PIC2DConst::device_mElectron;

        rhoMHD = U[indexMHD].rho;
        uMHD   = U[indexMHD].rhoU / rhoMHD;
        vMHD   = U[indexMHD].rhoV / rhoMHD;
        wMHD   = U[indexMHD].rhoW / rhoMHD;
        bXMHD  = U[indexMHD].bX;
        bYMHD  = U[indexMHD].bY;
        bZMHD  = U[indexMHD].bZ;
        eMHD   = U[indexMHD].e;
        pMHD   = (IdealMHD2DConst::device_gamma - 1.0)
               * (eMHD - 0.5 * rhoMHD * (uMHD * uMHD + vMHD * vMHD + wMHD * wMHD)
               - 0.5 * (bXMHD * bXMHD + bYMHD * bYMHD + bZMHD * bZMHD));

        //tiMHD, teMHDはMHDの情報のままにするために、この計算が必要。
        niMHD = rhoMHD / (mIon + mElectron);
        neMHD = niMHD;
        tiMHD = pMHD / 2.0 / niMHD;
        teMHD = pMHD / 2.0 / neMHD;
        
        rhoPICtoMHD =  mIon * zerothMomentIon_PICtoMHD[indexPICtoMHD].n + mElectron * ZerothMomentElectron_PICtoMHD[indexPICtoMHD].n;
        uPICtoMHD   = (mIon * firstMomentIon_PICtoMHD[indexPICtoMHD].x  + mElectron * firstMomentElectron_PICtoMHD[indexPICtoMHD].x) / rhoPICtoMHD;
        vPICtoMHD   = (mIon * firstMomentIon_PICtoMHD[indexPICtoMHD].y  + mElectron * firstMomentElectron_PICtoMHD[indexPICtoMHD].y) / rhoPICtoMHD;
        wPICtoMHD   = (mIon * firstMomentIon_PICtoMHD[indexPICtoMHD].z  + mElectron * firstMomentElectron_PICtoMHD[indexPICtoMHD].z) / rhoPICtoMHD;
        bXPICtoMHD  = B_PICtoMHD[indexPICtoMHD].bX; 
        bYPICtoMHD  = B_PICtoMHD[indexPICtoMHD].bY; 
        bZPICtoMHD  = B_PICtoMHD[indexPICtoMHD].bZ; 

        int indexForInterlocking = j * Interface2DConst::device_gridSizeRatio + (i * Interface2DConst::device_gridSizeRatio + bufferPIC) * PIC2DConst::device_ny; 

        rhoMHD = interlockingFunctionY[indexForInterlocking] * rhoMHD + (1.0 - interlockingFunctionY[indexForInterlocking]) * rhoPICtoMHD;
        uMHD   = interlockingFunctionY[indexForInterlocking] * uMHD   + (1.0 - interlockingFunctionY[indexForInterlocking]) * uPICtoMHD;
        vMHD   = interlockingFunctionY[indexForInterlocking] * vMHD   + (1.0 - interlockingFunctionY[indexForInterlocking]) * vPICtoMHD;
        wMHD   = interlockingFunctionY[indexForInterlocking] * wMHD   + (1.0 - interlockingFunctionY[indexForInterlocking]) * wPICtoMHD;
        bXMHD  = interlockingFunctionY[indexForInterlocking] * bXMHD  + (1.0 - interlockingFunctionY[indexForInterlocking]) * bXPICtoMHD;
        bYMHD  = interlockingFunctionY[indexForInterlocking] * bYMHD  + (1.0 - interlockingFunctionY[indexForInterlocking]) * bYPICtoMHD;
        bZMHD  = interlockingFunctionY[indexForInterlocking] * bZMHD  + (1.0 - interlockingFunctionY[indexForInterlocking]) * bZPICtoMHD;

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
    dim3 blocksPerGrid((mPIInfoMHD.localNx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (PIC2DConst::ny / Interface2DConst::gridSizeRatio + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendPICtoMHD_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()), 
        thrust::raw_pointer_cast(B_PICtoMHD.data()), 
        thrust::raw_pointer_cast(zerothMomentIon_PICtoMHD.data()), 
        thrust::raw_pointer_cast(zerothMomentElectron_PICtoMHD.data()), 
        thrust::raw_pointer_cast(firstMomentIon_PICtoMHD.data()), 
        thrust::raw_pointer_cast(firstMomentElectron_PICtoMHD.data()), 
        thrust::raw_pointer_cast(U.data()), 
        indexOfInterfaceStartInMHD, 
        mPIInfoMHD.localNx, mPIInfoPIC.buffer, mPIInfoMHD.buffer
    );
}


