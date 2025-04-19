#include "interface.hpp"


__global__ void sendPICtoMHD_kernel(
    const double* interlockingFunctionY, 
    const MagneticField* B_PICtoMHD, 
    const ZerothMoment* zerothMomentIon_PICtoMHD, 
    const ZerothMoment* zerothMomentElectron_PICtoMHD, 
    const FirstMoment* firstMomentIon_PICtoMHD, 
    const FirstMoment* firstMomentElectron_PICtoMHD, 
    const SecondMoment* secondMomentIon_PICtoMHD, 
    const SecondMoment* secondMomentElectron_PICtoMHD, 
    ConservationParameter* U, 
    const int indexOfInterfaceStartInMHD, 
    const int localNxMHD, const int bufferPIC, const int bufferMHD
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localNxMHD && j < PIC2DConst::device_ny / Interface2DConst::device_gridSizeRatio) {
        int indexMHD = indexOfInterfaceStartInMHD + j  + (i + bufferMHD) * IdealMHD2DConst::device_ny;
        double rhoMHD, uMHD, vMHD, wMHD, bXMHD, bYMHD, bZMHD, eMHD, pMHD;

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
        
        int indexPICtoMHD = j + i * PIC2DConst::device_ny / Interface2DConst::device_gridSizeRatio;
        double rhoPICtoMHD, uPICtoMHD, vPICtoMHD, wPICtoMHD, bXPICtoMHD, bYPICtoMHD, bZPICtoMHD, pPICtoMHD;

        rhoPICtoMHD =  PIC2DConst::device_mIon * zerothMomentIon_PICtoMHD[indexPICtoMHD].n + PIC2DConst::device_mElectron * zerothMomentElectron_PICtoMHD[indexPICtoMHD].n;
        uPICtoMHD   = (PIC2DConst::device_mIon * firstMomentIon_PICtoMHD[indexPICtoMHD].x  + PIC2DConst::device_mElectron * firstMomentElectron_PICtoMHD[indexPICtoMHD].x) / rhoPICtoMHD;
        vPICtoMHD   = (PIC2DConst::device_mIon * firstMomentIon_PICtoMHD[indexPICtoMHD].y  + PIC2DConst::device_mElectron * firstMomentElectron_PICtoMHD[indexPICtoMHD].y) / rhoPICtoMHD;
        wPICtoMHD   = (PIC2DConst::device_mIon * firstMomentIon_PICtoMHD[indexPICtoMHD].z  + PIC2DConst::device_mElectron * firstMomentElectron_PICtoMHD[indexPICtoMHD].z) / rhoPICtoMHD;
        bXPICtoMHD  = B_PICtoMHD[indexPICtoMHD].bX; 
        bYPICtoMHD  = B_PICtoMHD[indexPICtoMHD].bY; 
        bZPICtoMHD  = B_PICtoMHD[indexPICtoMHD].bZ; 
        pPICtoMHD   = PIC2DConst::device_mIon
                    * (secondMomentIon_PICtoMHD[indexPICtoMHD].xx + secondMomentIon_PICtoMHD[indexPICtoMHD].yy + secondMomentIon_PICtoMHD[indexPICtoMHD].zz
                    - (pow(firstMomentIon_PICtoMHD[indexPICtoMHD].x, 2) + pow(firstMomentIon_PICtoMHD[indexPICtoMHD].y, 2) + pow(firstMomentIon_PICtoMHD[indexPICtoMHD].z, 2))
                    / (zerothMomentIon_PICtoMHD[indexPICtoMHD].n + Interface2DConst::device_EPS)) / 3.0
                    + PIC2DConst::device_mElectron
                    * (secondMomentElectron_PICtoMHD[indexPICtoMHD].xx + secondMomentElectron_PICtoMHD[indexPICtoMHD].yy + secondMomentElectron_PICtoMHD[indexPICtoMHD].zz
                    - (pow(firstMomentElectron_PICtoMHD[indexPICtoMHD].x, 2) + pow(firstMomentElectron_PICtoMHD[indexPICtoMHD].y, 2) + pow(firstMomentElectron_PICtoMHD[indexPICtoMHD].z, 2))
                    / (zerothMomentElectron_PICtoMHD[indexPICtoMHD].n + Interface2DConst::device_EPS)) / 3.0;

        int indexForInterlocking = j * Interface2DConst::device_gridSizeRatio + (i * Interface2DConst::device_gridSizeRatio + bufferPIC) * PIC2DConst::device_ny; 

        rhoMHD = interlockingFunctionY[indexForInterlocking] * rhoMHD + (1.0 - interlockingFunctionY[indexForInterlocking]) * rhoPICtoMHD;
        uMHD   = interlockingFunctionY[indexForInterlocking] * uMHD   + (1.0 - interlockingFunctionY[indexForInterlocking]) * uPICtoMHD;
        vMHD   = interlockingFunctionY[indexForInterlocking] * vMHD   + (1.0 - interlockingFunctionY[indexForInterlocking]) * vPICtoMHD;
        wMHD   = interlockingFunctionY[indexForInterlocking] * wMHD   + (1.0 - interlockingFunctionY[indexForInterlocking]) * wPICtoMHD;
        bXMHD  = interlockingFunctionY[indexForInterlocking] * bXMHD  + (1.0 - interlockingFunctionY[indexForInterlocking]) * bXPICtoMHD;
        bYMHD  = interlockingFunctionY[indexForInterlocking] * bYMHD  + (1.0 - interlockingFunctionY[indexForInterlocking]) * bYPICtoMHD;
        bZMHD  = interlockingFunctionY[indexForInterlocking] * bZMHD  + (1.0 - interlockingFunctionY[indexForInterlocking]) * bZPICtoMHD;
        pMHD   = interlockingFunctionY[indexForInterlocking] * pMHD   + (1.0 - interlockingFunctionY[indexForInterlocking]) * pPICtoMHD;

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
        thrust::raw_pointer_cast(secondMomentIon_PICtoMHD.data()), 
        thrust::raw_pointer_cast(secondMomentElectron_PICtoMHD.data()), 
        thrust::raw_pointer_cast(U.data()), 
        indexOfInterfaceStartInMHD, 
        mPIInfoMHD.localNx, mPIInfoPIC.buffer, mPIInfoMHD.buffer
    );
}


