#include "interface.hpp"


__global__ void sendPICtoMHD_kernel(
    const double* interlockingFunction, 
    const MagneticField* B_PICtoMHD, 
    const ZerothMoment* zerothMomentIon_PICtoMHD, 
    const ZerothMoment* zerothMomentElectron_PICtoMHD, 
    const FirstMoment* firstMomentIon_PICtoMHD, 
    const FirstMoment* firstMomentElectron_PICtoMHD, 
    const SecondMoment* secondMomentIon_PICtoMHD, 
    const SecondMoment* secondMomentElectron_PICtoMHD, 
    ConservationParameter* U, 
    const int indexOfInterfaceStartInMHD_x, 
    const int indexOfInterfaceStartInMHD_y
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < PIC2DConst::device_nx / Interface2DConst::device_gridSizeRatio && j < PIC2DConst::device_ny / Interface2DConst::device_gridSizeRatio) {
        unsigned long long indexMHD = indexOfInterfaceStartInMHD_y + static_cast<int>(j / Interface2DConst::device_gridSizeRatio)
                                    + (indexOfInterfaceStartInMHD_x + static_cast<int>(i / Interface2DConst::device_gridSizeRatio))
                                    * IdealMHD2DConst::device_ny;
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
        
        unsigned long long indexPICtoMHD = j + i * PIC2DConst::device_ny / Interface2DConst::device_gridSizeRatio;
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

        unsigned long long indexForInterlocking = j * Interface2DConst::device_gridSizeRatio + (i * Interface2DConst::device_gridSizeRatio) * PIC2DConst::device_ny; 

        rhoMHD = interlockingFunction[indexForInterlocking] * rhoMHD + (1.0 - interlockingFunction[indexForInterlocking]) * rhoPICtoMHD;
        uMHD   = interlockingFunction[indexForInterlocking] * uMHD   + (1.0 - interlockingFunction[indexForInterlocking]) * uPICtoMHD;
        vMHD   = interlockingFunction[indexForInterlocking] * vMHD   + (1.0 - interlockingFunction[indexForInterlocking]) * vPICtoMHD;
        wMHD   = interlockingFunction[indexForInterlocking] * wMHD   + (1.0 - interlockingFunction[indexForInterlocking]) * wPICtoMHD;
        bXMHD  = interlockingFunction[indexForInterlocking] * bXMHD  + (1.0 - interlockingFunction[indexForInterlocking]) * bXPICtoMHD;
        bYMHD  = interlockingFunction[indexForInterlocking] * bYMHD  + (1.0 - interlockingFunction[indexForInterlocking]) * bYPICtoMHD;
        bZMHD  = interlockingFunction[indexForInterlocking] * bZMHD  + (1.0 - interlockingFunction[indexForInterlocking]) * bZPICtoMHD;
        pMHD   = interlockingFunction[indexForInterlocking] * pMHD   + (1.0 - interlockingFunction[indexForInterlocking]) * pPICtoMHD;

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
    dim3 blocksPerGrid((PIC2DConst::nx / Interface2DConst::gridSizeRatio + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (PIC2DConst::ny / Interface2DConst::gridSizeRatio + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendPICtoMHD_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunction.data()), 
        thrust::raw_pointer_cast(B_PICtoMHD.data()), 
        thrust::raw_pointer_cast(zerothMomentIon_PICtoMHD.data()), 
        thrust::raw_pointer_cast(zerothMomentElectron_PICtoMHD.data()), 
        thrust::raw_pointer_cast(firstMomentIon_PICtoMHD.data()), 
        thrust::raw_pointer_cast(firstMomentElectron_PICtoMHD.data()), 
        thrust::raw_pointer_cast(secondMomentIon_PICtoMHD.data()), 
        thrust::raw_pointer_cast(secondMomentElectron_PICtoMHD.data()), 
        thrust::raw_pointer_cast(U.data()), 
        indexOfInterfaceStartInMHD_x, 
        indexOfInterfaceStartInMHD_y
    );
    cudaDeviceSynchronize();
}


