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
    int indexOfInterfaceStartInPIC, 
    int buffer, 
    int localSizeXPIC, 
    int localSizeXMHD, 
    int interfaceSizeX
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < interfaceSizeX - 1 && 0 < j && j < Interface2DConst::device_interfaceLength - 1) {
        int indexPIC = indexOfInterfaceStartInPIC + j + i * PIC2DConst::device_ny;
        int indexMHD = indexOfInterfaceStartInMHD + j + i * IdealMHD2DConst::device_ny;
        double rhoMHD, uMHD, vMHD, wMHD, bXMHD, bYMHD, bZMHD, eMHD, pMHD;
        double rhoPIC, uPIC, vPIC, wPIC, bXPIC, bYPIC, bZPIC;
        double niMHD, neMHD, tiMHD, teMHD;
        double mIon = PIC2DConst::device_mIon, mElectron = PIC2DConst::device_mElectron;

        //MHDのグリッドにPICを合わせる
        rhoMHD      = max(U[indexMHD].rho, IdealMHD2DConst::device_rho0 * 0.1);
        uMHD        = U[indexMHD].rhoU / (rhoMHD + IdealMHD2DConst::device_EPS);
        vMHD        = U[indexMHD].rhoV / (rhoMHD + IdealMHD2DConst::device_EPS);
        wMHD        = U[indexMHD].rhoW / (rhoMHD + IdealMHD2DConst::device_EPS);
        bXMHD       = U[indexMHD].bX;
        bYMHD       = U[indexMHD].bY;
        bZMHD       = U[indexMHD].bZ;
        eMHD        = U[indexMHD].e;
        pMHD        = (IdealMHD2DConst::device_gamma - 1.0)
                    * (eMHD - 0.5 * rhoMHD * (uMHD * uMHD + vMHD * vMHD + wMHD * wMHD)
                    - 0.5 * (bXMHD * bXMHD + bYMHD * bYMHD + bZMHD * bZMHD));
        pMHD        = max(pMHD, IdealMHD2DConst::device_p0 * 0.1);

        //tiMHD, teMHDはMHDの情報のままにするために、この計算が必要。
        niMHD = rhoMHD / (mIon + mElectron);
        neMHD = niMHD;
        tiMHD = pMHD / 2.0 / niMHD;
        teMHD = pMHD / 2.0 / neMHD;
        
        rhoPIC =  max(mIon * zerothMomentIon[indexPIC].n + mElectron * ZerothMomentElectron[indexPIC].n, IdealMHD2DConst::device_rho0 * 0.1);
        uPIC   = (mIon * firstMomentIon[indexPIC].x  + mElectron * firstMomentElectron[indexPIC].x) / (rhoPIC + PIC2DConst::device_EPS);
        vPIC   = (mIon * firstMomentIon[indexPIC].y  + mElectron * firstMomentElectron[indexPIC].y) / (rhoPIC + PIC2DConst::device_EPS);
        wPIC   = (mIon * firstMomentIon[indexPIC].z  + mElectron * firstMomentElectron[indexPIC].z) / (rhoPIC + PIC2DConst::device_EPS);
        bXPIC  = B[indexPIC].bX; 
        bYPIC  = B[indexPIC].bY; 
        bZPIC  = B[indexPIC].bZ; 

        rhoMHD = interlockingFunctionY[j] * rhoMHD + (1.0 - interlockingFunctionY[j]) * rhoPIC;
        uMHD   = interlockingFunctionY[j] * uMHD   + (1.0 - interlockingFunctionY[j]) * uPIC;
        vMHD   = interlockingFunctionY[j] * vMHD   + (1.0 - interlockingFunctionY[j]) * vPIC;
        wMHD   = interlockingFunctionY[j] * wMHD   + (1.0 - interlockingFunctionY[j]) * wPIC;
        bXMHD  = interlockingFunctionY[j] * bXMHD  + (1.0 - interlockingFunctionY[j]) * bXPIC;
        bYMHD  = interlockingFunctionY[j] * bYMHD  + (1.0 - interlockingFunctionY[j]) * bYPIC;
        bZMHD  = interlockingFunctionY[j] * bZMHD  + (1.0 - interlockingFunctionY[j]) * bZPIC;

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
    dim3 blocksPerGrid((localSizeXInterface + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (Interface2DConst::interfaceLength + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendPICtoMHD_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()), 
        thrust::raw_pointer_cast(zerothMomentIon_timeAve.data()), 
        thrust::raw_pointer_cast(zerothMomentElectron_timeAve.data()), 
        thrust::raw_pointer_cast(firstMomentIon_timeAve.data()), 
        thrust::raw_pointer_cast(firstMomentElectron_timeAve.data()), 
        thrust::raw_pointer_cast(B_timeAve.data()), 
        thrust::raw_pointer_cast(U.data()), 
        indexOfInterfaceStartInMHD, 
        indexOfInterfaceStartInPIC, 
        mPIInfoMHD.buffer, 
        localSizeXPIC, 
        localSizeXMHD, 
        localSizeXInterface
    );
}


