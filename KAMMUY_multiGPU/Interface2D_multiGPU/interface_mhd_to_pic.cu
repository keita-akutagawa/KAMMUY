#include "interface.hpp"


__global__ void sendMHDtoPIC_magneticField_yDirection_kernel(
    const double* interlockingFunctionY, 
    const double* interlockingFunctionYHalf, 
    const ConservationParameter* U, 
    MagneticField* B, 
    int indexOfInterfaceStartInMHD, 
    int indexOfInterfaceStartInPIC, 
    int interfaceLength, 
    int localNxPIC, int localNyPIC, int buffer, 
    int localSizeXPIC, int localSizeYPIC, 
    int localSizeXMHD, int localSizeYMHD, 
    int interfaceSizeX, int interfaceSizeY
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < interfaceSizeX - 1 && 0 < y && y < interfaceSizeY - 1) {
        double bXPIC, bYPIC, bZPIC;
        double bXMHD, bYMHD, bZMHD;
        double bXInterface, bYInterface, bZInterface;

        int indexPIC = indexOfInterfaceStartInPIC + j + (i + buffer) * localSizeYPIC;
        int indexMHD = indexOfInterfaceStartInMHD + j + (i + buffer) * localSizeYMHD;

        //PICのグリッドにMHDを合わせる
        bXPIC = B[indexPIC].bX;
        bYPIC = B[indexPIC].bY;
        bZPIC = B[indexPIC].bZ;
        bXMHD = 0.25 * (U[indexMHD].bX + U[indexMHD - localSizeYMHD].bX + U[indexMHD + 1].bX + U[indexMHD + 1 - localSizeYMHD].bX);
        bYMHD = 0.25 * (U[indexMHD].bY + U[indexMHD + localSizeYMHD].bY + U[indexMHD - 1].bY + U[indexMHD - 1 + localSizeYMHD].bY);
        bZMHD = 0.25 * (U[indexMHD].bZ + U[indexMHD + localSizeYMHD].bZ + U[indexMHD + 1].bZ + U[indexMHD + 1 + localSizeYMHD].bZ);

        bXInterface = interlockingFunctionYHalf[j] * bXMHD + (1.0 - interlockingFunctionYHalf[j]) * bXPIC;
        bYInterface = interlockingFunctionY[j]     * bYMHD + (1.0 - interlockingFunctionY[j])     * bYPIC;
        bZInterface = interlockingFunctionYHalf[j] * bZMHD + (1.0 - interlockingFunctionYHalf[j]) * bZPIC;
        
        B[indexPIC].bX = bXInterface;
        B[indexPIC].bY = bYInterface;
        B[indexPIC].bZ = bZInterface;
    }
}

void Interface2D::sendMHDtoPIC_magneticField_yDirection(
    const thrust::device_vector<ConservationParameter>& U, 
    thrust::device_vector<MagneticField>& B
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((interfaceSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (interfaceSizeY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendMHDtoPIC_magneticField_yDirection_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()),
        thrust::raw_pointer_cast(interlockingFunctionYHalf.data()),
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(B.data()), 
        indexOfInterfaceStartInMHD, 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        mPIInfoPIC.localNx, mPIInfoPIC.localNy, mPIInfoPIC.buffer, 
        mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY, 
        mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY, 
        interfaceSizeX, interfaceSizeY
    );
    cudaDeviceSynchronize();
}


__global__ void sendMHDtoPIC_electricField_yDirection_kernel(
    const double* interlockingFunctionY, 
    const double* interlockingFunctionYHalf, 
    const ConservationParameter* U, 
    ElectricField* E, 
    int indexOfInterfaceStartInMHD, 
    int indexOfInterfaceStartInPIC, 
    int interfaceLength, 
    int localNxPIC, int localNyPIC, int buffer, 
    int localSizeXPIC, int localSizeYPIC, 
    int localSizeXMHD, int localSizeYMHD
    int interfaceSizeX, int interfaceSizeY
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < interfaceSizeX - 1 && 0 < y && y < interfaceSizeY - 1) {
        double eXPIC, eYPIC, eZPIC;
        double eXMHD, eYMHD, eZMHD;
        double eXPlusX1MHD;
        double eYPlusY1MHD;
        double rho, u, v, w;
        double bXMHD, bYMHD, bZMHD;
        double eXInterface, eYInterface, eZInterface;

        int indexPIC = indexOfInterfaceStartInPIC + j + (i + buffer) * localSizeYPIC;
        int indexMHD = indexOfInterfaceStartInMHD + j + (i + buffer) * localSizeYMHD;

        //PICのグリッドにMHDを合わせる
        eXPIC = E[indexPIC].eX;
        eYPIC = E[indexPIC].eY;
        eZPIC = E[indexPIC].eZ;

        rho = max(U[indexMHD].rho, IdealMHD2DConst::device_EPS);
        u = U[indexMHD].rhoU / (rho + IdealMHD2DConst::device_EPS);
        v = U[indexMHD].rhoV / (rho + IdealMHD2DConst::device_EPS);
        w = U[indexMHD].rhoW / (rho + IdealMHD2DConst::device_EPS); 
        bXMHD = 0.5 * (U[indexMHD].bX + U[indexMHD - localSizeYMHD].bX);
        bYMHD = 0.5 * (U[indexMHD].bY + U[indexMHD - 1].bY);
        bZMHD = U[indexMHD].bZ;
        eXMHD = -(v * bZMHD - w * bYMHD);
        eYMHD = -(w * bXMHD - u * bZMHD);
        eZMHD = -(u * bYMHD - v * bXMHD);

        rho = max(U[indexMHD + localSizeYMHD].rho, IdealMHD2DConst::device_EPS);
        u = U[indexMHD + localSizeYMHD].rhoU / (rho + IdealMHD2DConst::device_EPS);
        v = U[indexMHD + localSizeYMHD].rhoV / (rho + IdealMHD2DConst::device_EPS);
        w = U[indexMHD + localSizeYMHD].rhoW / (rho + IdealMHD2DConst::device_EPS); 
        bXMHD = 0.5 * (U[indexMHD + localSizeYMHD].bX + U[indexMHD].bX);
        bYMHD = 0.5 * (U[indexMHD + localSizeYMHD].bY + U[indexMHD - 1 + localSizeYMHD].bY);
        bZMHD = U[indexMHD + localSizeYMHD].bZ;
        eXPlusX1MHD = -(v * bZMHD - w * bYMHD);

        rho = max(U[indexMHD + 1].rho, IdealMHD2DConst::device_EPS);
        u = U[indexMHD + 1].rhoU / (rho + IdealMHD2DConst::device_EPS);
        v = U[indexMHD + 1].rhoV / (rho + IdealMHD2DConst::device_EPS);
        w = U[indexMHD + 1].rhoW / (rho + IdealMHD2DConst::device_EPS); 
        bXMHD = 0.5 * (U[indexMHD + 1].bX + U[indexMHD + 1 - localSizeYMHD].bX);
        bYMHD = 0.5 * (U[indexMHD + 1].bY + U[indexMHD].bY);
        bZMHD = U[indexMHD + 1].bZ;
        eYPlusY1MHD = -(w * bXMHD - u * bZMHD);


        eXInterface = interlockingFunctionY[j]     * 0.5 * (eXMHD + eXPlusX1MHD) + (1.0 - interlockingFunctionY[j])     * eXPIC;
        eYInterface = interlockingFunctionYHalf[j] * 0.5 * (eYMHD + eYPlusY1MHD) + (1.0 - interlockingFunctionYHalf[j]) * eYPIC;
        eZInterface = interlockingFunctionY[j]     * eZMHD                       + (1.0 - interlockingFunctionY[j])     * eZPIC;
         
        E[indexPIC].eX = eXInterface;
        E[indexPIC].eY = eYInterface;
        E[indexPIC].eZ = eZInterface;
    }
}

void Interface2D::sendMHDtoPIC_electricField_yDirection(
    const thrust::device_vector<ConservationParameter>& U, 
    thrust::device_vector<ElectricField>& E
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfoPIC.localNx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (interfaceLength + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendMHDtoPIC_electricField_yDirection_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()), 
        thrust::raw_pointer_cast(interlockingFunctionYHalf.data()), 
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(E.data()), 
        indexOfInterfaceStartInMHD, 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        mPIInfoPIC.localNx, mPIInfoPIC.localNy, mPIInfoPIC.buffer, 
        mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY, 
        mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY, 
        interfaceSizeX, interfaceSizeY
    );
    cudaDeviceSynchronize();
}


__global__ void sendMHDtoPIC_currentField_yDirection_kernel(
    const double* interlockingFunctionY, 
    const double* interlockingFunctionYHalf, 
    const ConservationParameter* U, 
    CurrentField* current, 
    int indexOfInterfaceStartInMHD, 
    int indexOfInterfaceStartInPIC, 
    int interfaceLength, 
    int localNxPIC, int localNyPIC, int buffer, 
    int localSizeXPIC, int localSizeYPIC, 
    int localSizeXMHD, int localSizeYMHD, 
    int interfaceSizeX, int interfaceSizeY
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < interfaceSizeX - 1 && 0 < y && y < interfaceSizeY - 1) {
        double jXPIC, jYPIC, jZPIC;
        double jXMHD, jYMHD, jZMHD;
        double jXPlusX1MHD; 
        double jYPlusY1MHD; 
        double jXInterface, jYInterface, jZInterface;
        double dx = IdealMHD2DConst::device_dx, dy = IdealMHD2DConst::device_dy;

        int indexPIC = indexOfInterfaceStartInPIC + j + (i + buffer) * localSizeYPIC;
        int indexMHD = indexOfInterfaceStartInMHD + j + (i + buffer) * localSizeYMHD;

        //PICのグリッドにMHDを合わせる
        jXPIC = current[indexPIC].jX;
        jYPIC = current[indexPIC].jY;
        jZPIC = current[indexPIC].jZ;
        jXMHD = (U[indexMHD + 1].bZ - U[indexMHD - 1].bZ) / (2.0 * dy);
        jYMHD = -(U[indexMHD + localSizeYMHD].bZ - U[indexMHD - localSizeYMHD].bZ) / (2.0 * dx);
        jZMHD = 0.25 * (
                (U[indexMHD + localSizeYMHD].bY - U[indexMHD].bY) / dx - (U[indexMHD + 1].bX - U[indexMHD].bX) / dy 
              + (U[indexMHD].bY - U[indexMHD - localSizeYMHD].bY) / dx - (U[indexMHD + 1 - localSizeYMHD].bX - U[indexMHD - localSizeYMHD].bX) / dy
              + (U[indexMHD - 1 + localSizeYMHD].bY - U[indexMHD - 1].bY) / dx - (U[indexMHD].bX - U[indexMHD - 1].bX) / dy
              + (U[indexMHD - 1].bY - U[indexMHD - 1 - localSizeYMHD].bY) / dx - (U[indexMHD - localSizeYMHD].bX - U[indexMHD - 1 - localSizeYMHD].bX) / dy);

        jXPlusX1MHD = (U[indexMHD + 1 + localSizeYMHD].bZ - U[indexMHD - 1 + localSizeYMHD].bZ) / (2.0 * dy);
        jYPlusY1MHD = -(U[indexMHD + localSizeYMHD + 1].bZ - U[indexMHD - localSizeYMHD + 1].bZ) / (2.0 * dx);

        jXInterface = interlockingFunctionY[j]     * 0.5 * (jXMHD + jXPlusX1MHD) + (1.0 - interlockingFunctionY[j])     * jXPIC;
        jYInterface = interlockingFunctionYHalf[j] * 0.5 * (jYMHD + jYPlusY1MHD) + (1.0 - interlockingFunctionYHalf[j]) * jYPIC;
        jZInterface = interlockingFunctionY[j]     * jZMHD                       + (1.0 - interlockingFunctionY[j])     * jZPIC;
        
        current[indexPIC].jX = jXInterface;
        current[indexPIC].jY = jYInterface;
        current[indexPIC].jZ = jZInterface;
    }
}

void Interface2D::sendMHDtoPIC_currentField_yDirection(
    const thrust::device_vector<ConservationParameter>& U, 
    thrust::device_vector<CurrentField>& current
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfoPIC.localNx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (interfaceLength + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendMHDtoPIC_currentField_yDirection_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()), 
        thrust::raw_pointer_cast(interlockingFunctionYHalf.data()), 
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(current.data()), 
        indexOfInterfaceStartInMHD, 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        mPIInfoPIC.localNx, mPIInfoPIC.localNy, mPIInfoPIC.buffer, 
        mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY, 
        mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY, 
        interfaceSizeX, interfaceSizeY
    );
    cudaDeviceSynchronize();
}


__global__ void sendMHDtoPIC_particle_yDirection_kernel(
    const double* interlockingFunctionY, 
    const ZerothMoment* zerothMomentIon, 
    const ZerothMoment* zerothMomentElectron, 
    const FirstMoment* firstMomentIon, 
    const FirstMoment* firstMomentElectron, 
    const ConservationParameter* U, 
    ReloadParticlesData* reloadParticlesDataIon, 
    ReloadParticlesData* reloadParticlesDataElectron, 
    int indexOfInterfaceStartInMHD, 
    int indexOfInterfaceStartInPIC, 
    int interfaceLength, 
    int localNxPIC, int localNyPIC, int buffer, 
    int localSizeXPIC, int localSizeYPIC, 
    int localSizeXMHD, int localSizeYMHD, 
    int interfaceSizeX, int interfaceSizeY
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < interfaceSizeX - 1 && 0 < y && y < interfaceSizeY - 1) {
        int indexForReload = j + i * interfaceLength; 
        int indexPIC = indexOfInterfaceStartInPIC + j + (i + buffer) * localSizeYPIC;
        int indexMHD = indexOfInterfaceStartInMHD + j + (i + buffer) * localSizeYMHD;
        double rhoMHD, uMHD, vMHD, wMHD, bXMHD, bYMHD, bZMHD, eMHD, pMHD;
        double jXMHD, jYMHD, jZMHD, niMHD, neMHD, tiMHD, teMHD;
        double rhoPIC, uPIC, vPIC, wPIC;
        double jXPIC, jYPIC, jZPIC, niPIC, nePIC, vThiPIC, vThePIC;
        double dx = IdealMHD2DConst::device_dx, dy = IdealMHD2DConst::device_dy;
        double mIon = PIC2DConst::device_mIon, mElectron = PIC2DConst::device_mElectron;
        double qIon = PIC2DConst::device_qIon, qElectron = PIC2DConst::device_qElectron;

        //整数格子点上で計算する。リロードに使う。
        rhoMHD = max(U[indexMHD].rho, IdealMHD2DConst::device_EPS);
        uMHD   = U[indexMHD].rhoU / (rhoMHD + IdealMHD2DConst::device_EPS);
        vMHD   = U[indexMHD].rhoV / (rhoMHD + IdealMHD2DConst::device_EPS);
        wMHD   = U[indexMHD].rhoW / (rhoMHD + IdealMHD2DConst::device_EPS);
        bXMHD  = 0.5 * (U[indexMHD].bX + U[indexMHD - localSizeYMHD].bX);
        bYMHD  = 0.5 * (U[indexMHD].bY + U[indexMHD - 1].bY);
        bZMHD  = U[indexMHD].bZ;
        eMHD   = U[indexMHD].e;
        pMHD   = (IdealMHD2DConst::device_gamma - 1.0)
               * (eMHD - 0.5 * rhoMHD * (uMHD * uMHD + vMHD * vMHD + wMHD * wMHD)
               - 0.5 * (bXMHD * bXMHD + bYMHD * bYMHD + bZMHD * bZMHD));
        pMHD   = max(pMHD, IdealMHD2DConst::device_EPS);
        jXMHD  = (U[indexMHD + 1].bZ - U[indexMHD - 1].bZ) / (2.0 * dy);
        jYMHD  = -(U[indexMHD + localSizeYMHD].bZ - U[indexMHD - localSizeYMHD].bZ) / (2.0 * dx);
        jZMHD  = 0.25 * (
                 (U[indexMHD + localSizeYMHD].bY - U[indexMHD].bY) / dx - (U[indexMHD + 1].bX - U[indexMHD].bX) / dy 
               + (U[indexMHD].bY - U[indexMHD - localSizeYMHD].bY) / dx - (U[indexMHD + 1 - localSizeYMHD].bX - U[indexMHD - localSizeYMHD].bX) / dy
               + (U[indexMHD - 1 + localSizeYMHD].bY - U[indexMHD - 1].bY) / dx - (U[indexMHD].bX - U[indexMHD - 1].bX) / dy
               + (U[indexMHD - 1].bY - U[indexMHD - 1 - localSizeYMHD].bY) / dx - (U[indexMHD - localSizeYMHD].bX - U[indexMHD - 1 - localSizeYMHD].bX) / dy);

        niMHD = rhoMHD / (mIon + mElectron);
        neMHD = niMHD;
        tiMHD = pMHD / 2.0 / niMHD;
        teMHD = pMHD / 2.0 / neMHD;

        rhoPIC =  max(mIon * zerothMomentIon[indexPIC].n + mElectron * zerothMomentElectron[indexPIC].n, PIC2DConst::device_EPS);
        uPIC   = (mIon * firstMomentIon[indexPIC].x  + mElectron * firstMomentElectron[indexPIC].x) / (rhoPIC + PIC2DConst::device_EPS);
        vPIC   = (mIon * firstMomentIon[indexPIC].y  + mElectron * firstMomentElectron[indexPIC].y) / (rhoPIC + PIC2DConst::device_EPS);
        wPIC   = (mIon * firstMomentIon[indexPIC].z  + mElectron * firstMomentElectron[indexPIC].z) / (rhoPIC + PIC2DConst::device_EPS);
        jXPIC  = qIon  * firstMomentIon[indexPIC].x  + qElectron * firstMomentElectron[indexPIC].x;
        jYPIC  = qIon  * firstMomentIon[indexPIC].y  + qElectron * firstMomentElectron[indexPIC].y;
        jZPIC  = qIon  * firstMomentIon[indexPIC].z  + qElectron * firstMomentElectron[indexPIC].z;

        rhoPIC = interlockingFunctionY[j] * rhoMHD + (1.0 - interlockingFunctionY[j]) * rhoPIC;
        uPIC   = interlockingFunctionY[j] * uMHD   + (1.0 - interlockingFunctionY[j]) * uPIC;
        vPIC   = interlockingFunctionY[j] * vMHD   + (1.0 - interlockingFunctionY[j]) * vPIC;
        wPIC   = interlockingFunctionY[j] * wMHD   + (1.0 - interlockingFunctionY[j]) * wPIC;
        jXPIC  = interlockingFunctionY[j] * jXMHD  + (1.0 - interlockingFunctionY[j]) * jXPIC;
        jYPIC  = interlockingFunctionY[j] * jYMHD  + (1.0 - interlockingFunctionY[j]) * jYPIC;
        jZPIC  = interlockingFunctionY[j] * jZMHD  + (1.0 - interlockingFunctionY[j]) * jZPIC;

        niPIC   = rhoPIC / (mIon + mElectron);
        nePIC   = niPIC;
        vThiPIC = sqrt(2.0 * tiMHD / mIon);
        vThePIC = sqrt(2.0 * teMHD / mElectron);


        reloadParticlesDataIon     [indexForReload].numberAndIndex = max(static_cast<unsigned long long>(round(niPIC)), static_cast<unsigned long long>(1));
        reloadParticlesDataElectron[indexForReload].numberAndIndex = max(static_cast<unsigned long long>(round(nePIC)), static_cast<unsigned long long>(1));
        reloadParticlesDataIon     [indexForReload].u              = uPIC;
        reloadParticlesDataIon     [indexForReload].v              = vPIC;
        reloadParticlesDataIon     [indexForReload].w              = wPIC;
        reloadParticlesDataElectron[indexForReload].u              = uPIC - jXPIC / max(round(nePIC), 1.0) / abs(qElectron);
        reloadParticlesDataElectron[indexForReload].v              = vPIC - jYPIC / max(round(nePIC), 1.0) / abs(qElectron);
        reloadParticlesDataElectron[indexForReload].w              = wPIC - jZPIC / max(round(nePIC), 1.0) / abs(qElectron);
        reloadParticlesDataIon     [indexForReload].vth            = vThiPIC;
        reloadParticlesDataElectron[indexForReload].vth            = vThePIC;
    }
}


__global__ void deleteParticles_kernel(
    const double* interlockingFunctionY, 
    Particle* particlesSpecies, 
    const int indexOfInterfaceStartInPIC, 
    int interfaceLength, 
    const unsigned long long existNumSpecies, 
    int step, 
    const float xminForProcs, const float xmaxForProcs, 
    const float yminForProcs, const float ymaxForProcs, 
    const int buffer
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        float x = particlesSpecies[i].x;
        float y = particlesSpecies[i].y;
        float deleteXMin = xminForProcs - buffer * PIC2DConst::device_dx;
        float deleteXMax = xmaxForProcs + buffer * PIC2DConst::device_dx;
        float deleteYMin = (indexOfInterfaceStartInPIC - buffer) * PIC2DConst::device_dy + PIC2DConst::device_ymin;
        float deleteYMax = (indexOfInterfaceStartInPIC - buffer + interfaceLength) * PIC2DConst::device_dy + PIC2DConst::device_ymin;

        if (deleteXMin < x && x < deleteXMax && deleteYMin < y && y < deleteYMax) {
            int j = floor(y) - deleteYMin;
            curandState state; 
            curand_init(step, i, 0, &state);
            double randomValue = curand_uniform_double(&state);
            if (randomValue < interlockingFunctionY[j]) {
                particlesSpecies[i].isExist = false;
            }
        }
    }
}


void Interface2D::deleteParticles(
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron, 
    int step
)
{

    dim3 threadsPerBlockForIon(256);
    dim3 blocksPerGridForIon((mPIInfoPIC.existNumIonPerProcs + threadsPerBlockForIon.x - 1) / threadsPerBlockForIon.x);
    
    deleteParticles_kernel<<<blocksPerGridForIon, threadsPerBlockForIon>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()), 
        thrust::raw_pointer_cast(particlesIon.data()),
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        mPIInfoPIC.existNumIonPerProcs, 
        step, 
        mPIInfoPIC.xminForProcs, mPIInfoPIC.xmaxForProcs, 
        mPIInfoPIC.yminForProcs, mPIInfoPIC.ymaxForProcs, 
        mPIInfoPIC.buffer
    );
    cudaDeviceSynchronize();

    dim3 threadsPerBlockForElectron(256);
    dim3 blocksPerGridForElectron((mPIInfoPIC.existNumElectronPerProcs + threadsPerBlockForElectron.x - 1) / threadsPerBlockForElectron.x);

    deleteParticles_kernel<<<blocksPerGridForElectron, threadsPerBlockForElectron>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()), 
        thrust::raw_pointer_cast(particlesElectron.data()),
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        mPIInfoPIC.existNumElectronPerProcs, 
        step, 
        mPIInfoPIC.xminForProcs, mPIInfoPIC.xmaxForProcs, 
        mPIInfoPIC.yminForProcs, mPIInfoPIC.ymaxForProcs, 
        mPIInfoPIC.buffer
    );
    cudaDeviceSynchronize();

    auto partitionEndIon = thrust::partition(
        particlesIon.begin(), particlesIon.end(), 
        [] __device__ (const Particle& p) { return p.isExist; }
    );
    cudaDeviceSynchronize();

    auto partitionEndElectron = thrust::partition(
        particlesElectron.begin(), particlesElectron.end(), 
        [] __device__ (const Particle& p) { return p.isExist; }
    );
    cudaDeviceSynchronize();

    mPIInfoPIC.existNumIonPerProcs = thrust::distance(particlesIon.begin(), partitionEndIon);
    mPIInfoPIC.existNumElectronPerProcs = thrust::distance(particlesElectron.begin(), partitionEndElectron);
}


__global__ void reloadParticles_kernel(
    const double* interlockingFunctionY, 
    const ReloadParticlesData* reloadParticlesDataSpecies, 
    const Particle* reloadParticlesSpecies, 
    unsigned long long reloadParticlesTotalNumSpecies, 
    Particle* particlesSpecies, 
    unsigned long long restartParticlesIndexSpecies, 
    int indexOfInterfaceStartInPIC, 
    int interfaceLength, 
    unsigned long long existNumSpecies, 
    int step, 
    const float xminForProcs, const float xmaxForProcs, 
    const float yminForProcs, const float ymaxForProcs, 
    int localNxPIC, int localNyPIC, int buffer, 
    int localSizeXPIC, int localSizeYPIC, 
    int interfaceSizeX, int interfaceSizeY
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < interfaceSizeX - 1 && 0 < y && y < interfaceSizeY - 1) {
        int index = j + i * interfaceLength;
        double u = reloadParticlesDataSpecies[index].u;
        double v = reloadParticlesDataSpecies[index].v;
        double w = reloadParticlesDataSpecies[index].w;
        double vth = reloadParticlesDataSpecies[index].vth;
        Particle particleSource, particleReload;
        double x, y, z, vx, vy, vz, gamma;

        for (unsigned long long k = reloadParticlesDataSpecies[index].numberAndIndex; k < reloadParticlesDataSpecies[index + 1].numberAndIndex; k++) {
            curandState state; 
            curand_init(step, k, 0, &state);
            double randomValue = curand_uniform_double(&state);

            if (randomValue < interlockingFunctionY[j]) {
                particleSource = reloadParticlesSpecies[
                    static_cast<unsigned long long>((restartParticlesIndexSpecies + k) % reloadParticlesTotalNumSpecies)
                ];

                x = particleSource.x; x += i * PIC2DConst::device_dx + xminForProcs;
                y = particleSource.y; y += (indexOfInterfaceStartInPIC - buffer + j) * PIC2DConst::device_dy + PIC2DConst::device_ymin;
                z = particleSource.z;
                vx = particleSource.vx; vx = u + vx * vth;
                vy = particleSource.vy; vy = v + vy * vth;
                vz = particleSource.vz; vz = w + vz * vth;
                gamma = 1.0f / sqrt(1.0f - (vx * vx + vy * vy + vz * vz) / pow(PIC2DConst::device_c, 2));
                if (1.0f - (vx * vx + vy * vy + vz * vz) / pow(PIC2DConst::device_c, 2) < 0.0f){
                    printf("particle exceeds light speed... ");
                    continue; //delete if particle speed exceeds light speed c. 
                };

                particleReload.x = x; particleReload.y = y; particleReload.z = z;
                particleReload.vx = vx * gamma; particleReload.vy = vy * gamma, particleReload.vz = vz * gamma; 
                particleReload.gamma = gamma;
                particleReload.isExist = true;

                particlesSpecies[
                    static_cast<unsigned long long>(existNumSpecies + k)
                ] = particleReload;
            } 
        }
    }
}

void Interface2D::sendMHDtoPIC_particle(
    const thrust::device_vector<ConservationParameter>& U, 
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron, 
    int step
)
{
    setMoments(particlesIon, particlesElectron); 

    if (isLower) {
        interfaceNoiseRemover2D.convolveMoments(
            zerothMomentIon, zerothMomentElectron, 
            firstMomentIon, firstMomentElectron, 
            isLower, isUpper
        );
    }

    if (isUpper) { 
        interfaceNoiseRemover2D.convolveMoments(
            zerothMomentIon, zerothMomentElectron, 
            firstMomentIon, firstMomentElectron, 
            isLower, isUpper
        );
    }

    thrust::fill(reloadParticlesDataIon.begin(), reloadParticlesDataIon.end(), ReloadParticlesData());
    thrust::fill(reloadParticlesDataElectron.begin(), reloadParticlesDataElectron.end(), ReloadParticlesData());

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((interfaceSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (interfaceSizeY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendMHDtoPIC_particle_yDirection_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()), 
        thrust::raw_pointer_cast(zerothMomentIon.data()), 
        thrust::raw_pointer_cast(zerothMomentElectron.data()), 
        thrust::raw_pointer_cast(firstMomentIon.data()), 
        thrust::raw_pointer_cast(firstMomentElectron.data()), 
        thrust::raw_pointer_cast(U.data()),  
        thrust::raw_pointer_cast(reloadParticlesDataIon.data()), 
        thrust::raw_pointer_cast(reloadParticlesDataElectron.data()), 
        indexOfInterfaceStartInMHD, 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        mPIInfoPIC.localNx, mPIInfoPIC.localNy, mPIInfoPIC.buffer, 
        mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY, 
        mPIInfoMHD.localSizeX, mPIInfoMHD.localSizeY, 
        interfaceSizeX, interfaceSizeY
    );
    cudaDeviceSynchronize();
    
    deleteParticles(particlesIon, particlesElectron, step);

    
    host_reloadParticlesDataIon = reloadParticlesDataIon;
    host_reloadParticlesDataElectron = reloadParticlesDataElectron;

    for (int i = 0; i < mPIInfoPIC.localNx; i++) {
        for (int j = 0; j < interfaceLength; j++) {
            int index;
            index = j + i * interfaceLength;
            host_reloadParticlesDataIon[index + 1].numberAndIndex += host_reloadParticlesDataIon[index].numberAndIndex;
            host_reloadParticlesDataElectron[index + 1].numberAndIndex += host_reloadParticlesDataElectron[index].numberAndIndex;
        }
    }

    reloadParticlesDataIon = host_reloadParticlesDataIon;
    reloadParticlesDataElectron = host_reloadParticlesDataElectron;


    std::mt19937 genIon(step), genElectron(step + 1);
    std::uniform_int_distribution<unsigned long long> distIon(0, Interface2DConst::reloadParticlesTotalNum);
    std::uniform_int_distribution<unsigned long long> distElectron(0, Interface2DConst::reloadParticlesTotalNum);
    restartParticlesIndexIon = distIon(genIon);
    restartParticlesIndexElectron = distElectron(genElectron); 

    reloadParticles_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()), 
        thrust::raw_pointer_cast(reloadParticlesDataIon.data()), 
        thrust::raw_pointer_cast(reloadParticlesSourceIon.data()), 
        Interface2DConst::reloadParticlesTotalNum,   
        thrust::raw_pointer_cast(particlesIon.data()), 
        restartParticlesIndexIon, 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        mPIInfoPIC.existNumIonPerProcs, 
        step, 
        mPIInfoPIC.xminForProcs, mPIInfoPIC.xmaxForProcs, 
        mPIInfoPIC.yminForProcs, mPIInfoPIC.ymaxForProcs, 
        mPIInfoPIC.localNx, mPIInfoPIC.localNy, mPIInfoPIC.buffer, 
        mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY, 
        interfaceSizeX, interfaceSizeY
    );
    cudaDeviceSynchronize();

    reloadParticles_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()), 
        thrust::raw_pointer_cast(reloadParticlesDataElectron.data()), 
        thrust::raw_pointer_cast(reloadParticlesSourceElectron.data()), 
        Interface2DConst::reloadParticlesTotalNum, 
        thrust::raw_pointer_cast(particlesElectron.data()), 
        restartParticlesIndexElectron, 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        mPIInfoPIC.existNumElectronPerProcs, 
        step, 
        mPIInfoPIC.xminForProcs, mPIInfoPIC.xmaxForProcs, 
        mPIInfoPIC.yminForProcs, mPIInfoPIC.ymaxForProcs, 
        mPIInfoPIC.localNx, mPIInfoPIC.localNy, mPIInfoPIC.buffer, 
        mPIInfoPIC.localSizeX, mPIInfoPIC.localSizeY, 
        interfaceSizeX, interfaceSizeY
    );
    cudaDeviceSynchronize();


    auto partitionEndIon = thrust::partition(
        particlesIon.begin(), particlesIon.end(), 
        [] __device__ (const Particle& p) { return p.isExist; }
    );
    cudaDeviceSynchronize();

    auto partitionEndElectron = thrust::partition(
        particlesElectron.begin(), particlesElectron.end(), 
        [] __device__ (const Particle& p) { return p.isExist; }
    );
    cudaDeviceSynchronize();

    mPIInfoPIC.existNumIonPerProcs = thrust::distance(particlesIon.begin(), partitionEndIon);
    mPIInfoPIC.existNumElectronPerProcs = thrust::distance(particlesElectron.begin(), partitionEndElectron);
}


