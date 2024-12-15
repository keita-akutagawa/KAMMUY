#include "interface.hpp"


__global__ void sendMHDtoPIC_magneticField_y_kernel(
    const double* interlockingFunctionY, 
    const double* interlockingFunctionYHalf, 
    const ConservationParameter* U, 
    MagneticField* B, 
    int indexOfInterfaceStartInMHD, 
    int indexOfInterfaceStartInPIC, 
    int localSizeXPIC, int localSizeYPIC, 
    int localSizeXMHD, int localSizeYMHD, 
    int localSizeXInterface, int localSizeYInterface
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < localSizeXInterface - 1 && 0 < j && j < localSizeYInterface - 1) {
        double bXPIC, bYPIC, bZPIC;
        double bXMHD, bYMHD, bZMHD;
        double bXInterface, bYInterface, bZInterface;

        int indexPIC = indexOfInterfaceStartInPIC + j + i * localSizeYPIC;
        int indexMHD = indexOfInterfaceStartInMHD + j + i * localSizeYMHD;

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

void Interface2D::sendMHDtoPIC_magneticField_y(
    const thrust::device_vector<ConservationParameter>& U, 
    thrust::device_vector<MagneticField>& B
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((localSizeXInterface + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (localSizeYInterface + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendMHDtoPIC_magneticField_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()),
        thrust::raw_pointer_cast(interlockingFunctionYHalf.data()),
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(B.data()), 
        indexOfInterfaceStartInMHD, 
        indexOfInterfaceStartInPIC, 
        localSizeXPIC, localSizeYPIC,  
        localSizeXMHD, localSizeYMHD, 
        localSizeXInterface, localSizeYInterface
    );
    cudaDeviceSynchronize();
}


__global__ void sendMHDtoPIC_electricField_y_kernel(
    const double* interlockingFunctionY, 
    const double* interlockingFunctionYHalf, 
    const ConservationParameter* U, 
    ElectricField* E, 
    int indexOfInterfaceStartInMHD, 
    int indexOfInterfaceStartInPIC, 
    int localSizeXPIC, int localSizeYPIC, 
    int localSizeXMHD, int localSizeYMHD, 
    int localSizeXInterface, int localSizeYInterface
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < localSizeXInterface - 1 && 0 < j && j < localSizeYInterface - 1) {
        double eXPIC, eYPIC, eZPIC;
        double eXMHD, eYMHD, eZMHD;
        double eXPlusX1MHD;
        double eYPlusY1MHD;
        double rho, u, v, w;
        double bXMHD, bYMHD, bZMHD;
        double eXInterface, eYInterface, eZInterface;
        double mIon = PIC2DConst::device_mIon, mElectron = PIC2DConst::device_mElectron; 

        int indexPIC = indexOfInterfaceStartInPIC + j + i * localSizeYPIC;
        int indexMHD = indexOfInterfaceStartInMHD + j + i * localSizeYMHD;

        //PICのグリッドにMHDを合わせる
        eXPIC = E[indexPIC].eX;
        eYPIC = E[indexPIC].eY;
        eZPIC = E[indexPIC].eZ;

        rho = max(U[indexMHD].rho, mIon * 1 + mElectron * 1);
        u = U[indexMHD].rhoU / (rho + IdealMHD2DConst::device_EPS);
        v = U[indexMHD].rhoV / (rho + IdealMHD2DConst::device_EPS);
        w = U[indexMHD].rhoW / (rho + IdealMHD2DConst::device_EPS); 
        bXMHD = 0.5 * (U[indexMHD].bX + U[indexMHD - localSizeYMHD].bX);
        bYMHD = 0.5 * (U[indexMHD].bY + U[indexMHD - 1].bY);
        bZMHD = U[indexMHD].bZ;
        eXMHD = -(v * bZMHD - w * bYMHD);
        eYMHD = -(w * bXMHD - u * bZMHD);
        eZMHD = -(u * bYMHD - v * bXMHD);

        rho = max(U[indexMHD + localSizeYMHD].rho, mIon * 1 + mElectron * 1);
        u = U[indexMHD + localSizeYMHD].rhoU / (rho + IdealMHD2DConst::device_EPS);
        v = U[indexMHD + localSizeYMHD].rhoV / (rho + IdealMHD2DConst::device_EPS);
        w = U[indexMHD + localSizeYMHD].rhoW / (rho + IdealMHD2DConst::device_EPS); 
        bXMHD = 0.5 * (U[indexMHD + localSizeYMHD].bX + U[indexMHD].bX);
        bYMHD = 0.5 * (U[indexMHD + localSizeYMHD].bY + U[indexMHD - 1 + localSizeYMHD].bY);
        bZMHD = U[indexMHD + localSizeYMHD].bZ;
        eXPlusX1MHD = -(v * bZMHD - w * bYMHD);

        rho = max(U[indexMHD + 1].rho, mIon * 1 + mElectron * 1);
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

void Interface2D::sendMHDtoPIC_electricField_y(
    const thrust::device_vector<ConservationParameter>& U, 
    thrust::device_vector<ElectricField>& E
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((localSizeXInterface + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (localSizeYInterface + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendMHDtoPIC_electricField_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()), 
        thrust::raw_pointer_cast(interlockingFunctionYHalf.data()), 
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(E.data()), 
        indexOfInterfaceStartInMHD, 
        indexOfInterfaceStartInPIC, 
        localSizeXPIC, localSizeYPIC,  
        localSizeXMHD, localSizeYMHD, 
        localSizeXInterface, localSizeYInterface
    );
    cudaDeviceSynchronize();
}


__global__ void sendMHDtoPIC_currentField_y_kernel(
    const double* interlockingFunctionY, 
    const double* interlockingFunctionYHalf, 
    const ConservationParameter* U, 
    CurrentField* current, 
    int indexOfInterfaceStartInMHD, 
    int indexOfInterfaceStartInPIC, 
    int localSizeXPIC, int localSizeYPIC, 
    int localSizeXMHD, int localSizeYMHD, 
    int localSizeXInterface, int localSizeYInterface
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < localSizeXInterface - 1 && 0 < j && j < localSizeYInterface - 1) {
        double jXPIC, jYPIC, jZPIC;
        double jXMHD, jYMHD, jZMHD;
        double jXPlusX1MHD; 
        double jYPlusY1MHD; 
        double jXInterface, jYInterface, jZInterface;
        double dx = IdealMHD2DConst::device_dx, dy = IdealMHD2DConst::device_dy;

        int indexPIC = indexOfInterfaceStartInPIC + j + i * localSizeYPIC;
        int indexMHD = indexOfInterfaceStartInMHD + j + i * localSizeYMHD;

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

void Interface2D::sendMHDtoPIC_currentField_y(
    const thrust::device_vector<ConservationParameter>& U, 
    thrust::device_vector<CurrentField>& current
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((localSizeXInterface + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (localSizeYInterface + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendMHDtoPIC_currentField_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()), 
        thrust::raw_pointer_cast(interlockingFunctionYHalf.data()), 
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(current.data()), 
        indexOfInterfaceStartInMHD, 
        indexOfInterfaceStartInPIC, 
        localSizeXPIC, localSizeYPIC,  
        localSizeXMHD, localSizeYMHD, 
        localSizeXInterface, localSizeYInterface
    );
    cudaDeviceSynchronize();
}


__global__ void sendMHDtoPIC_particle_y_kernel(
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
    int localSizeXPIC, int localSizeYPIC, 
    int localSizeXMHD, int localSizeYMHD, 
    int localSizeXInterface, int localSizeYInterface
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < localSizeXInterface - 1 && 0 < j && j < localSizeYInterface - 1) {
        int indexForReload = j + i * localSizeYInterface;  
        int indexPIC = indexOfInterfaceStartInPIC + j + i * localSizeYPIC;
        int indexMHD = indexOfInterfaceStartInMHD + j + i * localSizeYMHD;
        double rhoMHD, uMHD, vMHD, wMHD, bXMHD, bYMHD, bZMHD, eMHD, pMHD;
        double jXMHD, jYMHD, jZMHD, niMHD, neMHD, tiMHD, teMHD;
        double rhoPIC, uPIC, vPIC, wPIC;
        double jXPIC, jYPIC, jZPIC, niPIC, nePIC, vThiPIC, vThePIC;
        double dx = IdealMHD2DConst::device_dx, dy = IdealMHD2DConst::device_dy;
        double mIon = PIC2DConst::device_mIon, mElectron = PIC2DConst::device_mElectron;
        double qIon = PIC2DConst::device_qIon, qElectron = PIC2DConst::device_qElectron;

        //整数格子点上で計算する。リロードに使う。
        rhoMHD = max(U[indexMHD].rho, mIon * 1 + mElectron * 1);
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

        rhoPIC =  max(mIon * zerothMomentIon[indexPIC].n + mElectron * zerothMomentElectron[indexPIC].n, mIon * 1 + mElectron * 1);
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


        reloadParticlesDataIon     [indexForReload].numAndIndex = max(static_cast<unsigned long long>(round(niPIC)), static_cast<unsigned long long>(1));
        reloadParticlesDataElectron[indexForReload].numAndIndex = max(static_cast<unsigned long long>(round(nePIC)), static_cast<unsigned long long>(1));
        reloadParticlesDataIon     [indexForReload].u              = uPIC;
        reloadParticlesDataIon     [indexForReload].v              = vPIC;
        reloadParticlesDataIon     [indexForReload].w              = wPIC;
        reloadParticlesDataElectron[indexForReload].u              = uPIC - jXPIC / max(round(nePIC), 1.0) / abs(qElectron);
        reloadParticlesDataElectron[indexForReload].v              = vPIC - jYPIC / max(round(nePIC), 1.0) / abs(qElectron);
        reloadParticlesDataElectron[indexForReload].w              = wPIC - jZPIC / max(round(nePIC), 1.0) / abs(qElectron);
        reloadParticlesDataIon     [indexForReload].vth            = vThiPIC;
        reloadParticlesDataElectron[indexForReload].vth            = vThePIC;

        if (j == 1) {
            reloadParticlesDataIon[indexForReload - 1]      = reloadParticlesDataIon[indexForReload];
            reloadParticlesDataElectron[indexForReload - 1] = reloadParticlesDataElectron[indexForReload];
        }
        if (j == localSizeYInterface - 2) {
            reloadParticlesDataIon[indexForReload + 1]      = reloadParticlesDataIon[indexForReload];
            reloadParticlesDataElectron[indexForReload + 1] = reloadParticlesDataElectron[indexForReload];
        }
    }
}


__global__ void deleteParticles_kernel(
    const double* interlockingFunctionY, 
    Particle* particlesSpecies, 
    const int indexOfInterfaceStartInPIC, 
    const unsigned long long existNumSpecies, 
    int seed, 
    const float xminForProcs, const float xmaxForProcs, 
    const int buffer, 
    int localSizeXInterface, int localSizeYInterface
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        float x = particlesSpecies[i].x;
        float y = particlesSpecies[i].y;
        float deleteXMin = xminForProcs - buffer * PIC2DConst::device_dx;
        float deleteXMax = xmaxForProcs + buffer * PIC2DConst::device_dx;
        float deleteYMin = indexOfInterfaceStartInPIC * PIC2DConst::device_dy;
        float deleteYMax = (indexOfInterfaceStartInPIC + localSizeYInterface) * PIC2DConst::device_dy;

        if (deleteXMin < x && x < deleteXMax && deleteYMin < y && y < deleteYMax) {
            int j = floorf(y - deleteYMin);
            curandState state; 
            curand_init(seed, i, 0, &state);
            float randomValue = curand_uniform(&state);
            if (randomValue < interlockingFunctionY[j]) {
                particlesSpecies[i].isExist = false;
            }
        }
    }
}


void Interface2D::deleteParticlesSpecies(
    thrust::device_vector<Particle>& particlesSpecies, 
    unsigned long long& existNumSpeciesPerProcs, 
    int seed
)
{

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((existNumSpeciesPerProcs + threadsPerBlock.x - 1) / threadsPerBlock.x);
    
    deleteParticles_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()), 
        thrust::raw_pointer_cast(particlesSpecies.data()),
        indexOfInterfaceStartInPIC, 
        existNumSpeciesPerProcs, 
        seed, 
        mPIInfoPIC.xminForProcs, mPIInfoPIC.xmaxForProcs, 
        mPIInfoPIC.buffer, 
        localSizeXInterface, localSizeYInterface
    );
    cudaDeviceSynchronize();

    auto partitionEnd = thrust::partition(
        particlesSpecies.begin(), particlesSpecies.end(), 
        [] __device__ (const Particle& p) { return p.isExist; }
    );
    cudaDeviceSynchronize();

    existNumSpeciesPerProcs = thrust::distance(particlesSpecies.begin(), partitionEnd);
}


__global__ void reloadParticlesSpecies_kernel(
    const double* interlockingFunctionY, 
    const ReloadParticlesData* reloadParticlesDataSpecies, 
    const Particle* reloadParticlesSpecies, 
    unsigned long long reloadParticlesTotalNumSpecies, 
    Particle* particlesSpecies, 
    unsigned long long restartParticlesIndexSpecies, 
    int indexOfInterfaceStartInPIC, 
    unsigned long long existNumSpecies, 
    int step, 
    const float xminForProcs, const float xmaxForProcs, 
    const float yminForProcs, const float ymaxForProcs, 
    int buffer, 
    int localSizeXInterface, int localSizeYInterface
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localSizeXInterface && j < localSizeYInterface) {
        int index = j + i * localSizeYInterface; 
        float u = reloadParticlesDataSpecies[index].u;
        float v = reloadParticlesDataSpecies[index].v;
        float w = reloadParticlesDataSpecies[index].w;
        float vth = reloadParticlesDataSpecies[index].vth;
        Particle particleSource, particleReload;
        float x, y, z, vx, vy, vz, gamma;

        for (unsigned long long k = reloadParticlesDataSpecies[index].numAndIndex; k < reloadParticlesDataSpecies[index + 1].numAndIndex; k++) {
            curandState state; 
            curand_init(step, k, 0, &state);
            float randomValue = curand_uniform(&state);

            if (randomValue < interlockingFunctionY[j]) {
                particleSource = reloadParticlesSpecies[(restartParticlesIndexSpecies + k) % reloadParticlesTotalNumSpecies];

                x = particleSource.x; x += i * PIC2DConst::device_dx + (xminForProcs - buffer * PIC2DConst::device_dx);
                y = particleSource.y; y += (indexOfInterfaceStartInPIC + j) * PIC2DConst::device_dy;
                z = particleSource.z;
                vx = particleSource.vx; vx = u + vx * vth;
                vy = particleSource.vy; vy = v + vy * vth;
                vz = particleSource.vz; vz = w + vz * vth;
                if (1.0f - (vx * vx + vy * vy + vz * vz) / pow(PIC2DConst::device_c, 2) < 0.0f){
                    printf("particle exceeds light speed... ");
                    continue; //delete if particle speed exceeds light speed c. 
                };
                gamma = 1.0f / sqrt(1.0f - (vx * vx + vy * vy + vz * vz) / pow(PIC2DConst::device_c, 2));

                particleReload.x = x; particleReload.y = y; particleReload.z = z;
                particleReload.vx = vx * gamma; particleReload.vy = vy * gamma, particleReload.vz = vz * gamma; 
                particleReload.gamma = gamma;
                particleReload.isExist = true;

                particlesSpecies[existNumSpecies + k] = particleReload;
            } 
        }
    }
}


void Interface2D::reloadParticlesSpecies(
    thrust::device_vector<Particle>& particlesSpecies, 
    thrust::device_vector<ReloadParticlesData>& reloadParticlesDataSpecies, 
    thrust::device_vector<Particle>& reloadParticlesSourceSpecies, 
    unsigned long long& existNumSpeciesPerProcs, 
    int seed 
)
{
    std::mt19937 genSpecies(seed);
    std::uniform_int_distribution<unsigned long long> distSpecies(0, Interface2DConst::reloadParticlesTotalNum);
    unsigned long long restartParticlesIndexSpecies = distSpecies(genSpecies);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((localSizeXInterface + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (localSizeYInterface + threadsPerBlock.y - 1) / threadsPerBlock.y);

    reloadParticlesSpecies_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()), 
        thrust::raw_pointer_cast(reloadParticlesDataSpecies.data()), 
        thrust::raw_pointer_cast(reloadParticlesSourceSpecies.data()), 
        Interface2DConst::reloadParticlesTotalNum,   
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        restartParticlesIndexSpecies, 
        indexOfInterfaceStartInPIC, 
        existNumSpeciesPerProcs, 
        seed, 
        mPIInfoPIC.xminForProcs, mPIInfoPIC.xmaxForProcs, 
        mPIInfoPIC.yminForProcs, mPIInfoPIC.ymaxForProcs, 
        mPIInfoPIC.buffer, 
        localSizeXInterface, localSizeYInterface
    );
    cudaDeviceSynchronize();

    auto partitionEnd = thrust::partition(
        particlesSpecies.begin(), particlesSpecies.end(), 
        [] __device__ (const Particle& p) { return p.isExist; }
    );

    existNumSpeciesPerProcs = thrust::distance(particlesSpecies.begin(), partitionEnd);
}


void Interface2D::sendMHDtoPIC_particle(
    const thrust::device_vector<ConservationParameter>& U, 
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron, 
    int seed
)
{
    setMoments(particlesIon, particlesElectron); 

    for (int count = 0; count < Interface2DConst::convolutionCount; count++) {
        interfaceNoiseRemover2D.convolveMoments(
            zerothMomentIon, zerothMomentElectron, 
            firstMomentIon, firstMomentElectron
        );

        PIC2DMPI::sendrecv_field_x(zerothMomentIon, mPIInfoPIC, mPIInfoPIC.mpi_zerothMomentType);
        PIC2DMPI::sendrecv_field_x(zerothMomentElectron, mPIInfoPIC, mPIInfoPIC.mpi_zerothMomentType);
        PIC2DMPI::sendrecv_field_x(firstMomentIon, mPIInfoPIC, mPIInfoPIC.mpi_firstMomentType);
        PIC2DMPI::sendrecv_field_x(firstMomentElectron, mPIInfoPIC, mPIInfoPIC.mpi_firstMomentType);
    }

    thrust::fill(reloadParticlesDataIon.begin(), reloadParticlesDataIon.end(), ReloadParticlesData());
    thrust::fill(reloadParticlesDataElectron.begin(), reloadParticlesDataElectron.end(), ReloadParticlesData());

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((localSizeXInterface + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (localSizeYInterface + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendMHDtoPIC_particle_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(
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
        localSizeXPIC, localSizeYPIC,  
        localSizeXMHD, localSizeYMHD, 
        localSizeXInterface, localSizeYInterface
    );
    cudaDeviceSynchronize();

    Interface2DMPI::sendrecv_reloadParticlesData_x(reloadParticlesDataIon, mPIInfoInterface);
    Interface2DMPI::sendrecv_reloadParticlesData_x(reloadParticlesDataElectron, mPIInfoInterface);
    
    deleteParticlesSpecies(
        particlesIon, mPIInfoPIC.existNumIonPerProcs, seed
    );
    deleteParticlesSpecies(
        particlesElectron, mPIInfoPIC.existNumElectronPerProcs, seed + 1
    );
    
    host_reloadParticlesDataIon = reloadParticlesDataIon;
    host_reloadParticlesDataElectron = reloadParticlesDataElectron;
    
    for (int i = 0; i < localSizeXInterface; i++) {
        for (int j = 0; j < localSizeYInterface; j++) {
            int index;
            index = j + i * localSizeYInterface;
            host_reloadParticlesDataIon[index + 1].numAndIndex += host_reloadParticlesDataIon[index].numAndIndex;
            host_reloadParticlesDataElectron[index + 1].numAndIndex += host_reloadParticlesDataElectron[index].numAndIndex;
        }
    }
    reloadParticlesDataIon = host_reloadParticlesDataIon;
    reloadParticlesDataElectron = host_reloadParticlesDataElectron;

    reloadParticlesSpecies(
        particlesIon, reloadParticlesDataIon, reloadParticlesSourceIon, 
        mPIInfoPIC.existNumIonPerProcs, seed
    ); 
    reloadParticlesSpecies(
        particlesElectron, reloadParticlesDataElectron, reloadParticlesSourceElectron, 
        mPIInfoPIC.existNumElectronPerProcs, seed + 1
    ); 
}


