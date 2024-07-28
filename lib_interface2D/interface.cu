#include "interface.hpp"
#include <cmath>
#include <curand_kernel.h>


__global__ void initializeReloadParticlesSource_kernel(
    Particle* reloadParticlesSourceSpecies, 
    int reloadParticlesNumSpecies, 
    int seed
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < reloadParticlesNumSpecies) {
        curandState stateX; 
        curandState stateY;
        curandState stateVx; 
        curandState stateVy; 
        curandState stateVz;  
        curand_init(seed, i, 0, &stateX);
        curand_init(seed + 1, i, 0, &stateY);
        curand_init(seed + 2, i, 0, &stateVx);
        curand_init(seed + 3, i, 0, &stateVy);
        curand_init(seed + 4, i, 0, &stateVz);

        reloadParticlesSourceSpecies[i].x = curand_uniform(&stateX);
        reloadParticlesSourceSpecies[i].y = curand_uniform(&stateY);
        reloadParticlesSourceSpecies[i].vx = curand_normal(&stateVx);
        reloadParticlesSourceSpecies[i].vy = curand_normal(&stateVy);
        reloadParticlesSourceSpecies[i].vz = curand_normal(&stateVz);
    }
}

Interface2D::Interface2D(
    int indexStartMHD, 
    int indexStartPIC
)
    :  indexOfInterfaceStartInMHD(indexStartMHD), 
       indexOfInterfaceStartInPIC(indexStartPIC), 
       interlockingFunctionY(Interface2DConst::interfaceLength), 
       interlockingFunctionYHalf(Interface2DConst::interfaceLength - 1),
       host_interlockingFunctionY(Interface2DConst::interfaceLength), 
       host_interlockingFunctionYHalf(Interface2DConst::interfaceLength - 1),

       zerothMomentIon(PIC2DConst::nx * PIC2DConst::ny), 
       zerothMomentElectron(PIC2DConst::nx * PIC2DConst::ny), 
       firstMomentIon(PIC2DConst::nx * PIC2DConst::ny), 
       firstMomentElectron(PIC2DConst::nx * PIC2DConst::ny),

       reloadParticlesNumIon(0),
       reloadParticlesNumElectron(0), 
       restartParticlesIndexIon(0), 
       restartParticlesIndexElectron(0), 

       reloadParticlesDataIon(PIC2DConst::nx * Interface2DConst::interfaceLength), 
       reloadParticlesDataElectron(PIC2DConst::nx * Interface2DConst::interfaceLength), 

       reloadParticlesSourceIon(Interface2DConst::reloadParticlesTotalNumIon), 
       reloadParticlesSourceElectron(Interface2DConst::reloadParticlesTotalNumElectron), 

       reloadParticlesIndexIon(PIC2DConst::nx * Interface2DConst::interfaceLength), 
       reloadParticlesIndexElectron(PIC2DConst::nx * Interface2DConst::interfaceLength), 
       host_reloadParticlesIndexIon(PIC2DConst::nx * Interface2DConst::interfaceLength), 
       host_reloadParticlesIndexElectron(PIC2DConst::nx * Interface2DConst::interfaceLength)
{
    indexOfInterfaceEndInMHD = indexOfInterfaceStartInMHD + Interface2DConst::interfaceLength;
    indexOfInterfaceEndInPIC = indexOfInterfaceStartInPIC + Interface2DConst::interfaceLength;

    for(int i = 0; Interface2DConst::interfaceLength; i++) {
        host_interlockingFunctionY[i] = 0.5f * (
            1.0f + cos(Interface2DConst::PI  * (i - 0.0f) / (Interface2DConst::interfaceLength - 0.0f))
        );
    }
    for(int i = 0; Interface2DConst::interfaceLength - 1; i++) {
        host_interlockingFunctionY[i] = 0.5f * (
            1.0f + cos(Interface2DConst::PI  * (i + 0.5f - 0.0f) / (Interface2DConst::interfaceLength - 0.0f))
        );
    }

    interlockingFunctionY = host_interlockingFunctionY;
    interlockingFunctionYHalf = host_interlockingFunctionYHalf;


    dim3 threadsPerBlockForIon(256);
    dim3 blocksPerGridForIon((Interface2DConst::reloadParticlesTotalNumIon + threadsPerBlockForIon.x - 1) / threadsPerBlockForIon.x);

    initializeReloadParticlesSource_kernel<<<blocksPerGridForIon, threadsPerBlockForIon>>>(
        thrust::raw_pointer_cast(reloadParticlesSourceIon.data()),
        Interface2DConst::reloadParticlesTotalNumIon, 
        10000
    );

    cudaDeviceSynchronize();

    dim3 threadsPerBlockForElectron(256);
    dim3 blocksPerGridForElectron((Interface2DConst::reloadParticlesTotalNumElectron + threadsPerBlockForElectron.x - 1) / threadsPerBlockForElectron.x);

    initializeReloadParticlesSource_kernel<<<blocksPerGridForElectron, threadsPerBlockForElectron>>>(
        thrust::raw_pointer_cast(reloadParticlesSourceElectron.data()),
        Interface2DConst::reloadParticlesTotalNumElectron, 
        20000
    );

    cudaDeviceSynchronize();
}


__global__ void sendMHDtoPIC_magneticField_yDirection_kernel(
    const float* interlockingFunctionY, 
    const float* interlockingFunctionYHalf, 
    const ConservationParameter* U, 
    MagneticField* B, 
    int indexOfInterfaceStartInMHD, 
    int indexOfInterfaceStartInPIC
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < PIC2DConst::device_nx && 0 < j && j < Interface2DConst::interfaceLength - 1) {
        float bXPIC, bYPIC, bZPIC;
        float bXMHD, bYMHD, bZMHD;
        float bXInterface, bYInterface, bZInterface;

        int indexPIC = indexOfInterfaceStartInPIC +  j + i * PIC2DConst::device_nx;
        int indexMHD = indexOfInterfaceStartInMHD +  j + i * IdealMHD2DConst::device_nx;

        //PICのグリッドにMHDを合わせる
        bXPIC = B[indexPIC].bX;
        bYPIC = B[indexPIC].bY;
        bZPIC = B[indexPIC].bZ;
        bXMHD = 0.25f * (U[indexMHD].bX + U[indexMHD - IdealMHD2DConst::device_nx].bX + U[indexMHD + 1].bX + U[indexMHD + 1 - IdealMHD2DConst::device_nx].bX);
        bYMHD = 0.25f * (U[indexMHD].bY + U[indexMHD + IdealMHD2DConst::device_nx].bY + U[indexMHD - 1].bY + U[indexMHD - 1 + IdealMHD2DConst::device_nx].bY);
        bZMHD = 0.25f * (U[indexMHD].bZ + U[indexMHD + IdealMHD2DConst::device_nx].bZ + U[indexMHD + 1].bZ + U[indexMHD + 1 + IdealMHD2DConst::device_nx].bZ);

        bXInterface = interlockingFunctionYHalf[j] * bXMHD + (1.0f - interlockingFunctionYHalf[j]) * bXPIC;
        bYInterface = interlockingFunctionY[j]     * bYMHD + (1.0f - interlockingFunctionY[j])     * bYPIC;
        bZInterface = interlockingFunctionYHalf[j] * bZMHD + (1.0f - interlockingFunctionYHalf[j]) * bZPIC;
        
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
    dim3 blocksPerGrid((PIC2DConst::nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (Interface2DConst::interfaceLength + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendMHDtoPIC_magneticField_yDirection_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()),
        thrust::raw_pointer_cast(interlockingFunctionYHalf.data()),
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(B.data()), 
        indexOfInterfaceStartInMHD, 
        indexOfInterfaceStartInPIC 
    );

    cudaDeviceSynchronize();
}


__global__ void sendMHDtoPIC_electricField_yDirection_kernel(
    const float* interlockingFunctionY, 
    const float* interlockingFunctionYHalf, 
    const ConservationParameter* U, 
    ElectricField* E, 
    int indexOfInterfaceStartInMHD, 
    int indexOfInterfaceStartInPIC
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < PIC2DConst::device_nx && 0 < j &&  j < Interface2DConst::interfaceLength - 1) {
        float eXPIC, eYPIC, eZPIC;
        float eXMHD, eYMHD, eZMHD;
        float eXPlusX1MHD;
        float eYPlusY1MHD;
        float eXMHD, eYMHD, eZMHD;
        float rho, u, v, w;
        float bXMHD, bYMHD, bZMHD;
        float eXInterface, eYInterface, eZInterface;

        int indexPIC = indexOfInterfaceStartInPIC +  j + i * PIC2DConst::device_nx;
        int indexMHD = indexOfInterfaceStartInMHD +  j + i * IdealMHD2DConst::device_nx;

        //PICのグリッドにMHDを合わせる
        eXPIC = E[indexPIC].eX;
        eYPIC = E[indexPIC].eY;
        eZPIC = E[indexPIC].eZ;

        rho = U[indexMHD].rho;
        u = U[indexMHD].rhoU / rho;
        v = U[indexMHD].rhoV / rho;
        w = U[indexMHD].rhoW / rho; 
        bXMHD = 0.5f * (U[indexMHD].bX + U[indexMHD - IdealMHD2DConst::device_nx].bX);
        bYMHD = 0.5f * (U[indexMHD].bY + U[indexMHD - 1].bY);
        bZMHD = U[indexMHD].bZ;
        eXMHD = -(v * bZMHD - w * bYMHD);
        eYMHD = -(w * bXMHD - u * bZMHD);
        eZMHD = -(u * bYMHD - v * bXMHD);

        rho = U[indexMHD + IdealMHD2DConst::device_nx].rho;
        u = U[indexMHD + IdealMHD2DConst::device_nx].rhoU / rho;
        v = U[indexMHD + IdealMHD2DConst::device_nx].rhoV / rho;
        w = U[indexMHD + IdealMHD2DConst::device_nx].rhoW / rho; 
        bXMHD = 0.5f * (U[indexMHD + IdealMHD2DConst::device_nx].bX + U[indexMHD].bX);
        bYMHD = 0.5f * (U[indexMHD + IdealMHD2DConst::device_nx].bY + U[indexMHD - 1 + IdealMHD2DConst::device_nx].bY);
        bZMHD = U[indexMHD + IdealMHD2DConst::device_nx].bZ;
        eXPlusX1MHD = -(v * bZMHD - w * bYMHD);

        rho = U[indexMHD + 1].rho;
        u = U[indexMHD + 1].rhoU / rho;
        v = U[indexMHD + 1].rhoV / rho;
        w = U[indexMHD + 1].rhoW / rho; 
        bXMHD = 0.5f * (U[indexMHD + 1].bX + U[indexMHD + 1 - IdealMHD2DConst::device_nx].bX);
        bYMHD = 0.5f * (U[indexMHD + 1].bY + U[indexMHD].bY);
        bZMHD = U[indexMHD + 1].bZ;
        eYPlusY1MHD = -(w * bXMHD - u * bZMHD);


        eXInterface = interlockingFunctionY[j]     * 0.5f * (eXMHD + eXPlusX1MHD) + (1.0f - interlockingFunctionY[j])     * eXPIC;
        eYInterface = interlockingFunctionYHalf[j] * 0.5f * (eYMHD + eYPlusY1MHD) + (1.0f - interlockingFunctionYHalf[j]) * eYPIC;
        eZInterface = interlockingFunctionY[j]     * eZMHD                        + (1.0f - interlockingFunctionY[j])     * eZPIC;
         
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
    dim3 blocksPerGrid((PIC2DConst::nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (Interface2DConst::interfaceLength + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendMHDtoPIC_electricField_yDirection_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()), 
        thrust::raw_pointer_cast(interlockingFunctionYHalf.data()), 
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(E.data()), 
        indexOfInterfaceStartInMHD, 
        indexOfInterfaceStartInPIC 
    );

    cudaDeviceSynchronize();
}


__global__ void sendMHDtoPIC_currentField_yDirection_kernel(
    const float* interlockingFunctionY, 
    const float* interlockingFunctionYHalf, 
    const ConservationParameter* U, 
    CurrentField* current, 
    int indexOfInterfaceStartInMHD, 
    int indexOfInterfaceStartInPIC
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < PIC2DConst::device_nx && 0 < j && j < Interface2DConst::interfaceLength - 1) {
        float jXPIC, jYPIC, jZPIC;
        float jXMHD, jYMHD, jZMHD;
        float jXPlusX1MHD; 
        float jYPlusY1MHD; 
        float jXInterface, jYInterface, jZInterface;
        int nx = IdealMHD2DConst::device_nx;
        float dx = IdealMHD2DConst::device_dx, dy = IdealMHD2DConst::device_dy;

        int indexPIC = indexOfInterfaceStartInPIC +  j + i * PIC2DConst::device_nx;
        int indexMHD = indexOfInterfaceStartInMHD +  j + i * IdealMHD2DConst::device_nx;

        //PICのグリッドにMHDを合わせる
        jXPIC = current[indexPIC].jX;
        jYPIC = current[indexPIC].jY;
        jZPIC = current[indexPIC].jZ;
        jXMHD = (U[indexMHD + 1].bZ - U[indexMHD - 1].bZ) / (2.0f * dy);
        jYMHD = -(U[indexMHD + nx].bZ - U[indexMHD - nx].bZ) / (2.0f * dx);
        jZMHD = 0.25f * ((U[indexMHD + nx].bY - U[indexMHD].bY) / dx - (U[indexMHD + 1].bX - U[indexMHD].bX) / dy 
                       + (U[indexMHD].bY - U[indexMHD - nx].bY) / dx - (U[indexMHD + 1 - nx].bX - U[indexMHD - nx].bX) / dy
                       + (U[indexMHD - 1 + nx].bY - U[indexMHD - 1].bY) / dx - (U[indexMHD].bX - U[indexMHD - 1].bX) / dy
                       + (U[indexMHD - 1].bY - U[indexMHD - 1 - nx].bY) / dx - (U[indexMHD - nx].bX - U[indexMHD - 1 - nx].bX) / dy);

        jXPlusX1MHD = (U[indexMHD + 2].bZ - U[indexMHD].bZ) / (2.0f * dy);
        jYPlusY1MHD = -(U[indexMHD + 2 * nx].bZ - U[indexMHD].bZ) / (2.0f * dx);

        jXInterface = interlockingFunctionY[j]     * 0.5f * (jXMHD + jXPlusX1MHD) + (1.0f - interlockingFunctionY[j])     * jXPIC;
        jYInterface = interlockingFunctionYHalf[j] * 0.5f * (jYMHD + jYPlusY1MHD) + (1.0f - interlockingFunctionYHalf[j]) * jYPIC;
        jZInterface = interlockingFunctionY[j]     * jZMHD                        + (1.0f - interlockingFunctionY[j])     * jZPIC;
        
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
    dim3 blocksPerGrid((PIC2DConst::nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (Interface2DConst::interfaceLength + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendMHDtoPIC_currentField_yDirection_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()), 
        thrust::raw_pointer_cast(interlockingFunctionYHalf.data()), 
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(current.data()), 
        indexOfInterfaceStartInMHD, 
        indexOfInterfaceStartInPIC 
    );

    cudaDeviceSynchronize();
}


__global__ void sendMHDtoPIC_particle_yDirection_kernel(
    const float* interlockingFunctionY, 
    const float* interlockingFunctionYHalf, 
    const ZerothMoment* zerothMomentIon, 
    const ZerothMoment* zerothMomentElectron, 
    const FirstMoment* firstMomentIon, 
    const FirstMoment* firstMomentElectron, 
    const ConservationParameter* U, 
    int reloadParticlesNumIon, int reloadParticlesNumElectron, 
    ReloadParticlesData* reloadParticlesDataIon, 
    ReloadParticlesData* reloadParticlesDataElectron, 
    int* reloadParticlesIndexIon, 
    int* reloadParticlesIndexElectron, 
    int indexOfInterfaceStartInMHD, 
    int indexOfInterfaceStartInPIC
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < PIC2DConst::device_nx && 0 < j && j < Interface2DConst::interfaceLength - 1) {
        int indexPIC = indexOfInterfaceStartInPIC + j + i * PIC2DConst::device_nx;
        int indexMHD = indexOfInterfaceStartInMHD + j + i * IdealMHD2DConst::device_nx;
        float rhoMHD, uMHD, vMHD, wMHD, bXMHD, bYMHD, bZMHD, eMHD, pMHD;
        float jXMHD, jYMHD, jZMHD, niMHD, neMHD, tiMHD, teMHD;
        float rhoPIC, uPIC, vPIC, wPIC;
        float jXPIC, jYPIC, jZPIC, niPIC, nePIC, vThiPIC, vThePIC;
        int nx = IdealMHD2DConst::device_nx;
        float dx = IdealMHD2DConst::device_dx, dy = IdealMHD2DConst::device_dy;

        rhoMHD = U[indexMHD].rho;
        uMHD = U[indexMHD].rhoU / rhoMHD;
        vMHD = U[indexMHD].rhoV / rhoMHD;
        wMHD = U[indexMHD].rhoW / rhoMHD;
        bXMHD = U[indexMHD].bX;
        bYMHD = U[indexMHD].bY;
        bZMHD = U[indexMHD].bZ;
        eMHD = U[indexMHD].e;
        pMHD = (IdealMHD2DConst::gamma_mhd - 1.0f)
             * (eMHD - 0.5f * rhoMHD * (uMHD * uMHD + vMHD * vMHD + wMHD * wMHD)
             - 0.5f * (bXMHD * bXMHD + bYMHD * bYMHD + bZMHD * bZMHD));
        jXMHD = (U[indexMHD + 1].bZ - U[indexMHD - 1].bZ) / (2.0f * dy);
        jYMHD = -(U[indexMHD + nx].bZ - U[indexMHD - nx].bZ) / (2.0f * dx);
        jZMHD = 0.25f * ((U[indexMHD + nx].bY - U[indexMHD].bY) / dx - (U[indexMHD + 1].bX - U[indexMHD].bX) / dy 
                       + (U[indexMHD].bY - U[indexMHD - nx].bY) / dx - (U[indexMHD + 1 - nx].bX - U[indexMHD - nx].bX) / dy
                       + (U[indexMHD - 1 + nx].bY - U[indexMHD - 1].bY) / dx - (U[indexMHD].bX - U[indexMHD - 1].bX) / dy
                       + (U[indexMHD - 1].bY - U[indexMHD - 1 - nx].bY) / dx - (U[indexMHD - nx].bX - U[indexMHD - 1 - nx].bX) / dy);

        niMHD = rhoMHD / (PIC2DConst::mIon + PIC2DConst::mElectron);
        neMHD = niMHD;
        tiMHD = pMHD / 2.0f / niMHD;
        teMHD = pMHD / 2.0f / neMHD;

        rhoPIC = PIC2DConst::mIon * zerothMomentIon[indexPIC].n + PIC2DConst::mElectron * zerothMomentElectron[indexPIC].n;
        uPIC = (PIC2DConst::mIon * firstMomentIon[indexPIC].x + PIC2DConst::mElectron * firstMomentElectron[indexPIC].x) / rhoPIC;
        vPIC = (PIC2DConst::mIon * firstMomentIon[indexPIC].y + PIC2DConst::mElectron * firstMomentElectron[indexPIC].y) / rhoPIC;
        wPIC = (PIC2DConst::mIon * firstMomentIon[indexPIC].z + PIC2DConst::mElectron * firstMomentElectron[indexPIC].z) / rhoPIC;
        jXPIC = PIC2DConst::qIon * firstMomentIon[indexPIC].x + PIC2DConst::qElectron * firstMomentElectron[indexPIC].x;
        jYPIC = PIC2DConst::qIon * firstMomentIon[indexPIC].y + PIC2DConst::qElectron * firstMomentElectron[indexPIC].y;
        jZPIC = PIC2DConst::qIon * firstMomentIon[indexPIC].z + PIC2DConst::qElectron * firstMomentElectron[indexPIC].z;


        rhoPIC = interlockingFunctionY[j] * rhoMHD + (1.0f - interlockingFunctionY[j]) * rhoPIC;
        uPIC = interlockingFunctionY[j] * uMHD + (1.0f - interlockingFunctionY[j]) * uPIC;
        vPIC = interlockingFunctionY[j] * vMHD + (1.0f - interlockingFunctionY[j]) * vPIC;
        wPIC = interlockingFunctionY[j] * wMHD + (1.0f - interlockingFunctionY[j]) * wPIC;
        jXPIC = interlockingFunctionY[j] * jXMHD + (1.0f - interlockingFunctionY[j]) * jXPIC;
        jYPIC = interlockingFunctionY[j] * jYMHD + (1.0f - interlockingFunctionY[j]) * jYPIC;
        jZPIC = interlockingFunctionY[j] * jZMHD + (1.0f - interlockingFunctionY[j]) * jZPIC;

        niPIC = rhoPIC / (PIC2DConst::mIon + PIC2DConst::mElectron);
        nePIC = niPIC;
        vThiPIC = sqrt(2.0f * tiMHD / PIC2DConst::mIon);
        vThePIC = sqrt(2.0f * teMHD / PIC2DConst::mElectron);

        atomicAdd(&reloadParticlesNumIon, round(niPIC));
        atomicAdd(&reloadParticlesNumElectron, round(nePIC));

        reloadParticlesDataIon[j + i * PIC2DConst::device_nx].number = round(niPIC);
        reloadParticlesDataElectron[j + i * PIC2DConst::device_nx].number = round(nePIC);
        reloadParticlesDataIon[j + i * PIC2DConst::device_nx].u = uPIC;
        reloadParticlesDataIon[j + i * PIC2DConst::device_nx].v = vPIC;
        reloadParticlesDataIon[j + i * PIC2DConst::device_nx].w = wPIC;
        reloadParticlesDataElectron[j + i * PIC2DConst::device_nx].u = uPIC - jXPIC / round(nePIC) / abs(PIC2DConst::qElectron);
        reloadParticlesDataElectron[j + i * PIC2DConst::device_nx].v = vPIC - jYPIC / round(nePIC) / abs(PIC2DConst::qElectron);
        reloadParticlesDataElectron[j + i * PIC2DConst::device_nx].w = wPIC - jZPIC / round(nePIC) / abs(PIC2DConst::qElectron);
        reloadParticlesDataIon[j + i * PIC2DConst::device_nx].vth = vThiPIC;
        reloadParticlesDataElectron[j + i * PIC2DConst::device_nx].vth = vThePIC;
        reloadParticlesDataIon[j + i * PIC2DConst::device_nx].hasData = true;
        reloadParticlesDataElectron[j + i * PIC2DConst::device_nx].hasData = true;

        reloadParticlesIndexIon[j + i * PIC2DConst::device_nx] = round(niPIC);
        reloadParticlesIndexElectron[j + i * PIC2DConst::device_nx] = round(nePIC);
    }
}



__global__ void reloadParticles_kernel(
    const ReloadParticlesData* reloadParticlesDataSpecies, 
    const int* reloadParticlesIndexSpecies, 
    const Particle* reloadParticlesSpecies, 
    Particle* particlesSpecies, 
    int reloadParticlesNumSpecies, 
    int restartParticlesIndexSpecies, 
    int indexOfInterfaceStartInPIC, 
    int existNumSpecies
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < PIC2DConst::device_nx && 0 < j && j < Interface2DConst::interfaceLength - 1) {
        int index = j + i * PIC2DConst::device_nx;
        int reloadNum = reloadParticlesDataSpecies[index].number;
        float u = reloadParticlesDataSpecies[index].u;
        float v = reloadParticlesDataSpecies[index].v;
        float w = reloadParticlesDataSpecies[index].w;
        float vth = reloadParticlesDataSpecies[index].vth;
        Particle particleSource, particleReload;
        float x, y, z, vx, vy, vz, gamma;

        for (int k = reloadParticlesIndexSpecies[index]; k < reloadParticlesIndexSpecies[index + 1]; k++) {
            particleSource = reloadParticlesSpecies[(restartParticlesIndexSpecies + k) % Interface2DConst::device_reloadParticlesTotalNumIon];

            x = particleSource.x; x += i * PIC2DConst::device_dx;
            y = particleSource.y; y += (indexOfInterfaceStartInPIC + j) * PIC2DConst::device_dy;
            z = 0.0f;
            vx = particleSource.vx; vx = u + vx * vth;
            vy = particleSource.vx; vy = v + vy * vth;
            vz = particleSource.vx; vz = w + vz * vth;
            gamma = sqrt(1.0f + (vx * vx + vy * vy + vz * vz) / (PIC2DConst::device_c * PIC2DConst::device_c));
            
            particleReload.x = x; particleReload.y = y; particleReload.z = z;
            particleReload.vx = vx; particleReload.vy = vy, particleReload.vz = vz; 
            particleReload.gamma = gamma;

            particlesSpecies[existNumSpecies + k] = particleReload;
        }

    }
}

void Interface2D::sendMHDtoPIC_particle(
    const thrust::device_vector<ConservationParameter>& U, 
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron
)
{
    setMoments(particlesIon, particlesElectron);

    reloadParticlesNumIon = 0;
    reloadParticlesNumElectron = 0;

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((PIC2DConst::nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (Interface2DConst::interfaceLength + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendMHDtoPIC_particle_yDirection_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()), 
        thrust::raw_pointer_cast(interlockingFunctionYHalf.data()), 
        thrust::raw_pointer_cast(zerothMomentIon.data()), 
        thrust::raw_pointer_cast(zerothMomentElectron.data()), 
        thrust::raw_pointer_cast(firstMomentIon.data()), 
        thrust::raw_pointer_cast(firstMomentElectron.data()), 
        thrust::raw_pointer_cast(U.data()), 
        reloadParticlesNumIon, reloadParticlesNumElectron, 
        thrust::raw_pointer_cast(reloadParticlesDataIon.data()), 
        thrust::raw_pointer_cast(reloadParticlesDataElectron.data()), 
        thrust::raw_pointer_cast(reloadParticlesIndexIon.data()), 
        thrust::raw_pointer_cast(reloadParticlesIndexElectron.data()), 
        indexOfInterfaceStartInMHD, 
        indexOfInterfaceStartInPIC
    );

    cudaDeviceSynchronize();


    host_reloadParticlesIndexIon = reloadParticlesIndexIon;
    host_reloadParticlesIndexElectron = reloadParticlesIndexElectron;

    for (int i = 0; i < PIC2DConst::nx; i++) {
        for (int j = 0; j < Interface2DConst::interfaceLength; j++) {

            if (j == 0 && i == 0) continue;

            host_reloadParticlesIndexIon[j + i * PIC2DConst::nx] += host_reloadParticlesIndexIon[j + i * PIC2DConst::nx - 1];
            host_reloadParticlesIndexElectron[j + i * PIC2DConst::nx] += host_reloadParticlesIndexElectron[j + i * PIC2DConst::nx - 1];
        }
    }

    reloadParticlesIndexIon = host_reloadParticlesIndexIon;
    reloadParticlesIndexElectron = host_reloadParticlesIndexElectron;


    reloadParticles_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(reloadParticlesDataIon.data()), 
        thrust::raw_pointer_cast(reloadParticlesIndexIon.data()), 
        thrust::raw_pointer_cast(reloadParticlesSourceIon.data()), 
        thrust::raw_pointer_cast(particlesIon.data()), 
        reloadParticlesNumIon, restartParticlesIndexIon, 
        indexOfInterfaceStartInPIC, 
        PIC2DConst::existNumIon
    );

    cudaDeviceSynchronize();

    reloadParticles_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(reloadParticlesDataElectron.data()), 
        thrust::raw_pointer_cast(reloadParticlesIndexElectron.data()), 
        thrust::raw_pointer_cast(reloadParticlesSourceElectron.data()), 
        thrust::raw_pointer_cast(particlesElectron.data()), 
        reloadParticlesNumElectron, restartParticlesIndexElectron, 
        indexOfInterfaceStartInPIC, 
        PIC2DConst::existNumElectron
    );

    cudaDeviceSynchronize();

    restartParticlesIndexIon = reloadParticlesNumIon % Interface2DConst::reloadParticlesTotalNumIon;
    restartParticlesIndexElectron = reloadParticlesNumElectron % Interface2DConst::reloadParticlesTotalNumElectron;
}


void Interface2D::setMoments(
    const thrust::device_vector<Particle>& particlesIon, 
    const thrust::device_vector<Particle>& particlesElectron
)
{
    momentCalculater.calculateZerothMomentOfOneSpecies(
        zerothMomentIon, particlesIon, PIC2DConst::totalNumIon
    );
    momentCalculater.calculateZerothMomentOfOneSpecies(
        zerothMomentElectron, particlesElectron, PIC2DConst::totalNumElectron
    );

    momentCalculater.calculateFirstMomentOfOneSpecies(
        firstMomentIon, particlesIon, PIC2DConst::totalNumIon
    );
    momentCalculater.calculateFirstMomentOfOneSpecies(
        firstMomentElectron, particlesElectron, PIC2DConst::totalNumElectron
    );
}


__global__ void deleteParticles_kernel(
    Particle* particlesSpecies, 
    int indexOfInterfaceStartInPIC, 
    unsigned long long totalNumSpecies
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < totalNumSpecies) {
        float y = particlesSpecies[i].y;
        float interfaceMin = indexOfInterfaceStartInPIC * PIC2DConst::device_dy;
        float interfaceMax = (indexOfInterfaceStartInPIC + Interface2DConst::device_interfaceLength) * PIC2DConst::device_dy;
        if (y >= interfaceMin + PIC2DConst::dy && y <= interfaceMax - PIC2DConst::dy) {
            particlesSpecies[i].isExist = false;
        }
    }
}

void Interface2D::deleteParticles(
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron
)
{
    dim3 threadsPerBlockForIon(256);
    dim3 blocksPerGridForIon((PIC2DConst::totalNumIon + threadsPerBlockForIon.x - 1) / threadsPerBlockForIon.x);

    initializeReloadParticlesSource_kernel<<<blocksPerGridForIon, threadsPerBlockForIon>>>(
        thrust::raw_pointer_cast(particlesIon.data()),
        indexOfInterfaceStartInPIC, 
        PIC2DConst::totalNumIon
    );

    cudaDeviceSynchronize();

    dim3 threadsPerBlockForElectron(256);
    dim3 blocksPerGridForElectron((PIC2DConst::totalNumElectron + threadsPerBlockForElectron.x - 1) / threadsPerBlockForElectron.x);

    initializeReloadParticlesSource_kernel<<<blocksPerGridForElectron, threadsPerBlockForElectron>>>(
        thrust::raw_pointer_cast(particlesElectron.data()),
        indexOfInterfaceStartInPIC, 
        PIC2DConst::totalNumElectron
    );

    cudaDeviceSynchronize();
}
