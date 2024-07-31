#include "interface.hpp"
#include <cmath>
#include <curand_kernel.h>
#include <thrust/partition.h>


using namespace IdealMHD2DConst;
using namespace PIC2DConst;
using namespace Interface2DConst;


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
    int indexStartPIC, 
    int length
)
    :  indexOfInterfaceStartInMHD(indexStartMHD), 
       indexOfInterfaceStartInPIC(indexStartPIC), 
       interfaceLength(length), 
       interlockingFunctionY(interfaceLength), 
       interlockingFunctionYHalf(interfaceLength - 1),
       host_interlockingFunctionY(interfaceLength), 
       host_interlockingFunctionYHalf(interfaceLength - 1),

       zerothMomentIon(PIC2DConst::nx_PIC * PIC2DConst::ny_PIC), 
       zerothMomentElectron(PIC2DConst::nx_PIC * PIC2DConst::ny_PIC), 
       firstMomentIon(PIC2DConst::nx_PIC * PIC2DConst::ny_PIC), 
       firstMomentElectron(PIC2DConst::nx_PIC * PIC2DConst::ny_PIC),

       reloadParticlesNumIon(0),
       reloadParticlesNumElectron(0), 
       restartParticlesIndexIon(0), 
       restartParticlesIndexElectron(0), 

       reloadParticlesDataIon(PIC2DConst::nx_PIC * interfaceLength), 
       reloadParticlesDataElectron(PIC2DConst::nx_PIC * interfaceLength), 

       reloadParticlesSourceIon(Interface2DConst::reloadParticlesTotalNumIon), 
       reloadParticlesSourceElectron(Interface2DConst::reloadParticlesTotalNumElectron), 

       reloadParticlesIndexIon(PIC2DConst::nx_PIC * interfaceLength), 
       reloadParticlesIndexElectron(PIC2DConst::nx_PIC * interfaceLength), 
       host_reloadParticlesIndexIon(PIC2DConst::nx_PIC * interfaceLength), 
       host_reloadParticlesIndexElectron(PIC2DConst::nx_PIC * interfaceLength), 

       B_timeAve(PIC2DConst::nx_PIC * PIC2DConst::ny_PIC), 
       zerothMomentIon_timeAve(PIC2DConst::nx_PIC * PIC2DConst::ny_PIC), 
       zerothMomentElectron_timeAve(PIC2DConst::nx_PIC * PIC2DConst::ny_PIC), 
       firstMomentIon_timeAve(PIC2DConst::nx_PIC * PIC2DConst::ny_PIC), 
       firstMomentElectron_timeAve(PIC2DConst::nx_PIC * PIC2DConst::ny_PIC)
{
    indexOfInterfaceEndInMHD = indexOfInterfaceStartInMHD + interfaceLength;
    indexOfInterfaceEndInPIC = indexOfInterfaceStartInPIC + interfaceLength;

    for(int i = 0; interfaceLength; i++) {
        host_interlockingFunctionY[i] = 0.5f * (
            1.0f + cos(Interface2DConst::PI  * (i - 0.0f) / (interfaceLength - 0.0f))
        );
    }
    for(int i = 0; interfaceLength - 1; i++) {
        host_interlockingFunctionY[i] = 0.5f * (
            1.0f + cos(Interface2DConst::PI  * (i + 0.5f - 0.0f) / (interfaceLength - 0.0f))
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
    int indexOfInterfaceStartInPIC, 
    int interfaceLength
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < PIC2DConst::device_nx_PIC && 0 < j && j < interfaceLength - 1) {
        float bXPIC, bYPIC, bZPIC;
        float bXMHD, bYMHD, bZMHD;
        float bXInterface, bYInterface, bZInterface;

        int indexPIC = indexOfInterfaceStartInPIC +  j + i * PIC2DConst::device_nx_PIC;
        int indexMHD = indexOfInterfaceStartInMHD +  j + i * IdealMHD2DConst::device_nx_MHD;

        //PICのグリッドにMHDを合わせる
        bXPIC = B[indexPIC].bX;
        bYPIC = B[indexPIC].bY;
        bZPIC = B[indexPIC].bZ;
        bXMHD = 0.25f * (U[indexMHD].bX + U[indexMHD - IdealMHD2DConst::device_nx_MHD].bX + U[indexMHD + 1].bX + U[indexMHD + 1 - IdealMHD2DConst::device_nx_MHD].bX);
        bYMHD = 0.25f * (U[indexMHD].bY + U[indexMHD + IdealMHD2DConst::device_nx_MHD].bY + U[indexMHD - 1].bY + U[indexMHD - 1 + IdealMHD2DConst::device_nx_MHD].bY);
        bZMHD = 0.25f * (U[indexMHD].bZ + U[indexMHD + IdealMHD2DConst::device_nx_MHD].bZ + U[indexMHD + 1].bZ + U[indexMHD + 1 + IdealMHD2DConst::device_nx_MHD].bZ);

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
    dim3 blocksPerGrid((PIC2DConst::nx_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (interfaceLength + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendMHDtoPIC_magneticField_yDirection_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()),
        thrust::raw_pointer_cast(interlockingFunctionYHalf.data()),
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(B.data()), 
        indexOfInterfaceStartInMHD, 
        indexOfInterfaceStartInPIC, 
        interfaceLength
    );

    cudaDeviceSynchronize();
}


__global__ void sendMHDtoPIC_electricField_yDirection_kernel(
    const float* interlockingFunctionY, 
    const float* interlockingFunctionYHalf, 
    const ConservationParameter* U, 
    ElectricField* E, 
    int indexOfInterfaceStartInMHD, 
    int indexOfInterfaceStartInPIC, 
    int interfaceLength
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < PIC2DConst::device_nx_PIC && 0 < j &&  j < interfaceLength - 1) {
        float eXPIC, eYPIC, eZPIC;
        float eXMHD, eYMHD, eZMHD;
        float eXPlusX1MHD;
        float eYPlusY1MHD;
        float eXMHD, eYMHD, eZMHD;
        float rho, u, v, w;
        float bXMHD, bYMHD, bZMHD;
        float eXInterface, eYInterface, eZInterface;

        int indexPIC = indexOfInterfaceStartInPIC +  j + i * PIC2DConst::device_nx_PIC;
        int indexMHD = indexOfInterfaceStartInMHD +  j + i * IdealMHD2DConst::device_nx_MHD;

        //PICのグリッドにMHDを合わせる
        eXPIC = E[indexPIC].eX;
        eYPIC = E[indexPIC].eY;
        eZPIC = E[indexPIC].eZ;

        rho = U[indexMHD].rho;
        u = U[indexMHD].rhoU / rho;
        v = U[indexMHD].rhoV / rho;
        w = U[indexMHD].rhoW / rho; 
        bXMHD = 0.5f * (U[indexMHD].bX + U[indexMHD - IdealMHD2DConst::device_nx_MHD].bX);
        bYMHD = 0.5f * (U[indexMHD].bY + U[indexMHD - 1].bY);
        bZMHD = U[indexMHD].bZ;
        eXMHD = -(v * bZMHD - w * bYMHD);
        eYMHD = -(w * bXMHD - u * bZMHD);
        eZMHD = -(u * bYMHD - v * bXMHD);

        rho = U[indexMHD + IdealMHD2DConst::device_nx_MHD].rho;
        u = U[indexMHD + IdealMHD2DConst::device_nx_MHD].rhoU / rho;
        v = U[indexMHD + IdealMHD2DConst::device_nx_MHD].rhoV / rho;
        w = U[indexMHD + IdealMHD2DConst::device_nx_MHD].rhoW / rho; 
        bXMHD = 0.5f * (U[indexMHD + IdealMHD2DConst::device_nx_MHD].bX + U[indexMHD].bX);
        bYMHD = 0.5f * (U[indexMHD + IdealMHD2DConst::device_nx_MHD].bY + U[indexMHD - 1 + IdealMHD2DConst::device_nx_MHD].bY);
        bZMHD = U[indexMHD + IdealMHD2DConst::device_nx_MHD].bZ;
        eXPlusX1MHD = -(v * bZMHD - w * bYMHD);

        rho = U[indexMHD + 1].rho;
        u = U[indexMHD + 1].rhoU / rho;
        v = U[indexMHD + 1].rhoV / rho;
        w = U[indexMHD + 1].rhoW / rho; 
        bXMHD = 0.5f * (U[indexMHD + 1].bX + U[indexMHD + 1 - IdealMHD2DConst::device_nx_MHD].bX);
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
    dim3 blocksPerGrid((PIC2DConst::nx_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (interfaceLength + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendMHDtoPIC_electricField_yDirection_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()), 
        thrust::raw_pointer_cast(interlockingFunctionYHalf.data()), 
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(E.data()), 
        indexOfInterfaceStartInMHD, 
        indexOfInterfaceStartInPIC, 
        interfaceLength
    );

    cudaDeviceSynchronize();
}


__global__ void sendMHDtoPIC_currentField_yDirection_kernel(
    const float* interlockingFunctionY, 
    const float* interlockingFunctionYHalf, 
    const ConservationParameter* U, 
    CurrentField* current, 
    int indexOfInterfaceStartInMHD, 
    int indexOfInterfaceStartInPIC, 
    int interfaceLength
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < PIC2DConst::device_nx_PIC && 0 < j && j < interfaceLength - 1) {
        float jXPIC, jYPIC, jZPIC;
        float jXMHD, jYMHD, jZMHD;
        float jXPlusX1MHD; 
        float jYPlusY1MHD; 
        float jXInterface, jYInterface, jZInterface;
        int nx = IdealMHD2DConst::device_nx_MHD;
        float dx = IdealMHD2DConst::device_dx_MHD, dy = IdealMHD2DConst::device_dy_MHD;

        int indexPIC = indexOfInterfaceStartInPIC +  j + i * PIC2DConst::device_nx_PIC;
        int indexMHD = indexOfInterfaceStartInMHD +  j + i * IdealMHD2DConst::device_nx_MHD;

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
    dim3 blocksPerGrid((PIC2DConst::nx_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (interfaceLength + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendMHDtoPIC_currentField_yDirection_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()), 
        thrust::raw_pointer_cast(interlockingFunctionYHalf.data()), 
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(current.data()), 
        indexOfInterfaceStartInMHD, 
        indexOfInterfaceStartInPIC, 
        interfaceLength
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
    int indexOfInterfaceStartInPIC, 
    int interfaceLength
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < PIC2DConst::device_nx_PIC && 0 < j && j < interfaceLength - 1) {
        int indexPIC = indexOfInterfaceStartInPIC + j + i * PIC2DConst::device_nx_PIC;
        int indexMHD = indexOfInterfaceStartInMHD + j + i * IdealMHD2DConst::device_nx_MHD;
        float rhoMHD, uMHD, vMHD, wMHD, bXMHD, bYMHD, bZMHD, eMHD, pMHD;
        float jXMHD, jYMHD, jZMHD, niMHD, neMHD, tiMHD, teMHD;
        float rhoPIC, uPIC, vPIC, wPIC;
        float jXPIC, jYPIC, jZPIC, niPIC, nePIC, vThiPIC, vThePIC;
        int nx = IdealMHD2DConst::device_nx_MHD;
        float dx = IdealMHD2DConst::device_dx_MHD, dy = IdealMHD2DConst::device_dy_MHD;

        //PICのグリッドにMHDを合わせる
        rhoMHD = U[indexMHD].rho;
        uMHD = U[indexMHD].rhoU / rhoMHD;
        vMHD = U[indexMHD].rhoV / rhoMHD;
        wMHD = U[indexMHD].rhoW / rhoMHD;
        bXMHD = 0.5f * (U[indexMHD].bX + U[indexMHD - nx].bX);
        bYMHD = 0.5f * (U[indexMHD].bY + U[indexMHD - 1].bY);
        bZMHD = U[indexMHD].bZ;
        eMHD = U[indexMHD].e;
        pMHD = (IdealMHD2DConst::gamma_MHD - 1.0f)
             * (eMHD - 0.5f * rhoMHD * (uMHD * uMHD + vMHD * vMHD + wMHD * wMHD)
             - 0.5f * (bXMHD * bXMHD + bYMHD * bYMHD + bZMHD * bZMHD));
        jXMHD = (U[indexMHD + 1].bZ - U[indexMHD - 1].bZ) / (2.0f * dy);
        jYMHD = -(U[indexMHD + nx].bZ - U[indexMHD - nx].bZ) / (2.0f * dx);
        jZMHD = 0.25f * ((U[indexMHD + nx].bY - U[indexMHD].bY) / dx - (U[indexMHD + 1].bX - U[indexMHD].bX) / dy 
                       + (U[indexMHD].bY - U[indexMHD - nx].bY) / dx - (U[indexMHD + 1 - nx].bX - U[indexMHD - nx].bX) / dy
                       + (U[indexMHD - 1 + nx].bY - U[indexMHD - 1].bY) / dx - (U[indexMHD].bX - U[indexMHD - 1].bX) / dy
                       + (U[indexMHD - 1].bY - U[indexMHD - 1 - nx].bY) / dx - (U[indexMHD - nx].bX - U[indexMHD - 1 - nx].bX) / dy);

        niMHD = rhoMHD / (PIC2DConst::mIon_PIC + PIC2DConst::mElectron_PIC);
        neMHD = niMHD;
        tiMHD = pMHD / 2.0f / niMHD;
        teMHD = pMHD / 2.0f / neMHD;

        rhoPIC = PIC2DConst::mIon_PIC * zerothMomentIon[indexPIC].n + PIC2DConst::mElectron_PIC * zerothMomentElectron[indexPIC].n;
        uPIC = (PIC2DConst::mIon_PIC * firstMomentIon[indexPIC].x + PIC2DConst::mElectron_PIC * firstMomentElectron[indexPIC].x) / rhoPIC;
        vPIC = (PIC2DConst::mIon_PIC * firstMomentIon[indexPIC].y + PIC2DConst::mElectron_PIC * firstMomentElectron[indexPIC].y) / rhoPIC;
        wPIC = (PIC2DConst::mIon_PIC * firstMomentIon[indexPIC].z + PIC2DConst::mElectron_PIC * firstMomentElectron[indexPIC].z) / rhoPIC;
        jXPIC = PIC2DConst::qIon_PIC * firstMomentIon[indexPIC].x + PIC2DConst::qElectron_PIC * firstMomentElectron[indexPIC].x;
        jYPIC = PIC2DConst::qIon_PIC * firstMomentIon[indexPIC].y + PIC2DConst::qElectron_PIC * firstMomentElectron[indexPIC].y;
        jZPIC = PIC2DConst::qIon_PIC * firstMomentIon[indexPIC].z + PIC2DConst::qElectron_PIC * firstMomentElectron[indexPIC].z;


        rhoPIC = interlockingFunctionY[j] * rhoMHD + (1.0f - interlockingFunctionY[j]) * rhoPIC;
        uPIC = interlockingFunctionY[j] * uMHD + (1.0f - interlockingFunctionY[j]) * uPIC;
        vPIC = interlockingFunctionY[j] * vMHD + (1.0f - interlockingFunctionY[j]) * vPIC;
        wPIC = interlockingFunctionY[j] * wMHD + (1.0f - interlockingFunctionY[j]) * wPIC;
        jXPIC = interlockingFunctionY[j] * jXMHD + (1.0f - interlockingFunctionY[j]) * jXPIC;
        jYPIC = interlockingFunctionY[j] * jYMHD + (1.0f - interlockingFunctionY[j]) * jYPIC;
        jZPIC = interlockingFunctionY[j] * jZMHD + (1.0f - interlockingFunctionY[j]) * jZPIC;

        niPIC = rhoPIC / (PIC2DConst::mIon_PIC + PIC2DConst::mElectron_PIC);
        nePIC = niPIC;
        vThiPIC = sqrt(2.0f * tiMHD / PIC2DConst::mIon_PIC);
        vThePIC = sqrt(2.0f * teMHD / PIC2DConst::mElectron_PIC);

        atomicAdd(&reloadParticlesNumIon, round(niPIC));
        atomicAdd(&reloadParticlesNumElectron, round(nePIC));

        reloadParticlesDataIon[j + i * PIC2DConst::device_nx_PIC].number = round(niPIC);
        reloadParticlesDataElectron[j + i * PIC2DConst::device_nx_PIC].number = round(nePIC);
        reloadParticlesDataIon[j + i * PIC2DConst::device_nx_PIC].u = uPIC;
        reloadParticlesDataIon[j + i * PIC2DConst::device_nx_PIC].v = vPIC;
        reloadParticlesDataIon[j + i * PIC2DConst::device_nx_PIC].w = wPIC;
        reloadParticlesDataElectron[j + i * PIC2DConst::device_nx_PIC].u = uPIC - jXPIC / round(nePIC) / abs(PIC2DConst::qElectron_PIC);
        reloadParticlesDataElectron[j + i * PIC2DConst::device_nx_PIC].v = vPIC - jYPIC / round(nePIC) / abs(PIC2DConst::qElectron_PIC);
        reloadParticlesDataElectron[j + i * PIC2DConst::device_nx_PIC].w = wPIC - jZPIC / round(nePIC) / abs(PIC2DConst::qElectron_PIC);
        reloadParticlesDataIon[j + i * PIC2DConst::device_nx_PIC].vth = vThiPIC;
        reloadParticlesDataElectron[j + i * PIC2DConst::device_nx_PIC].vth = vThePIC;

        reloadParticlesIndexIon[j + i * PIC2DConst::device_nx_PIC] = round(niPIC);
        reloadParticlesIndexElectron[j + i * PIC2DConst::device_nx_PIC] = round(nePIC);
    }
}



__global__ void reloadParticles_kernel(
    const float* interlockingFunctionY, 
    const ReloadParticlesData* reloadParticlesDataSpecies, 
    const int* reloadParticlesIndexSpecies, 
    const Particle* reloadParticlesSpecies, 
    Particle* particlesSpecies, 
    int reloadParticlesNumSpecies, 
    int restartParticlesIndexSpecies, 
    int indexOfInterfaceStartInPIC, 
    int interfaceLength, 
    int existNumSpecies, 
    int step
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < PIC2DConst::device_nx_PIC && 0 < j && j < interfaceLength - 1) {
        int index = j + i * PIC2DConst::device_nx_PIC;
        int reloadNum = reloadParticlesDataSpecies[index].number;
        float u = reloadParticlesDataSpecies[index].u;
        float v = reloadParticlesDataSpecies[index].v;
        float w = reloadParticlesDataSpecies[index].w;
        float vth = reloadParticlesDataSpecies[index].vth;
        Particle particleSource, particleReload;
        float x, y, z, vx, vy, vz, gamma;

        for (int k = reloadParticlesIndexSpecies[index]; k < reloadParticlesIndexSpecies[index + 1]; k++) {
            curandState state; 
            curand_init(step, k, 0, &state);
            float randomValue = curand_uniform(&state);

            if (randomValue > 1.0f - interlockingFunctionY[j]) {
                particleSource = reloadParticlesSpecies[(restartParticlesIndexSpecies + k) % Interface2DConst::device_reloadParticlesTotalNumIon];

                x = particleSource.x; x += i * PIC2DConst::device_dx_PIC;
                y = particleSource.y; y += (indexOfInterfaceStartInPIC + j) * PIC2DConst::device_dy_PIC;
                z = 0.0f;
                vx = particleSource.vx; vx = u + vx * vth;
                vy = particleSource.vx; vy = v + vy * vth;
                vz = particleSource.vx; vz = w + vz * vth;
                gamma = sqrt(1.0f + (vx * vx + vy * vy + vz * vz) / (PIC2DConst::device_c_PIC * PIC2DConst::device_c_PIC));
                
                particleReload.x = x; particleReload.y = y; particleReload.z = z;
                particleReload.vx = vx; particleReload.vy = vy, particleReload.vz = vz; 
                particleReload.gamma = gamma;

                particlesSpecies[existNumSpecies + k] = particleReload;
                particlesSpecies[existNumSpecies + k].isExist = true;
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

    deleteParticles(particlesIon, particlesElectron, step);

    reloadParticlesNumIon = 0;
    reloadParticlesNumElectron = 0;

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((PIC2DConst::nx_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (interfaceLength + threadsPerBlock.y - 1) / threadsPerBlock.y);

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
        indexOfInterfaceStartInPIC, 
        interfaceLength
    );

    cudaDeviceSynchronize();


    host_reloadParticlesIndexIon = reloadParticlesIndexIon;
    host_reloadParticlesIndexElectron = reloadParticlesIndexElectron;

    for (int i = 0; i < PIC2DConst::nx_PIC; i++) {
        for (int j = 0; j < interfaceLength; j++) {

            if (j == 0 && i == 0) continue;

            host_reloadParticlesIndexIon[j + i * PIC2DConst::nx_PIC] += host_reloadParticlesIndexIon[j + i * PIC2DConst::nx_PIC - 1];
            host_reloadParticlesIndexElectron[j + i * PIC2DConst::nx_PIC] += host_reloadParticlesIndexElectron[j + i * PIC2DConst::nx_PIC - 1];
        }
    }

    reloadParticlesIndexIon = host_reloadParticlesIndexIon;
    reloadParticlesIndexElectron = host_reloadParticlesIndexElectron;


    reloadParticles_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()), 
        thrust::raw_pointer_cast(reloadParticlesDataIon.data()), 
        thrust::raw_pointer_cast(reloadParticlesIndexIon.data()), 
        thrust::raw_pointer_cast(reloadParticlesSourceIon.data()), 
        thrust::raw_pointer_cast(particlesIon.data()), 
        reloadParticlesNumIon, restartParticlesIndexIon, 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        PIC2DConst::existNumIon_PIC, 
        step
    );

    cudaDeviceSynchronize();

    reloadParticles_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()), 
        thrust::raw_pointer_cast(reloadParticlesDataElectron.data()), 
        thrust::raw_pointer_cast(reloadParticlesIndexElectron.data()), 
        thrust::raw_pointer_cast(reloadParticlesSourceElectron.data()), 
        thrust::raw_pointer_cast(particlesElectron.data()), 
        reloadParticlesNumElectron, restartParticlesIndexElectron, 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        PIC2DConst::existNumElectron_PIC, 
        step
    );

    cudaDeviceSynchronize();

    restartParticlesIndexIon = reloadParticlesNumIon % Interface2DConst::reloadParticlesTotalNumIon;
    restartParticlesIndexElectron = reloadParticlesNumElectron % Interface2DConst::reloadParticlesTotalNumElectron;


    thrust::partition(
        particlesIon.begin(), particlesIon.end(), 
        [] __device__ (const Particle& p) { return p.isExist; }
    );
    cudaDeviceSynchronize();

    thrust::partition(
        particlesElectron.begin(), particlesElectron.end(), 
        [] __device__ (const Particle& p) { return p.isExist; }
    );
    cudaDeviceSynchronize();
}


void Interface2D::setMoments(
    const thrust::device_vector<Particle>& particlesIon, 
    const thrust::device_vector<Particle>& particlesElectron
)
{
    momentCalculater.calculateZerothMomentOfOneSpecies(
        zerothMomentIon, particlesIon, PIC2DConst::totalNumIon_PIC
    );
    momentCalculater.calculateZerothMomentOfOneSpecies(
        zerothMomentElectron, particlesElectron, PIC2DConst::totalNumElectron_PIC
    );

    momentCalculater.calculateFirstMomentOfOneSpecies(
        firstMomentIon, particlesIon, PIC2DConst::totalNumIon_PIC
    );
    momentCalculater.calculateFirstMomentOfOneSpecies(
        firstMomentElectron, particlesElectron, PIC2DConst::totalNumElectron_PIC
    );
}


__global__ void deleteParticles_kernel(
    const float* interlockingFunctionY, 
    Particle* particlesSpecies, 
    const int indexOfInterfaceStartInPIC, 
    int interfaceLength, 
    const unsigned long long existNumSpecies, 
    unsigned long long existNumParticleAfterDeleteSpecies, 
    int step
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        float y = particlesSpecies[i].y;
        float interfaceMin = indexOfInterfaceStartInPIC * PIC2DConst::device_dy_PIC;
        float interfaceMax = (indexOfInterfaceStartInPIC + interfaceLength) * PIC2DConst::device_dy_PIC;
        if (y >= interfaceMin + PIC2DConst::dy_PIC && y <= interfaceMax - PIC2DConst::dy_PIC) {
            int j = floor(y) - indexOfInterfaceStartInPIC;
            curandState state; 
            curand_init(step, i, 0, &state);
            float randomValue = curand_uniform(&state);
            if (randomValue > 1.0f - interlockingFunctionY[j]) {
                particlesSpecies[i].isExist = false;
                return;
            }
        }
        atomicAdd(&existNumParticleAfterDeleteSpecies, 1);
    }
}

void Interface2D::deleteParticles(
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron, 
    int step
)
{
    unsigned long long existNumParticleAfterDeleteIon;
    unsigned long long existNumParticleAfterDeleteElectron;
    existNumParticleAfterDeleteIon = 0;
    existNumParticleAfterDeleteElectron = 0;

    dim3 threadsPerBlockForIon(256);
    dim3 blocksPerGridForIon((PIC2DConst::existNumIon_PIC + threadsPerBlockForIon.x - 1) / threadsPerBlockForIon.x);

    deleteParticles_kernel<<<blocksPerGridForIon, threadsPerBlockForIon>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()), 
        thrust::raw_pointer_cast(particlesIon.data()),
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        PIC2DConst::existNumIon_PIC, 
        existNumParticleAfterDeleteIon, 
        step
    );

    cudaDeviceSynchronize();

    dim3 threadsPerBlockForElectron(256);
    dim3 blocksPerGridForElectron((PIC2DConst::existNumElectron_PIC + threadsPerBlockForElectron.x - 1) / threadsPerBlockForElectron.x);

    deleteParticles_kernel<<<blocksPerGridForElectron, threadsPerBlockForElectron>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()), 
        thrust::raw_pointer_cast(particlesElectron.data()),
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        PIC2DConst::existNumElectron_PIC, 
        existNumParticleAfterDeleteElectron, 
        step 
    );

    cudaDeviceSynchronize();

    PIC2DConst::existNumIon_PIC = existNumParticleAfterDeleteIon;
    PIC2DConst::existNumElectron_PIC = existNumParticleAfterDeleteElectron;


    thrust::partition(
        particlesIon.begin(), particlesIon.end(), 
        [] __device__ (const Particle& p) { return p.isExist; }
    );
    cudaDeviceSynchronize();

    thrust::partition(
        particlesElectron.begin(), particlesElectron.end(), 
        [] __device__ (const Particle& p) { return p.isExist; }
    );
    cudaDeviceSynchronize();
}



///////////////////////////////////////////////////////


__global__ void sendPICtoMHD_kernel(
    const float* interlockingFunctionY, 
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

    if (0 < i && i < PIC2DConst::device_nx_PIC && 0 < j && j < interfaceLength - 1) {
        int indexPIC = indexOfInterfaceStartInPIC + j + i * PIC2DConst::device_nx_PIC;
        int indexMHD = indexOfInterfaceStartInMHD + j + i * IdealMHD2DConst::device_nx_MHD;
        float rhoMHD, uMHD, vMHD, wMHD, bXMHD, bYMHD, bZMHD, eMHD, pMHD;
        float rhoPIC, uPIC, vPIC, wPIC, bXPIC, bYPIC, bZPIC;
        float niMHD, neMHD, tiMHD, teMHD;

        //MHDのグリッドにPICを合わせる(=MHDグリッドは整数格子点上にあるので、PICグリッドを整数格子点上に再配置する)
        rhoMHD = U[indexMHD].rho;
        uMHD = U[indexMHD].rhoU / rhoMHD;
        vMHD = U[indexMHD].rhoV / rhoMHD;
        wMHD = U[indexMHD].rhoW / rhoMHD;
        bXMHD = U[indexMHD].bX;
        bYMHD = U[indexMHD].bY;
        bZMHD = U[indexMHD].bZ;
        eMHD = U[indexMHD].e;
        pMHD = (IdealMHD2DConst::gamma_MHD - 1.0f)
             * (eMHD - 0.5f * rhoMHD * (uMHD * uMHD + vMHD * vMHD + wMHD * wMHD)
             - 0.5f * (bXMHD * bXMHD + bYMHD * bYMHD + bZMHD * bZMHD));
        //tiMHD, teMHDはMHDの情報のままにするために、この計算が必要。
        niMHD = rhoMHD / (PIC2DConst::mIon_PIC + PIC2DConst::mElectron_PIC);
        neMHD = niMHD;
        tiMHD = pMHD / 2.0f / niMHD;
        teMHD = pMHD / 2.0f / neMHD;
        
        rhoPIC = PIC2DConst::mIon_PIC * zerothMomentIon[indexPIC].n + PIC2DConst::mElectron_PIC * ZerothMomentElectron[indexPIC].n;
        uPIC = (PIC2DConst::mIon_PIC * firstMomentIon[indexPIC].x + PIC2DConst::mElectron_PIC * firstMomentElectron[indexPIC].x) / rhoPIC;
        vPIC = (PIC2DConst::mIon_PIC * firstMomentIon[indexPIC].y + PIC2DConst::mElectron_PIC * firstMomentElectron[indexPIC].y) / rhoPIC;
        wPIC = (PIC2DConst::mIon_PIC * firstMomentIon[indexPIC].z + PIC2DConst::mElectron_PIC * firstMomentElectron[indexPIC].z) / rhoPIC;
        bXPIC = 0.5f * (B[indexPIC].bX + B[indexPIC - 1].bX);
        bYPIC = 0.5f * (B[indexPIC].bY + B[indexPIC - PIC2DConst::device_nx_PIC].bY);
        bZPIC = 0.25f * (B[indexPIC].bZ + B[indexPIC - PIC2DConst::device_nx_PIC].bZ + B[indexPIC - 1].bZ + B[indexPIC - PIC2DConst::device_nx_PIC - 1].bZ);

        rhoMHD = interlockingFunctionY[j] * rhoMHD + (1.0f - interlockingFunctionY[j]) * rhoPIC;
        uMHD = interlockingFunctionY[j]   * uMHD   + (1.0f - interlockingFunctionY[j]) * uPIC;
        vMHD = interlockingFunctionY[j]   * vMHD   + (1.0f - interlockingFunctionY[j]) * vPIC;
        wMHD = interlockingFunctionY[j]   * wMHD   + (1.0f - interlockingFunctionY[j]) * wPIC;
        bXMHD = interlockingFunctionY[j]  * bXMHD  + (1.0f - interlockingFunctionY[j]) * bXPIC;
        bYMHD = interlockingFunctionY[j]  * bYMHD  + (1.0f - interlockingFunctionY[j]) * bYPIC;
        bZMHD = interlockingFunctionY[j]  * bZMHD  + (1.0f - interlockingFunctionY[j]) * bZPIC;

        niMHD = rhoMHD / (PIC2DConst::mIon_PIC + PIC2DConst::mElectron_PIC);
        neMHD = niMHD;
        pMHD = niMHD * tiMHD + neMHD * teMHD;


        U[indexMHD].rho = rhoMHD;
        U[indexMHD].rhoU = rhoMHD * uMHD;
        U[indexMHD].rhoV = rhoMHD * vMHD;
        U[indexMHD].rhoW = rhoMHD * wMHD;
        U[indexMHD].bX = bXMHD;
        U[indexMHD].bY = bYMHD;
        U[indexMHD].bZ = bZMHD;
        eMHD = pMHD / (IdealMHD2DConst::gamma_MHD - 1.0f)
             + 0.5f * rhoMHD * (uMHD * uMHD + vMHD * vMHD + wMHD * wMHD)
             + 0.5f * (bXMHD * bXMHD + bYMHD * bYMHD + bZMHD * bZMHD);
        U[indexMHD].e = eMHD;
    }
}


//MHDのグリッドを整数格子点上に再配置してから使うこと
void Interface2D::sendPICtoMHD(
    const thrust::device_vector<MagneticField>& B, 
    thrust::device_vector<ConservationParameter>& U
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((PIC2DConst::nx_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (interfaceLength + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendPICtoMHD_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()), 
        thrust::raw_pointer_cast(zerothMomentIon.data()), 
        thrust::raw_pointer_cast(zerothMomentElectron.data()), 
        thrust::raw_pointer_cast(firstMomentIon.data()), 
        thrust::raw_pointer_cast(firstMomentElectron.data()), 
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(U.data()), 
        indexOfInterfaceStartInMHD, 
        indexOfInterfaceStartInPIC, 
        interfaceLength
    );

    cudaDeviceSynchronize();

}


//////////////////////////////


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


void Interface2D::calculateTimeAveParameters(
    const thrust::device_vector<MagneticField>& B, 
    const thrust::device_vector<Particle>& particlesIon, 
    const thrust::device_vector<Particle>& particlesElectron
)
{

}

