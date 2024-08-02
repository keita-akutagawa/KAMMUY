#include "interface.hpp"
#include <cmath>
#include <curand_kernel.h>
#include <random>
#include <algorithm>
#include <thrust/fill.h>
#include <thrust/partition.h>


using namespace IdealMHD2DConst;
using namespace PIC2DConst;
using namespace Interface2DConst;


__global__ void initializeReloadParticlesSource_kernel(
    Particle* reloadParticlesSourceSpecies, 
    unsigned long long reloadParticlesNumSpecies, 
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
       indexOfInterfaceEndInMHD(indexStartMHD + length), 
       indexOfInterfaceEndInPIC(indexStartPIC + length), 

       interlockingFunctionY(interfaceLength, 0.0), 
       interlockingFunctionYHalf(interfaceLength - 1, 0.0),

       host_interlockingFunctionY(interfaceLength, 0.0), 
       host_interlockingFunctionYHalf(interfaceLength - 1, 0.0),

       zerothMomentIon(PIC2DConst::nx_PIC * PIC2DConst::ny_PIC), 
       zerothMomentElectron(PIC2DConst::nx_PIC * PIC2DConst::ny_PIC), 
       firstMomentIon(PIC2DConst::nx_PIC * PIC2DConst::ny_PIC), 
       firstMomentElectron(PIC2DConst::nx_PIC * PIC2DConst::ny_PIC),

       restartParticlesIndexIon(0), 
       restartParticlesIndexElectron(0), 

       reloadParticlesDataIon(PIC2DConst::nx_PIC * interfaceLength), 
       reloadParticlesDataElectron(PIC2DConst::nx_PIC * interfaceLength), 
       reloadParticlesSourceIon(Interface2DConst::reloadParticlesTotalNumIon), 
       reloadParticlesSourceElectron(Interface2DConst::reloadParticlesTotalNumElectron), 

       reloadParticlesIndexIon(PIC2DConst::nx_PIC * interfaceLength, 0), 
       reloadParticlesIndexElectron(PIC2DConst::nx_PIC * interfaceLength, 0), 
       host_reloadParticlesIndexIon(PIC2DConst::nx_PIC * interfaceLength, 0), 
       host_reloadParticlesIndexElectron(PIC2DConst::nx_PIC * interfaceLength, 0), 

       B_timeAve(PIC2DConst::nx_PIC * PIC2DConst::ny_PIC), 
       zerothMomentIon_timeAve(PIC2DConst::nx_PIC * PIC2DConst::ny_PIC), 
       zerothMomentElectron_timeAve(PIC2DConst::nx_PIC * PIC2DConst::ny_PIC), 
       firstMomentIon_timeAve(PIC2DConst::nx_PIC * PIC2DConst::ny_PIC), 
       firstMomentElectron_timeAve(PIC2DConst::nx_PIC * PIC2DConst::ny_PIC), 

       USub(IdealMHD2DConst::nx_MHD * IdealMHD2DConst::ny_MHD), 
       UHalf(IdealMHD2DConst::nx_MHD * IdealMHD2DConst::ny_MHD)  
{

    for(int i = 0; i < interfaceLength; i++) {
        host_interlockingFunctionY[i] = min(
            0.5 * (1.0 + cos(Interface2DConst::PI * (i - 0.0) / (interfaceLength - 0.0))), 
            1e-20
        );
    }
    for(int i = 0; i < interfaceLength - 1; i++) {
        host_interlockingFunctionYHalf[i] = min(
            0.5 * (1.0 + cos(Interface2DConst::PI * (i + 0.5 - 0.0) / (interfaceLength - 0.0))), 
            1e-20
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
    const double* interlockingFunctionY, 
    const double* interlockingFunctionYHalf, 
    const ConservationParameter* U, 
    MagneticField* B, 
    int indexOfInterfaceStartInMHD, 
    int indexOfInterfaceStartInPIC, 
    int interfaceLength
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < PIC2DConst::device_nx_PIC - 1 && 0 < j && j < interfaceLength - 1) {
        double bXPIC, bYPIC, bZPIC;
        double bXMHD, bYMHD, bZMHD;
        double bXInterface, bYInterface, bZInterface;

        int indexPIC = indexOfInterfaceStartInPIC + j + i * PIC2DConst::device_ny_PIC;
        int indexMHD = indexOfInterfaceStartInMHD + j + i * IdealMHD2DConst::device_ny_MHD;

        //PICのグリッドにMHDを合わせる
        bXPIC = B[indexPIC].bX;
        bYPIC = B[indexPIC].bY;
        bZPIC = B[indexPIC].bZ;
        bXMHD = 0.25 * (U[indexMHD].bX + U[indexMHD - IdealMHD2DConst::device_ny_MHD].bX + U[indexMHD + 1].bX + U[indexMHD + 1 - IdealMHD2DConst::device_ny_MHD].bX);
        bYMHD = 0.25 * (U[indexMHD].bY + U[indexMHD + IdealMHD2DConst::device_ny_MHD].bY + U[indexMHD - 1].bY + U[indexMHD - 1 + IdealMHD2DConst::device_ny_MHD].bY);
        bZMHD = 0.25 * (U[indexMHD].bZ + U[indexMHD + IdealMHD2DConst::device_ny_MHD].bZ + U[indexMHD + 1].bZ + U[indexMHD + 1 + IdealMHD2DConst::device_ny_MHD].bZ);

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
    const double* interlockingFunctionY, 
    const double* interlockingFunctionYHalf, 
    const ConservationParameter* U, 
    ElectricField* E, 
    int indexOfInterfaceStartInMHD, 
    int indexOfInterfaceStartInPIC, 
    int interfaceLength
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < PIC2DConst::device_nx_PIC - 1 && 0 < j && j < interfaceLength - 1) {
        double eXPIC, eYPIC, eZPIC;
        double eXMHD, eYMHD, eZMHD;
        double eXPlusX1MHD;
        double eYPlusY1MHD;
        double rho, u, v, w;
        double bXMHD, bYMHD, bZMHD;
        double eXInterface, eYInterface, eZInterface;
        int ny = IdealMHD2DConst::device_ny_MHD;

        int indexPIC = indexOfInterfaceStartInPIC + j + i * PIC2DConst::device_ny_PIC;
        int indexMHD = indexOfInterfaceStartInMHD + j + i * IdealMHD2DConst::device_ny_MHD;

        //PICのグリッドにMHDを合わせる
        eXPIC = E[indexPIC].eX;
        eYPIC = E[indexPIC].eY;
        eZPIC = E[indexPIC].eZ;

        rho = U[indexMHD].rho;
        u = U[indexMHD].rhoU / rho;
        v = U[indexMHD].rhoV / rho;
        w = U[indexMHD].rhoW / rho; 
        bXMHD = 0.5 * (U[indexMHD].bX + U[indexMHD - ny].bX);
        bYMHD = 0.5 * (U[indexMHD].bY + U[indexMHD - 1].bY);
        bZMHD = U[indexMHD].bZ;
        eXMHD = -(v * bZMHD - w * bYMHD);
        eYMHD = -(w * bXMHD - u * bZMHD);
        eZMHD = -(u * bYMHD - v * bXMHD);

        rho = U[indexMHD + ny].rho;
        u = U[indexMHD + ny].rhoU / rho;
        v = U[indexMHD + ny].rhoV / rho;
        w = U[indexMHD + ny].rhoW / rho; 
        bXMHD = 0.5 * (U[indexMHD + ny].bX + U[indexMHD].bX);
        bYMHD = 0.5 * (U[indexMHD + ny].bY + U[indexMHD - 1 + ny].bY);
        bZMHD = U[indexMHD + ny].bZ;
        eXPlusX1MHD = -(v * bZMHD - w * bYMHD);

        rho = U[indexMHD + 1].rho;
        u = U[indexMHD + 1].rhoU / rho;
        v = U[indexMHD + 1].rhoV / rho;
        w = U[indexMHD + 1].rhoW / rho; 
        bXMHD = 0.5 * (U[indexMHD + 1].bX + U[indexMHD + 1 - ny].bX);
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
    const double* interlockingFunctionY, 
    const double* interlockingFunctionYHalf, 
    const ConservationParameter* U, 
    CurrentField* current, 
    int indexOfInterfaceStartInMHD, 
    int indexOfInterfaceStartInPIC, 
    int interfaceLength
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < PIC2DConst::device_nx_PIC - 1 && 0 < j && j < interfaceLength - 1) {
        double jXPIC, jYPIC, jZPIC;
        double jXMHD, jYMHD, jZMHD;
        double jXPlusX1MHD; 
        double jYPlusY1MHD; 
        double jXInterface, jYInterface, jZInterface;
        int ny = IdealMHD2DConst::device_ny_MHD;
        double dx = IdealMHD2DConst::device_dx_MHD, dy = IdealMHD2DConst::device_dy_MHD;

        int indexPIC = indexOfInterfaceStartInPIC + j + i * PIC2DConst::device_ny_PIC;
        int indexMHD = indexOfInterfaceStartInMHD + j + i * IdealMHD2DConst::device_ny_MHD;

        //PICのグリッドにMHDを合わせる
        jXPIC = current[indexPIC].jX;
        jYPIC = current[indexPIC].jY;
        jZPIC = current[indexPIC].jZ;
        jXMHD = (U[indexMHD + 1].bZ - U[indexMHD - 1].bZ) / (2.0 * dy);
        jYMHD = -(U[indexMHD + ny].bZ - U[indexMHD - ny].bZ) / (2.0 * dx);
        jZMHD = 0.25 * ((U[indexMHD + ny].bY - U[indexMHD].bY) / dx - (U[indexMHD + 1].bX - U[indexMHD].bX) / dy 
              + (U[indexMHD].bY - U[indexMHD - ny].bY) / dx - (U[indexMHD + 1 - ny].bX - U[indexMHD - ny].bX) / dy
              + (U[indexMHD - 1 + ny].bY - U[indexMHD - 1].bY) / dx - (U[indexMHD].bX - U[indexMHD - 1].bX) / dy
              + (U[indexMHD - 1].bY - U[indexMHD - 1 - ny].bY) / dx - (U[indexMHD - ny].bX - U[indexMHD - 1 - ny].bX) / dy);

        jXPlusX1MHD = (U[indexMHD + 2].bZ - U[indexMHD].bZ) / (2.0 * dy);
        jYPlusY1MHD = -(U[indexMHD + 2 * ny].bZ - U[indexMHD].bZ) / (2.0 * dx);

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


__device__ void cudaAssert(bool condition, int value1, double value2) {
    if (!condition) {
        printf("%d : %f \n", value1, value2);
    }
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
    unsigned long long* reloadParticlesIndexIon, 
    unsigned long long* reloadParticlesIndexElectron, 
    int indexOfInterfaceStartInMHD, 
    int indexOfInterfaceStartInPIC, 
    int interfaceLength
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < PIC2DConst::device_nx_PIC - 1 && 0 < j && j < interfaceLength - 1) {
        int indexPIC = indexOfInterfaceStartInPIC + j + i * PIC2DConst::device_ny_PIC;
        int indexMHD = indexOfInterfaceStartInMHD + j + i * IdealMHD2DConst::device_ny_MHD;
        double rhoMHD, uMHD, vMHD, wMHD, bXMHD, bYMHD, bZMHD, eMHD, pMHD;
        double jXMHD, jYMHD, jZMHD, niMHD, neMHD, tiMHD, teMHD;
        double rhoPIC, uPIC, vPIC, wPIC;
        double jXPIC, jYPIC, jZPIC, niPIC, nePIC, vThiPIC, vThePIC;
        int ny = IdealMHD2DConst::device_ny_MHD;
        double dx = IdealMHD2DConst::device_dx_MHD, dy = IdealMHD2DConst::device_dy_MHD;

        //PICのグリッドにMHDを合わせる
        rhoMHD = U[indexMHD].rho;
        uMHD = U[indexMHD].rhoU / rhoMHD;
        vMHD = U[indexMHD].rhoV / rhoMHD;
        wMHD = U[indexMHD].rhoW / rhoMHD;
        bXMHD = 0.5 * (U[indexMHD].bX + U[indexMHD - ny].bX);
        bYMHD = 0.5 * (U[indexMHD].bY + U[indexMHD - 1].bY);
        bZMHD = U[indexMHD].bZ;
        eMHD = U[indexMHD].e;
        pMHD = (IdealMHD2DConst::device_gamma_MHD - 1.0)
             * (eMHD - 0.5 * rhoMHD * (uMHD * uMHD + vMHD * vMHD + wMHD * wMHD)
             - 0.5 * (bXMHD * bXMHD + bYMHD * bYMHD + bZMHD * bZMHD));
        jXMHD = (U[indexMHD + 1].bZ - U[indexMHD - 1].bZ) / (2.0 * dy);
        jYMHD = -(U[indexMHD + ny].bZ - U[indexMHD - ny].bZ) / (2.0 * dx);
        jZMHD = 0.25 * ((U[indexMHD + ny].bY - U[indexMHD].bY) / dx - (U[indexMHD + 1].bX - U[indexMHD].bX) / dy 
                       + (U[indexMHD].bY - U[indexMHD - ny].bY) / dx - (U[indexMHD + 1 - ny].bX - U[indexMHD - ny].bX) / dy
                       + (U[indexMHD - 1 + ny].bY - U[indexMHD - 1].bY) / dx - (U[indexMHD].bX - U[indexMHD - 1].bX) / dy
                       + (U[indexMHD - 1].bY - U[indexMHD - 1 - ny].bY) / dx - (U[indexMHD - ny].bX - U[indexMHD - 1 - ny].bX) / dy);

        niMHD = rhoMHD / (PIC2DConst::device_mIon_PIC + PIC2DConst::device_mElectron_PIC);
        neMHD = niMHD;
        tiMHD = pMHD / 2.0 / niMHD;
        teMHD = pMHD / 2.0 / neMHD;

        rhoPIC = PIC2DConst::device_mIon_PIC * zerothMomentIon[indexPIC].n + PIC2DConst::device_mElectron_PIC * zerothMomentElectron[indexPIC].n;
        uPIC = (PIC2DConst::device_mIon_PIC * firstMomentIon[indexPIC].x + PIC2DConst::device_mElectron_PIC * firstMomentElectron[indexPIC].x) / rhoPIC;
        vPIC = (PIC2DConst::device_mIon_PIC * firstMomentIon[indexPIC].y + PIC2DConst::device_mElectron_PIC * firstMomentElectron[indexPIC].y) / rhoPIC;
        wPIC = (PIC2DConst::device_mIon_PIC * firstMomentIon[indexPIC].z + PIC2DConst::device_mElectron_PIC * firstMomentElectron[indexPIC].z) / rhoPIC;
        jXPIC = PIC2DConst::device_qIon_PIC * firstMomentIon[indexPIC].x + PIC2DConst::device_qElectron_PIC * firstMomentElectron[indexPIC].x;
        jYPIC = PIC2DConst::device_qIon_PIC * firstMomentIon[indexPIC].y + PIC2DConst::device_qElectron_PIC * firstMomentElectron[indexPIC].y;
        jZPIC = PIC2DConst::device_qIon_PIC * firstMomentIon[indexPIC].z + PIC2DConst::device_qElectron_PIC * firstMomentElectron[indexPIC].z;

        rhoPIC = interlockingFunctionY[j] * rhoMHD + (1.0 - interlockingFunctionY[j]) * rhoPIC;
        uPIC = interlockingFunctionY[j] * uMHD + (1.0 - interlockingFunctionY[j]) * uPIC;
        vPIC = interlockingFunctionY[j] * vMHD + (1.0 - interlockingFunctionY[j]) * vPIC;
        wPIC = interlockingFunctionY[j] * wMHD + (1.0 - interlockingFunctionY[j]) * wPIC;
        jXPIC = interlockingFunctionY[j] * jXMHD + (1.0 - interlockingFunctionY[j]) * jXPIC;
        jYPIC = interlockingFunctionY[j] * jYMHD + (1.0 - interlockingFunctionY[j]) * jYPIC;
        jZPIC = interlockingFunctionY[j] * jZMHD + (1.0 - interlockingFunctionY[j]) * jZPIC;

        niPIC = rhoPIC / (PIC2DConst::device_mIon_PIC + PIC2DConst::device_mElectron_PIC);
        nePIC = niPIC;
        vThiPIC = sqrt(2.0 * tiMHD / PIC2DConst::device_mIon_PIC);
        vThePIC = sqrt(2.0 * teMHD / PIC2DConst::device_mElectron_PIC);


        reloadParticlesDataIon[j + i * interfaceLength].number = round(niPIC);
        reloadParticlesDataElectron[j + i * interfaceLength].number = round(nePIC);
        reloadParticlesDataIon[j + i * interfaceLength].u = uPIC;
        reloadParticlesDataIon[j + i * interfaceLength].v = vPIC;
        reloadParticlesDataIon[j + i * interfaceLength].w = wPIC;
        reloadParticlesDataElectron[j + i * interfaceLength].u = uPIC - jXPIC / round(nePIC) / abs(PIC2DConst::device_qElectron_PIC);
        reloadParticlesDataElectron[j + i * interfaceLength].v = vPIC - jYPIC / round(nePIC) / abs(PIC2DConst::device_qElectron_PIC);
        reloadParticlesDataElectron[j + i * interfaceLength].w = wPIC - jZPIC / round(nePIC) / abs(PIC2DConst::device_qElectron_PIC);
        reloadParticlesDataIon[j + i * interfaceLength].vth = vThiPIC;
        reloadParticlesDataElectron[j + i * interfaceLength].vth = vThePIC;

        reloadParticlesIndexIon[j + i * interfaceLength] = round(niPIC);
        reloadParticlesIndexElectron[j + i * interfaceLength] = round(nePIC);
    }
}


__global__ void reloadParticles_kernel(
    const double* interlockingFunctionY, 
    const ReloadParticlesData* reloadParticlesDataSpecies, 
    const unsigned long long* reloadParticlesIndexSpecies, 
    const Particle* reloadParticlesSpecies, 
    unsigned long long reloadParticlesTotalNumSpecies, 
    Particle* particlesSpecies, 
    unsigned long long restartParticlesIndexSpecies, 
    int indexOfInterfaceStartInPIC, 
    int interfaceLength, 
    unsigned long long existNumSpecies, 
    int step
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < PIC2DConst::device_nx_PIC && j < interfaceLength) {
        int index = j + i * interfaceLength;
        //int reloadNum = reloadParticlesDataSpecies[index].number;
        double u = reloadParticlesDataSpecies[index].u;
        double v = reloadParticlesDataSpecies[index].v;
        double w = reloadParticlesDataSpecies[index].w;
        double vth = reloadParticlesDataSpecies[index].vth;
        Particle particleSource, particleReload;
        double x, y, z, vx, vy, vz, gamma;

        for (unsigned long long k = reloadParticlesIndexSpecies[index]; k < reloadParticlesIndexSpecies[index + 1]; k++) {
            curandState state; 
            curand_init(step, k, 0, &state);
            double randomValue = curand_uniform(&state);

            if (randomValue > 1.0 - interlockingFunctionY[j]) {
                particleSource = reloadParticlesSpecies[(restartParticlesIndexSpecies + k) % reloadParticlesTotalNumSpecies];

                x = particleSource.x; x += i * PIC2DConst::device_dx_PIC;
                y = particleSource.y; y += (indexOfInterfaceStartInPIC + j) * PIC2DConst::device_dy_PIC;
                z = 0.0;
                vx = particleSource.vx; vx = u + vx * vth;
                vy = particleSource.vx; vy = v + vy * vth;
                vz = particleSource.vx; vz = w + vz * vth;
                gamma = sqrt(1.0 + (vx * vx + vy * vy + vz * vz) / (PIC2DConst::device_c_PIC * PIC2DConst::device_c_PIC));
                
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

    thrust::fill(reloadParticlesIndexIon.begin(), reloadParticlesIndexIon.end(), 0);
    thrust::fill(reloadParticlesIndexElectron.begin(), reloadParticlesIndexElectron.end(), 0);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((PIC2DConst::nx_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (interfaceLength + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendMHDtoPIC_particle_yDirection_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()), 
        thrust::raw_pointer_cast(zerothMomentIon.data()), 
        thrust::raw_pointer_cast(zerothMomentElectron.data()), 
        thrust::raw_pointer_cast(firstMomentIon.data()), 
        thrust::raw_pointer_cast(firstMomentElectron.data()), 
        thrust::raw_pointer_cast(U.data()),  
        thrust::raw_pointer_cast(reloadParticlesDataIon.data()), 
        thrust::raw_pointer_cast(reloadParticlesDataElectron.data()), 
        thrust::raw_pointer_cast(reloadParticlesIndexIon.data()), 
        thrust::raw_pointer_cast(reloadParticlesIndexElectron.data()), 
        indexOfInterfaceStartInMHD, 
        indexOfInterfaceStartInPIC, 
        interfaceLength
    );

    cudaDeviceSynchronize();
    
    deleteParticles(particlesIon, particlesElectron, step);
    

    host_reloadParticlesIndexIon = reloadParticlesIndexIon;
    host_reloadParticlesIndexElectron = reloadParticlesIndexElectron;
    
    for (int i = 0; i < PIC2DConst::nx_PIC; i++) {
        for (int j = 0; j < interfaceLength; j++) {

            if (j == 0 && i == 0) continue;

            host_reloadParticlesIndexIon[j + i * interfaceLength] += host_reloadParticlesIndexIon[j + i * interfaceLength - 1];
            host_reloadParticlesIndexElectron[j + i * interfaceLength] += host_reloadParticlesIndexElectron[j + i * interfaceLength - 1];
        }
    }

    reloadParticlesIndexIon = host_reloadParticlesIndexIon;
    reloadParticlesIndexElectron = host_reloadParticlesIndexElectron;
    

    std::random_device seedIon, seedElectron;
    std::mt19937 genIon(seedIon()), genElectron(seedElectron());
    std::uniform_int_distribution<unsigned long long> distIon(0, Interface2DConst::reloadParticlesTotalNumIon);
    std::uniform_int_distribution<unsigned long long> distElectron(0, Interface2DConst::reloadParticlesTotalNumElectron);
    restartParticlesIndexIon = distIon(genIon);
    restartParticlesIndexElectron = distElectron(genElectron);

    reloadParticles_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()), 
        thrust::raw_pointer_cast(reloadParticlesDataIon.data()), 
        thrust::raw_pointer_cast(reloadParticlesIndexIon.data()), 
        thrust::raw_pointer_cast(reloadParticlesSourceIon.data()), 
        Interface2DConst::reloadParticlesTotalNumIon,  
        thrust::raw_pointer_cast(particlesIon.data()), 
        restartParticlesIndexIon, 
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
        Interface2DConst::reloadParticlesTotalNumElectron, 
        thrust::raw_pointer_cast(particlesElectron.data()), 
        restartParticlesIndexElectron, 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        PIC2DConst::existNumElectron_PIC, 
        step
    );

    cudaDeviceSynchronize();


    PIC2DConst::existNumIon_PIC = thrust::transform_reduce(
        particlesIon.begin(),
        particlesIon.end(),
        IsExistTransform(), 
        0,               
        thrust::plus<unsigned long long>()
    );

    cudaDeviceSynchronize();

    PIC2DConst::existNumElectron_PIC = thrust::transform_reduce(
        particlesElectron.begin(),
        particlesElectron.end(),
        IsExistTransform(), 
        0,               
        thrust::plus<unsigned long long>()
    );

    cudaDeviceSynchronize();

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
        zerothMomentIon, particlesIon, PIC2DConst::existNumIon_PIC
    );
    momentCalculater.calculateZerothMomentOfOneSpecies(
        zerothMomentElectron, particlesElectron, PIC2DConst::existNumElectron_PIC
    );

    momentCalculater.calculateFirstMomentOfOneSpecies(
        firstMomentIon, particlesIon, PIC2DConst::existNumIon_PIC
    );
    momentCalculater.calculateFirstMomentOfOneSpecies(
        firstMomentElectron, particlesElectron, PIC2DConst::existNumElectron_PIC
    );
}


__global__ void deleteParticles_kernel(
    const double* interlockingFunctionY, 
    Particle* particlesSpecies, 
    const int indexOfInterfaceStartInPIC, 
    int interfaceLength, 
    const unsigned long long existNumSpecies, 
    int step
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        double y = particlesSpecies[i].y;
        double interfaceMin = indexOfInterfaceStartInPIC * PIC2DConst::device_dy_PIC;
        double interfaceMax = (indexOfInterfaceStartInPIC + interfaceLength) * PIC2DConst::device_dy_PIC;
        if (interfaceMin < y && y < interfaceMax) {
            int j = floor(y) - indexOfInterfaceStartInPIC;
            curandState state; 
            curand_init(step, i, 0, &state);
            double randomValue = curand_uniform(&state);
            if (randomValue > 1.0 - interlockingFunctionY[j]) {
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
    dim3 blocksPerGridForIon((PIC2DConst::existNumIon_PIC + threadsPerBlockForIon.x - 1) / threadsPerBlockForIon.x);

    deleteParticles_kernel<<<blocksPerGridForIon, threadsPerBlockForIon>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()), 
        thrust::raw_pointer_cast(particlesIon.data()),
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        PIC2DConst::existNumIon_PIC, 
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
        step 
    );

    cudaDeviceSynchronize();

    
    PIC2DConst::existNumIon_PIC = thrust::transform_reduce(
        particlesIon.begin(),
        particlesIon.end(),
        IsExistTransform(), 
        0,               
        thrust::plus<unsigned long long>()
    );

    cudaDeviceSynchronize();

    PIC2DConst::existNumElectron_PIC = thrust::transform_reduce(
        particlesElectron.begin(),
        particlesElectron.end(),
        IsExistTransform(), 
        0,               
        thrust::plus<unsigned long long>()
    );

    cudaDeviceSynchronize();


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


__global__ void setUHalf_kernel(
    const ConservationParameter* UPast, 
    const ConservationParameter* UNext, 
    ConservationParameter* UHalf
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < IdealMHD2DConst::device_nx_MHD && j < IdealMHD2DConst::device_ny_MHD) {
        int index = j + i * IdealMHD2DConst::device_ny_MHD;

        UHalf[index].rho = 0.5 * (UPast[index].rho + UNext[index].rho);
        UHalf[index].rhoU = 0.5 * (UPast[index].rhoU + UNext[index].rhoU);
        UHalf[index].rhoV = 0.5 * (UPast[index].rhoV + UNext[index].rhoV);
        UHalf[index].rhoW = 0.5 * (UPast[index].rhoW + UNext[index].rhoW);
        UHalf[index].bX = 0.5 * (UPast[index].bX + UNext[index].bX);
        UHalf[index].bY = 0.5 * (UPast[index].bY + UNext[index].bY);
        UHalf[index].bZ = 0.5 * (UPast[index].bZ + UNext[index].bZ);
        UHalf[index].e = 0.5 * (UPast[index].e + UNext[index].e);
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
    int interfaceLength
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < PIC2DConst::device_nx_PIC - 1 && 0 < j && j < interfaceLength - 1) {
        int indexPIC = indexOfInterfaceStartInPIC + j + i * PIC2DConst::device_ny_PIC;
        int indexMHD = indexOfInterfaceStartInMHD + j + i * IdealMHD2DConst::device_ny_MHD;
        double rhoMHD, uMHD, vMHD, wMHD, bXMHD, bYMHD, bZMHD, eMHD, pMHD;
        double rhoPIC, uPIC, vPIC, wPIC, bXPIC, bYPIC, bZPIC;
        double niMHD, neMHD, tiMHD, teMHD;

        //MHDのグリッドにPICを合わせる(=MHDグリッドは整数格子点上にあるので、PICグリッドを整数格子点上に再配置する)
        rhoMHD = U[indexMHD].rho;
        uMHD = U[indexMHD].rhoU / rhoMHD;
        vMHD = U[indexMHD].rhoV / rhoMHD;
        wMHD = U[indexMHD].rhoW / rhoMHD;
        bXMHD = U[indexMHD].bX;
        bYMHD = U[indexMHD].bY;
        bZMHD = U[indexMHD].bZ;
        eMHD = U[indexMHD].e;
        pMHD = (IdealMHD2DConst::device_gamma_MHD - 1.0)
             * (eMHD - 0.5 * rhoMHD * (uMHD * uMHD + vMHD * vMHD + wMHD * wMHD)
             - 0.5 * (bXMHD * bXMHD + bYMHD * bYMHD + bZMHD * bZMHD));
        //tiMHD, teMHDはMHDの情報のままにするために、この計算が必要。
        niMHD = rhoMHD / (PIC2DConst::device_mIon_PIC + PIC2DConst::device_mElectron_PIC);
        neMHD = niMHD;
        tiMHD = pMHD / 2.0 / niMHD;
        teMHD = pMHD / 2.0 / neMHD;
        
        rhoPIC = PIC2DConst::device_mIon_PIC * zerothMomentIon[indexPIC].n + PIC2DConst::device_mElectron_PIC * ZerothMomentElectron[indexPIC].n;
        uPIC = (PIC2DConst::device_mIon_PIC * firstMomentIon[indexPIC].x + PIC2DConst::device_mElectron_PIC * firstMomentElectron[indexPIC].x) / rhoPIC;
        vPIC = (PIC2DConst::device_mIon_PIC * firstMomentIon[indexPIC].y + PIC2DConst::device_mElectron_PIC * firstMomentElectron[indexPIC].y) / rhoPIC;
        wPIC = (PIC2DConst::device_mIon_PIC * firstMomentIon[indexPIC].z + PIC2DConst::device_mElectron_PIC * firstMomentElectron[indexPIC].z) / rhoPIC;
        bXPIC = 0.5 * (B[indexPIC].bX + B[indexPIC - 1].bX);
        bYPIC = 0.5 * (B[indexPIC].bY + B[indexPIC - PIC2DConst::device_ny_PIC].bY);
        bZPIC = 0.25 * (B[indexPIC].bZ + B[indexPIC - PIC2DConst::device_ny_PIC].bZ + B[indexPIC - 1].bZ + B[indexPIC - PIC2DConst::device_ny_PIC - 1].bZ);

        rhoMHD = interlockingFunctionY[j] * rhoMHD + (1.0 - interlockingFunctionY[j]) * rhoPIC;
        uMHD = interlockingFunctionY[j]   * uMHD   + (1.0 - interlockingFunctionY[j]) * uPIC;
        vMHD = interlockingFunctionY[j]   * vMHD   + (1.0 - interlockingFunctionY[j]) * vPIC;
        wMHD = interlockingFunctionY[j]   * wMHD   + (1.0 - interlockingFunctionY[j]) * wPIC;
        bXMHD = interlockingFunctionY[j]  * bXMHD  + (1.0 - interlockingFunctionY[j]) * bXPIC;
        bYMHD = interlockingFunctionY[j]  * bYMHD  + (1.0 - interlockingFunctionY[j]) * bYPIC;
        bZMHD = interlockingFunctionY[j]  * bZMHD  + (1.0 - interlockingFunctionY[j]) * bZPIC;

        niMHD = rhoMHD / (PIC2DConst::device_mIon_PIC + PIC2DConst::device_mElectron_PIC);
        neMHD = niMHD;
        pMHD = niMHD * tiMHD + neMHD * teMHD;


        U[indexMHD].rho = rhoMHD;
        U[indexMHD].rhoU = rhoMHD * uMHD;
        U[indexMHD].rhoV = rhoMHD * vMHD;
        U[indexMHD].rhoW = rhoMHD * wMHD;
        U[indexMHD].bX = bXMHD;
        U[indexMHD].bY = bYMHD;
        U[indexMHD].bZ = bZMHD;
        eMHD = pMHD / (IdealMHD2DConst::device_gamma_MHD - 1.0)
             + 0.5 * rhoMHD * (uMHD * uMHD + vMHD * vMHD + wMHD * wMHD)
             + 0.5 * (bXMHD * bXMHD + bYMHD * bYMHD + bZMHD * bZMHD);
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
    dim3 blocksPerGridForSetUHalf((IdealMHD2DConst::nx_MHD + threadsPerBlockForSetUHalf.x - 1) / threadsPerBlockForSetUHalf.x,
                                  (IdealMHD2DConst::ny_MHD + threadsPerBlockForSetUHalf.y - 1) / threadsPerBlockForSetUHalf.y);

    setUHalf_kernel<<<blocksPerGridForSetUHalf, threadsPerBlockForSetUHalf>>>(
        thrust::raw_pointer_cast(UPast.data()), 
        thrust::raw_pointer_cast(UNext.data()), 
        thrust::raw_pointer_cast(UHalf.data())
    );


    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((PIC2DConst::nx_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (interfaceLength + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendPICtoMHD_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()), 
        thrust::raw_pointer_cast(zerothMomentIon_timeAve.data()), 
        thrust::raw_pointer_cast(zerothMomentElectron_timeAve.data()), 
        thrust::raw_pointer_cast(firstMomentIon_timeAve.data()), 
        thrust::raw_pointer_cast(firstMomentElectron_timeAve.data()), 
        thrust::raw_pointer_cast(B_timeAve.data()), 
        thrust::raw_pointer_cast(UHalf.data()), 
        indexOfInterfaceStartInMHD, 
        indexOfInterfaceStartInPIC, 
        interfaceLength
    );

    cudaDeviceSynchronize();

}


thrust::device_vector<ConservationParameter>& Interface2D::getUHalfRef()
{
    return UHalf;
}

//////////////////////////////


__global__ void calculateSubU_kernel(
    const ConservationParameter* UPast, 
    const ConservationParameter* UNext, 
    ConservationParameter* USub, 
    double mixingRatio, 
    int indexOfInterfaceStartInMHD
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < IdealMHD2DConst::device_nx_MHD && j < IdealMHD2DConst::device_ny_MHD) {
        int index = j + i * IdealMHD2DConst::device_ny_MHD;

        USub[index].rho = mixingRatio * UPast[index].rho + (1.0 - mixingRatio) * UNext[index].rho;
        USub[index].rhoU = mixingRatio * UPast[index].rhoU + (1.0 - mixingRatio) * UNext[index].rhoU;
        USub[index].rhoV = mixingRatio * UPast[index].rhoV + (1.0 - mixingRatio) * UNext[index].rhoV;
        USub[index].rhoW = mixingRatio * UPast[index].rhoW + (1.0 - mixingRatio) * UNext[index].rhoW;
        USub[index].bX = mixingRatio * UPast[index].bX + (1.0 - mixingRatio) * UNext[index].bX;
        USub[index].bY = mixingRatio * UPast[index].bY + (1.0 - mixingRatio) * UNext[index].bY;
        USub[index].bZ = mixingRatio * UPast[index].bZ + (1.0 - mixingRatio) * UNext[index].bZ;
        USub[index].e = mixingRatio * UPast[index].e + (1.0 - mixingRatio) * UNext[index].e;
    }
}

thrust::device_vector<ConservationParameter>& Interface2D::calculateAndGetSubU(
    const thrust::device_vector<ConservationParameter>& UPast, 
    const thrust::device_vector<ConservationParameter>& UNext, 
    double mixingRatio
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((IdealMHD2DConst::nx_MHD + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (IdealMHD2DConst::ny_MHD + threadsPerBlock.y - 1) / threadsPerBlock.y);

    calculateSubU_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(UPast.data()), 
        thrust::raw_pointer_cast(UNext.data()), 
        thrust::raw_pointer_cast(USub.data()), 
        mixingRatio, 
        indexOfInterfaceStartInMHD
    );

    cudaDeviceSynchronize();

    return USub;
}


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


void Interface2D::sumUpTimeAveParameters(
    const thrust::device_vector<MagneticField>& B, 
    const thrust::device_vector<Particle>& particlesIon, 
    const thrust::device_vector<Particle>& particlesElectron
)
{
    thrust::transform(
        B_timeAve.begin(), B_timeAve.end(), B.begin(), 
        B_timeAve.begin(), thrust::plus<MagneticField>()
    );
    
    setMoments(particlesIon, particlesElectron);
    thrust::transform(
        zerothMomentIon_timeAve.begin(), zerothMomentIon_timeAve.end(), zerothMomentIon.begin(), 
        zerothMomentIon_timeAve.begin(), thrust::plus<ZerothMoment>()
    );
    thrust::transform(
        zerothMomentElectron_timeAve.begin(), zerothMomentElectron_timeAve.end(), zerothMomentElectron.begin(), 
        zerothMomentElectron_timeAve.begin(), thrust::plus<ZerothMoment>()
    );
    thrust::transform(
        firstMomentIon_timeAve.begin(), firstMomentIon_timeAve.end(), firstMomentIon.begin(), 
        firstMomentIon_timeAve.begin(), thrust::plus<FirstMoment>()
    );
    thrust::transform(
        firstMomentElectron_timeAve.begin(), firstMomentElectron_timeAve.end(), firstMomentElectron.begin(), 
        firstMomentElectron_timeAve.begin(), thrust::plus<FirstMoment>()
    );
}


__global__ void calculateTimeAveParameters_kernel(
    MagneticField* B_timeAve, 
    ZerothMoment* zerothMomentIon_timeAve, 
    ZerothMoment* zerothMomentElectron_timeAve, 
    FirstMoment* firstMomentIon_timeAve, 
    FirstMoment* firstMomentElectron_timeAve, 
    int substeps 
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < PIC2DConst::device_nx_PIC && j < PIC2DConst::device_ny_PIC) {
        int index = j + i * PIC2DConst::device_ny_PIC;

        B_timeAve[index].bX /= static_cast<double>(substeps);
        B_timeAve[index].bY /= static_cast<double>(substeps);
        B_timeAve[index].bZ /= static_cast<double>(substeps);
        zerothMomentIon_timeAve[index].n /= static_cast<double>(substeps);
        zerothMomentElectron_timeAve[index].n /= static_cast<double>(substeps);
        firstMomentIon_timeAve[index].x /= static_cast<double>(substeps);
        firstMomentIon_timeAve[index].y /= static_cast<double>(substeps);
        firstMomentIon_timeAve[index].z /= static_cast<double>(substeps);
        firstMomentElectron_timeAve[index].x /= static_cast<double>(substeps);
        firstMomentElectron_timeAve[index].y /= static_cast<double>(substeps);
        firstMomentElectron_timeAve[index].z /= static_cast<double>(substeps);
    }
}

void Interface2D::calculateTimeAveParameters(int substeps)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((PIC2DConst::nx_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (PIC2DConst::ny_PIC + threadsPerBlock.y - 1) / threadsPerBlock.y);

    calculateTimeAveParameters_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(B_timeAve.data()), 
        thrust::raw_pointer_cast(zerothMomentIon_timeAve.data()), 
        thrust::raw_pointer_cast(zerothMomentElectron_timeAve.data()), 
        thrust::raw_pointer_cast(firstMomentIon_timeAve.data()), 
        thrust::raw_pointer_cast(firstMomentElectron_timeAve.data()), 
        substeps
    );

    cudaDeviceSynchronize();
}

