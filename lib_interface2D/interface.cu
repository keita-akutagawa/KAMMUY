#include "interface.hpp"
#include <cmath>
#include <curand_kernel.h>


using namespace Interface2DConst;

Interface2D::Interface2D(
    int indexStartMHD, 
    int indexStartPIC
)
    :  indexOfInterfaceStartInMHD(indexStartMHD), 
       indexOfInterfaceStartInPIC(indexStartPIC), 
       interlockingFunctionY(interfaceLength), 
       interlockingFunctionYHalf(interfaceLength - 1),
       host_interlockingFunctionY(interfaceLength), 
       host_interlockingFunctionYHalf(interfaceLength - 1),

       zerothMomentIon(PIC2DConst::nx * PIC2DConst::ny), 
       zerothMomentElectron(PIC2DConst::nx * PIC2DConst::ny), 
       firstMomentIon(PIC2DConst::nx * PIC2DConst::ny), 
       firstMomentElectron(PIC2DConst::nx * PIC2DConst::ny),

       reloadParticlesDataIon(PIC2DConst::nx * interfaceLength), 
       reloadParticlesDataElectron(PIC2DConst::nx * interfaceLength), 

       reloadParticlesIon(PIC2DConst::numberDensityIon * PIC2DConst::nx * (Interface2DConst::interfaceLength + 100)), 
       reloadParticlesElectron(PIC2DConst::numberDensityElectron * PIC2DConst::nx * (Interface2DConst::interfaceLength + 100))
{
    indexOfInterfaceEndInMHD = indexOfInterfaceStartInMHD + Interface2DConst::interfaceLength;
    indexOfInterfaceEndInPIC = indexOfInterfaceStartInPIC + Interface2DConst::interfaceLength;

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

    }
}

__global__ void reloadParticles_kernel(
    thrust::device_vector<Particle>& particlesSpecies
)
{

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
        indexOfInterfaceStartInMHD, 
        indexOfInterfaceStartInPIC 
    );

    cudaDeviceSynchronize();



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
