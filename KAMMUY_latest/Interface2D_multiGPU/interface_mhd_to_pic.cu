#include "interface.hpp"


//PICのグリッドに合わせる
__global__ void sendMHDtoPIC_magneticField_y_kernel(
    const double* interlockingFunctionY, 
    const ConservationParameter* U, 
    MagneticField* B, 
    int indexOfInterfaceStartInMHD, 
    int localSizeXPIC
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < localSizeXPIC - 1 && 0 < j && j < PIC2DConst::device_ny - 1) {
        double bXPIC, bYPIC, bZPIC;
        double bXMHD, bYMHD, bZMHD;
        double bXInterface, bYInterface, bZInterface;

        int indexPIC = j + i * PIC2DConst::device_ny;
        int indexMHD = indexOfInterfaceStartInMHD + j + i * IdealMHD2DConst::device_ny;

        bXPIC = B[indexPIC].bX;
        bYPIC = B[indexPIC].bY;
        bZPIC = B[indexPIC].bZ;
        bXMHD = 0.5 * (U[indexMHD].bX + U[indexMHD + IdealMHD2DConst::device_ny].bX);
        bYMHD = 0.5 * (U[indexMHD].bY + U[indexMHD + 1].bY);
        bZMHD = U[indexMHD].bZ;

        bXInterface = interlockingFunctionY[indexPIC] * bXMHD + (1.0 - interlockingFunctionY[indexPIC]) * bXPIC;
        bYInterface = interlockingFunctionY[indexPIC] * bYMHD + (1.0 - interlockingFunctionY[indexPIC]) * bYPIC;
        bZInterface = interlockingFunctionY[indexPIC] * bZMHD + (1.0 - interlockingFunctionY[indexPIC]) * bZPIC;
        
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
    dim3 blocksPerGrid((mPIInfoPIC.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (PIC2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendMHDtoPIC_magneticField_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()),
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(B.data()), 
        indexOfInterfaceStartInMHD, 
        mPIInfoPIC.localSizeX
    );
    cudaDeviceSynchronize();
}


//PICのグリッドに合わせる
__global__ void sendMHDtoPIC_electricField_y_kernel(
    const double* interlockingFunctionY, 
    const ConservationParameter* U, 
    ElectricField* E, 
    int indexOfInterfaceStartInMHD, 
    int localSizeXPIC
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < localSizeXPIC - 1 && 0 < j && j < PIC2DConst::device_ny - 1) {
        double eXPIC, eYPIC, eZPIC;
        double rho, u, v, w, bX, bY, bZ;
        double rho_x1, u_x1, v_x1, w_x1, bX_x1, bY_x1, bZ_x1;
        double rho_y1, u_y1, v_y1, w_y1, bX_y1, bY_y1, bZ_y1;
        double rho_x1y1, u_x1y1, v_x1y1, bX_x1y1, bY_x1y1;
        double eXMHD, eYMHD, eZMHD;
        double eXMHD_y1, eYMHD_x1, eZMHD_x1, eZMHD_y1, eZMHD_x1y1; 
        double eXInterface, eYInterface, eZInterface;

        int indexPIC = j + i * PIC2DConst::device_ny;
        int indexMHD = indexOfInterfaceStartInMHD + j + i * IdealMHD2DConst::device_ny;

        eXPIC = E[indexPIC].eX;
        eYPIC = E[indexPIC].eY;
        eZPIC = E[indexPIC].eZ;

        rho = U[indexMHD].rho;
        u = U[indexMHD].rhoU / rho;
        v = U[indexMHD].rhoV / rho;
        w = U[indexMHD].rhoW / rho; 
        bX = U[indexMHD].bX; 
        bY = U[indexMHD].bY; 
        bZ = U[indexMHD].bZ;
        eXMHD = -(v * bZ - w * bY);
        eYMHD = -(w * bX - u * bZ);
        eZMHD = -(u * bY - v * bX);

        rho_x1 = U[indexMHD + IdealMHD2DConst::device_ny].rho;
        u_x1   = U[indexMHD + IdealMHD2DConst::device_ny].rhoU / rho_x1;
        v_x1   = U[indexMHD + IdealMHD2DConst::device_ny].rhoV / rho_x1;
        w_x1   = U[indexMHD + IdealMHD2DConst::device_ny].rhoW / rho_x1; 
        bX_x1  = U[indexMHD + IdealMHD2DConst::device_ny].bX; 
        bY_x1  = U[indexMHD + IdealMHD2DConst::device_ny].bY; 
        bZ_x1  = U[indexMHD + IdealMHD2DConst::device_ny].bZ;
        eYMHD_x1 = -(w_x1 * bX_x1 - u_x1 * bZ_x1);
        eZMHD_x1 = -(u_x1 * bY_x1 - v_x1 * bX_x1);

        rho_y1 = U[indexMHD + 1].rho;
        u_y1   = U[indexMHD + 1].rhoU / rho_y1;
        v_y1   = U[indexMHD + 1].rhoV / rho_y1;
        w_y1   = U[indexMHD + 1].rhoW / rho_y1; 
        bX_y1  = U[indexMHD + 1].bX; 
        bY_y1  = U[indexMHD + 1].bY; 
        bZ_y1  = U[indexMHD + 1].bZ;
        eXMHD_y1 = -(v_y1 * bZ_y1 - w_y1 * bY_y1);
        eZMHD_y1 = -(u_y1 * bY_y1 - v_y1 * bX_y1);

        rho_x1y1 = U[indexMHD + IdealMHD2DConst::device_ny + 1].rho;
        u_x1y1   = U[indexMHD + IdealMHD2DConst::device_ny + 1].rhoU / rho_x1y1;
        v_x1y1   = U[indexMHD + IdealMHD2DConst::device_ny + 1].rhoV / rho_x1y1;
        bX_x1y1  = U[indexMHD + IdealMHD2DConst::device_ny + 1].bX; 
        bY_x1y1  = U[indexMHD + IdealMHD2DConst::device_ny + 1].bY; 
        eZMHD_x1y1 = -(u_x1y1 * bY_x1y1 - v_x1y1 * bX_x1y1);

        eXMHD = 0.5 * (eXMHD + eXMHD_y1); 
        eYMHD = 0.5 * (eYMHD + eYMHD_x1); 
        eZMHD = 0.25 * (eZMHD + eZMHD_x1 + eZMHD_y1 + eZMHD_x1y1);

        eXInterface = interlockingFunctionY[indexPIC] * eXMHD + (1.0 - interlockingFunctionY[indexPIC]) * eXPIC;
        eYInterface = interlockingFunctionY[indexPIC] * eYMHD + (1.0 - interlockingFunctionY[indexPIC]) * eYPIC;
        eZInterface = interlockingFunctionY[indexPIC] * eZMHD + (1.0 - interlockingFunctionY[indexPIC]) * eZPIC;
         
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
    dim3 blocksPerGrid((mPIInfoPIC.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (PIC2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendMHDtoPIC_electricField_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()), 
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(E.data()), 
        indexOfInterfaceStartInMHD, 
        mPIInfoPIC.localSizeX
    );
    cudaDeviceSynchronize();
}


//PICのグリッドに合わせる
__global__ void sendMHDtoPIC_currentField_y_kernel(
    const double* interlockingFunctionY, 
    const ConservationParameter* U, 
    CurrentField* current, 
    int indexOfInterfaceStartInMHD, 
    int localSizeXPIC
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < localSizeXPIC - 1 && 0 < j && j < PIC2DConst::device_ny - 1) {
        double jXPIC, jYPIC, jZPIC;
        double jXMHD, jYMHD, jZMHD;
        double jZMHD_x1, jZMHD_y1, jZMHD_x1y1; 
        double jXInterface, jYInterface, jZInterface;
        double dx = IdealMHD2DConst::device_dx, dy = IdealMHD2DConst::device_dy;

        int indexPIC = j + i * PIC2DConst::device_ny;
        int indexMHD = indexOfInterfaceStartInMHD + j + i * IdealMHD2DConst::device_ny;

        jXPIC = current[indexPIC].jX;
        jYPIC = current[indexPIC].jY;
        jZPIC = current[indexPIC].jZ;

        jXMHD      = (U[indexMHD + 1].bZ - U[indexMHD].bZ) / dy;
        jYMHD      = -(U[indexMHD + IdealMHD2DConst::device_ny].bZ - U[indexMHD].bZ) / dx;
        jZMHD      = (U[indexMHD + IdealMHD2DConst::device_ny].bY - U[indexMHD - IdealMHD2DConst::device_ny].bY) / (2.0 * dx)
                   - (U[indexMHD + 1].bX - U[indexMHD - 1].bX) / (2.0 * dy);
        jZMHD_x1   = (U[indexMHD + 2 * IdealMHD2DConst::device_ny].bY - U[indexMHD].bY) / (2.0 * dx)
                   - (U[indexMHD + IdealMHD2DConst::device_ny + 1].bX - U[indexMHD + IdealMHD2DConst::device_ny - 1].bX) / (2.0 * dy);
        jZMHD_y1   = (U[indexMHD + IdealMHD2DConst::device_ny + 1].bY - U[indexMHD - IdealMHD2DConst::device_ny + 1].bY) / (2.0 * dx)
                   - (U[indexMHD + 2].bX - U[indexMHD].bX) / (2.0 * dy);
        jZMHD_x1y1 = (U[indexMHD + 2 * IdealMHD2DConst::device_ny + 1].bY - U[indexMHD + 1].bY) / (2.0 * dx)
                   - (U[indexMHD + IdealMHD2DConst::device_ny + 2].bX - U[indexMHD + IdealMHD2DConst::device_ny].bX) / (2.0 * dy);
        jZMHD      = 0.25 * (jZMHD + jZMHD_x1 + jZMHD_y1 + jZMHD_x1y1);
        
        jXInterface = interlockingFunctionY[indexPIC] * jXMHD + (1.0 - interlockingFunctionY[indexPIC]) * jXPIC;
        jYInterface = interlockingFunctionY[indexPIC] * jYMHD + (1.0 - interlockingFunctionY[indexPIC]) * jYPIC;
        jZInterface = interlockingFunctionY[indexPIC] * jZMHD + (1.0 - interlockingFunctionY[indexPIC]) * jZPIC;
        
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
    dim3 blocksPerGrid((mPIInfoPIC.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (PIC2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendMHDtoPIC_currentField_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()), 
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(current.data()), 
        indexOfInterfaceStartInMHD, 
        mPIInfoPIC.localSizeX
    );
    cudaDeviceSynchronize();
}


//ここでのPICの電流はモーメントから計算しているので、整数格子点上にあることに注意
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
    int localSizeXPIC
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < localSizeXPIC - 1 && 0 < j && j < PIC2DConst::device_ny - 1) {
        int indexForReload = j + i * PIC2DConst::device_ny;
        int indexPIC = j + i * PIC2DConst::device_ny;
        int indexMHD = indexOfInterfaceStartInMHD + j + i * IdealMHD2DConst::device_ny;
        double rhoMHD, uMHD, vMHD, wMHD, bXMHD, bYMHD, bZMHD, eMHD, pMHD;
        double jXMHD, jYMHD, jZMHD, niMHD, neMHD, tiMHD, teMHD;
        double rhoPIC, uPIC, vPIC, wPIC;
        double jXPIC, jYPIC, jZPIC, niPIC, nePIC, vThiPIC, vThePIC;
        double dx = IdealMHD2DConst::device_dx, dy = IdealMHD2DConst::device_dy;
        double mIon = PIC2DConst::device_mIon, mElectron = PIC2DConst::device_mElectron;
        double qIon = PIC2DConst::device_qIon, qElectron = PIC2DConst::device_qElectron;

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
        pMHD   = max(pMHD, IdealMHD2DConst::device_p0 * 0.1);
        jXMHD  = (U[indexMHD + 1].bZ - U[indexMHD - 1].bZ) / (2.0 * dy);
        jYMHD  = -(U[indexMHD + IdealMHD2DConst::device_ny].bZ - U[indexMHD - IdealMHD2DConst::device_ny].bZ) / (2.0 * dx);
        jZMHD  = (U[indexMHD + IdealMHD2DConst::device_ny].bY - U[indexMHD - IdealMHD2DConst::device_ny].bY) / (2.0 * dx)
               - (U[indexMHD + 1].bX - U[indexMHD - 1].bX) / (2.0 * dy) ;

        niMHD = rhoMHD / (mIon + mElectron);
        neMHD = niMHD;
        tiMHD = pMHD / 2.0 / niMHD;
        teMHD = pMHD / 2.0 / neMHD;

        rhoPIC =  mIon * zerothMomentIon[indexPIC].n + mElectron * zerothMomentElectron[indexPIC].n;
        uPIC   = (mIon * firstMomentIon[indexPIC].x  + mElectron * firstMomentElectron[indexPIC].x) / rhoPIC;
        vPIC   = (mIon * firstMomentIon[indexPIC].y  + mElectron * firstMomentElectron[indexPIC].y) / rhoPIC;
        wPIC   = (mIon * firstMomentIon[indexPIC].z  + mElectron * firstMomentElectron[indexPIC].z) / rhoPIC;
        jXPIC  =  qIon * firstMomentIon[indexPIC].x  + qElectron * firstMomentElectron[indexPIC].x;
        jYPIC  =  qIon * firstMomentIon[indexPIC].y  + qElectron * firstMomentElectron[indexPIC].y;
        jZPIC  =  qIon * firstMomentIon[indexPIC].z  + qElectron * firstMomentElectron[indexPIC].z;

        rhoPIC = interlockingFunctionY[indexPIC] * rhoMHD + (1.0 - interlockingFunctionY[indexPIC]) * rhoPIC;
        uPIC   = interlockingFunctionY[indexPIC] * uMHD   + (1.0 - interlockingFunctionY[indexPIC]) * uPIC;
        vPIC   = interlockingFunctionY[indexPIC] * vMHD   + (1.0 - interlockingFunctionY[indexPIC]) * vPIC;
        wPIC   = interlockingFunctionY[indexPIC] * wMHD   + (1.0 - interlockingFunctionY[indexPIC]) * wPIC;
        jXPIC  = interlockingFunctionY[indexPIC] * jXMHD  + (1.0 - interlockingFunctionY[indexPIC]) * jXPIC;
        jYPIC  = interlockingFunctionY[indexPIC] * jYMHD  + (1.0 - interlockingFunctionY[indexPIC]) * jYPIC;
        jZPIC  = interlockingFunctionY[indexPIC] * jZMHD  + (1.0 - interlockingFunctionY[indexPIC]) * jZPIC;

        niPIC   = rhoPIC / (mIon + mElectron);
        nePIC   = niPIC;
        vThiPIC = sqrt(2.0 * tiMHD / mIon);
        vThePIC = sqrt(2.0 * teMHD / mElectron);

        reloadParticlesDataIon     [indexForReload].numAndIndex = round(niPIC);
        reloadParticlesDataElectron[indexForReload].numAndIndex = round(nePIC);
        reloadParticlesDataIon     [indexForReload].u           = uPIC;
        reloadParticlesDataIon     [indexForReload].v           = vPIC;
        reloadParticlesDataIon     [indexForReload].w           = wPIC;
        reloadParticlesDataElectron[indexForReload].u           = uPIC - jXPIC / round(nePIC) / abs(qElectron); //niPIC = nePIC
        reloadParticlesDataElectron[indexForReload].v           = vPIC - jYPIC / round(nePIC) / abs(qElectron); //niPIC = nePIC
        reloadParticlesDataElectron[indexForReload].w           = wPIC - jZPIC / round(nePIC) / abs(qElectron); //niPIC = nePIC
        reloadParticlesDataIon     [indexForReload].vth         = vThiPIC;
        reloadParticlesDataElectron[indexForReload].vth         = vThePIC;

        if (j == 1) {
            reloadParticlesDataIon[indexForReload - 1]      = reloadParticlesDataIon[indexForReload];
            reloadParticlesDataElectron[indexForReload - 1] = reloadParticlesDataElectron[indexForReload];
        }
        if (j == PIC2DConst::device_ny - 2) {
            reloadParticlesDataIon[indexForReload + 1]      = reloadParticlesDataIon[indexForReload];
            reloadParticlesDataElectron[indexForReload + 1] = reloadParticlesDataElectron[indexForReload];
        }
    }
}


__global__ void deleteParticles_kernel(
    const double* interlockingFunctionY, 
    Particle* particlesSpecies, 
    const unsigned long long existNumSpecies, 
    int seed, 
    const float xminForProcs, const float xmaxForProcs, 
    const int buffer
)
{
    unsigned long long k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < existNumSpecies) {
        float x = particlesSpecies[k].x;
        float y = particlesSpecies[k].y;
        float xmin = xminForProcs - buffer * PIC2DConst::device_dx;
        float ymin = PIC2DConst::device_ymin;

        int i = floorf(x - xmin); 
        int j = floorf(y - ymin);
        curandState state; 
        curand_init(seed, k, 0, &state);
        float randomValue = curand_uniform(&state);
        if (randomValue < interlockingFunctionY[j + i * PIC2DConst::device_ny]) {
            particlesSpecies[k].isExist = false;
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
        existNumSpeciesPerProcs, 
        seed, 
        mPIInfoPIC.xminForProcs, mPIInfoPIC.xmaxForProcs, 
        mPIInfoPIC.buffer
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
    const Particle* reloadParticlesSourceSpecies, 
    unsigned long long reloadParticlesTotalNumSpecies, 
    Particle* particlesSpecies, 
    unsigned long long* particlesNumCounter, 
    int seed, 
    const float xminForProcs, const float xmaxForProcs,
    int buffer, 
    int localSizeXPIC
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localSizeXPIC && j < PIC2DConst::device_ny) {
        int indexPIC = j + i * PIC2DConst::device_ny; 
        float u = reloadParticlesDataSpecies[indexPIC].u;
        float v = reloadParticlesDataSpecies[indexPIC].v;
        float w = reloadParticlesDataSpecies[indexPIC].w;
        float vth = reloadParticlesDataSpecies[indexPIC].vth;
        Particle particleSource, particleReload;
        float x, y, z, vx, vy, vz, gamma;

        curandState stateReloadIndex, stateReload; 
        curand_init(seed, i * j, 0, &stateReloadIndex);
        unsigned long long restartParticlesIndexSpecies = static_cast<unsigned long long>(
            curand_uniform(&stateReloadIndex) * reloadParticlesTotalNumSpecies
        );

        for (unsigned long long k = 0; k < reloadParticlesDataSpecies[indexPIC].numAndIndex; k++) {
            curand_init(seed + i * j, k, 0, &stateReload);
            float randomValue = curand_uniform(&stateReload);

            if (randomValue < interlockingFunctionY[indexPIC]) {
                particleSource = reloadParticlesSourceSpecies[
                    (restartParticlesIndexSpecies + k) % reloadParticlesTotalNumSpecies
                ];

                x = particleSource.x; 
                x = (x + i) * PIC2DConst::device_dx + (xminForProcs - buffer * PIC2DConst::device_dx);
                y = particleSource.y; 
                y = (y + j) * PIC2DConst::device_dy + PIC2DConst::device_ymin;
                z = particleSource.z;
                
                vx = particleSource.vx; vx = u + vx * vth;
                vy = particleSource.vy; vy = v + vy * vth;
                vz = particleSource.vz; vz = w + vz * vth;
                if (1.0f - (vx * vx + vy * vy + vz * vz) / pow(PIC2DConst::device_c, 2) < 0.0f){
                    float normalizedVelocity = sqrt(vx * vx + vy * vy + vz * vz);
                    vx = vx / normalizedVelocity * 0.9f * PIC2DConst::device_c;
                    vy = vy / normalizedVelocity * 0.9f * PIC2DConst::device_c;
                    vz = vz / normalizedVelocity * 0.9f * PIC2DConst::device_c;
                };
                gamma = 1.0f / sqrt(1.0f - (vx * vx + vy * vy + vz * vz) / pow(PIC2DConst::device_c, 2));

                particleReload.x = x; particleReload.y = y; particleReload.z = z;
                particleReload.vx = vx * gamma; particleReload.vy = vy * gamma, particleReload.vz = vz * gamma; 
                particleReload.gamma = gamma;
                particleReload.isExist = true;

                unsigned long long loadIndex = atomicAdd(&(particlesNumCounter[0]), 1);
                particlesSpecies[loadIndex] = particleReload;
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
    thrust::device_vector<unsigned long long> particlesNumCounter(1, 0);
    particlesNumCounter[0] = existNumSpeciesPerProcs;

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfoPIC.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (PIC2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    reloadParticlesSpecies_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()), 
        thrust::raw_pointer_cast(reloadParticlesDataSpecies.data()), 
        thrust::raw_pointer_cast(reloadParticlesSourceSpecies.data()), 
        Interface2DConst::reloadParticlesTotalNum, 
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        thrust::raw_pointer_cast(particlesNumCounter.data()), 
        seed, 
        mPIInfoPIC.xminForProcs, mPIInfoPIC.xmaxForProcs, 
        mPIInfoPIC.buffer, 
        mPIInfoPIC.localSizeX
    );
    cudaDeviceSynchronize();

    existNumSpeciesPerProcs = particlesNumCounter[0];
}


void Interface2D::sendMHDtoPIC_particle(
    const thrust::device_vector<ConservationParameter>& U, 
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron, 
    int seed
)
{
    setMoments(particlesIon, particlesElectron); 
    boundaryPIC.periodicBoundaryZerothMoment_x(zerothMomentIon); 
    boundaryPIC.freeBoundaryZerothMoment_y(zerothMomentIon); 
    boundaryPIC.periodicBoundaryZerothMoment_x(zerothMomentElectron); 
    boundaryPIC.freeBoundaryZerothMoment_y(zerothMomentElectron); 
    boundaryPIC.periodicBoundaryFirstMoment_x(firstMomentIon); 
    boundaryPIC.freeBoundaryFirstMoment_y(firstMomentIon); 
    boundaryPIC.periodicBoundaryFirstMoment_x(firstMomentElectron); 
    boundaryPIC.freeBoundaryFirstMoment_y(firstMomentElectron); 

    thrust::fill(reloadParticlesDataIon.begin(), reloadParticlesDataIon.end(), ReloadParticlesData());
    thrust::fill(reloadParticlesDataElectron.begin(), reloadParticlesDataElectron.end(), ReloadParticlesData());

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfoPIC.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (PIC2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

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
        mPIInfoPIC.localSizeX
    );
    cudaDeviceSynchronize();

    Interface2DMPI::sendrecv_reloadParticlesData_x(reloadParticlesDataIon, mPIInfoInterface);
    Interface2DMPI::sendrecv_reloadParticlesData_x(reloadParticlesDataElectron, mPIInfoInterface);
    
    deleteParticlesSpecies(
        particlesIon, mPIInfoPIC.existNumIonPerProcs, seed + 100
    );
    deleteParticlesSpecies(
        particlesElectron, mPIInfoPIC.existNumElectronPerProcs, seed + 200
    );

    reloadParticlesSpecies(
        particlesIon, reloadParticlesDataIon, reloadParticlesSourceIon, 
        mPIInfoPIC.existNumIonPerProcs, seed + 300
    ); 
    reloadParticlesSpecies(
        particlesElectron, reloadParticlesDataElectron, reloadParticlesSourceElectron, 
        mPIInfoPIC.existNumElectronPerProcs, seed + 400
    ); 
}


