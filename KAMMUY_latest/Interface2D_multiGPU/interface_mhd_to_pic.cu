#include "interface.hpp"


__global__ void sendMHDtoPIC_magneticField_y_kernel(
    const double* interlockingFunctionY, 
    const ConservationParameter* U, 
    MagneticField* B, 
    const int indexOfInterfaceStartInMHD, 
    const int localNxPIC, const int bufferPIC, const int bufferMHD
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localNxPIC && j < PIC2DConst::device_ny) {
        double bXPIC, bYPIC, bZPIC;
        double bXMHD, bYMHD, bZMHD;
        double bXInterface, bYInterface, bZInterface;

        int indexPIC = j + (i + bufferPIC) * PIC2DConst::device_ny;
        int indexMHD = indexOfInterfaceStartInMHD + static_cast<int>(j / Interface2DConst::device_gridSizeRatio)
                     + (static_cast<int>(i / Interface2DConst::device_gridSizeRatio) + bufferMHD) * IdealMHD2DConst::device_ny;

        bXPIC = B[indexPIC].bX;
        bYPIC = B[indexPIC].bY;
        bZPIC = B[indexPIC].bZ;
        bXMHD = U[indexMHD].bX;
        bYMHD = U[indexMHD].bY;
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
    dim3 blocksPerGrid((mPIInfoPIC.localNx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (PIC2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendMHDtoPIC_magneticField_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()),
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(B.data()), 
        indexOfInterfaceStartInMHD, 
        mPIInfoPIC.localNx, mPIInfoPIC.buffer, mPIInfoMHD.buffer
    );
    cudaDeviceSynchronize();
}


__global__ void sendMHDtoPIC_electricField_y_kernel(
    const double* interlockingFunctionY, 
    const ConservationParameter* U, 
    ElectricField* E, 
    const int indexOfInterfaceStartInMHD, 
    const int localNxPIC, const int bufferPIC, const int bufferMHD
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localNxPIC && j < PIC2DConst::device_ny) {
        double eXPIC, eYPIC, eZPIC;
        double rho, u, v, w, bX, bY, bZ;
        double eXMHD, eYMHD, eZMHD;
        double eXInterface, eYInterface, eZInterface;

        int indexPIC = j + (i + bufferPIC) * PIC2DConst::device_ny;
        int indexMHD = indexOfInterfaceStartInMHD + static_cast<int>(j / Interface2DConst::device_gridSizeRatio)
                     + (static_cast<int>(i / Interface2DConst::device_gridSizeRatio) + bufferMHD) * IdealMHD2DConst::device_ny;

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
    dim3 blocksPerGrid((mPIInfoPIC.localNx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (PIC2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendMHDtoPIC_electricField_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()), 
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(E.data()), 
        indexOfInterfaceStartInMHD, 
        mPIInfoPIC.localNx, mPIInfoPIC.buffer, mPIInfoMHD.buffer
    );
    cudaDeviceSynchronize();
}



__global__ void sendMHDtoPIC_currentField_y_kernel(
    const double* interlockingFunctionY, 
    const ConservationParameter* U, 
    CurrentField* current, 
    const int indexOfInterfaceStartInMHD, 
    const int localNxPIC, const int bufferPIC, const int bufferMHD
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < localNxPIC - 1 && 0 < j && j < PIC2DConst::device_ny - 1) {
        double jXPIC, jYPIC, jZPIC;
        double jXMHD, jYMHD, jZMHD;
        double jXInterface, jYInterface, jZInterface;
        double dx = IdealMHD2DConst::device_dx, dy = IdealMHD2DConst::device_dy;

        int indexPIC = j + (i + bufferPIC) * PIC2DConst::device_ny;
        int indexMHD = indexOfInterfaceStartInMHD + static_cast<int>(j / Interface2DConst::device_gridSizeRatio)
                     + (static_cast<int>(i / Interface2DConst::device_gridSizeRatio) + bufferMHD) * IdealMHD2DConst::device_ny;

        jXPIC = current[indexPIC].jX;
        jYPIC = current[indexPIC].jY;
        jZPIC = current[indexPIC].jZ;

        jXMHD      = (U[indexMHD + 1].bZ - U[indexMHD].bZ) / dy;
        jYMHD      = -(U[indexMHD + IdealMHD2DConst::device_ny].bZ - U[indexMHD].bZ) / dx;
        jZMHD      = (U[indexMHD + IdealMHD2DConst::device_ny].bY - U[indexMHD - IdealMHD2DConst::device_ny].bY) / (2.0 * dx)
                   - (U[indexMHD + 1].bX - U[indexMHD - 1].bX) / (2.0 * dy);
        
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
    dim3 blocksPerGrid((mPIInfoPIC.localNx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (PIC2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendMHDtoPIC_currentField_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()), 
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(current.data()), 
        indexOfInterfaceStartInMHD, 
        mPIInfoPIC.localNx, mPIInfoPIC.buffer, mPIInfoMHD.buffer
    );
    cudaDeviceSynchronize();
}


__global__ void deleteParticles_kernel(
    const double* interlockingFunctionY, 
    Particle* particlesSpecies, 
    const unsigned long long existNumSpecies, 
    const unsigned long long seed, 
    const float xminForProcs, const float xmaxForProcs, 
    const int bufferPIC
)
{
    unsigned long long k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < existNumSpecies) {
        float x = particlesSpecies[k].x;
        float y = particlesSpecies[k].y;
        float xmin = xminForProcs - bufferPIC * PIC2DConst::device_dx;
        float ymin = PIC2DConst::device_ymin;

        int i = floorf(x - xmin); 
        int j = floorf(y - ymin);
        if (i < bufferPIC || i >= PIC2DConst::device_nx + bufferPIC) {
            particlesSpecies[k].isExist = false; 
            return; 
        }
        
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
    unsigned long long seed
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

    existNumSpeciesPerProcs = static_cast<unsigned long long>(thrust::distance(particlesSpecies.begin(), partitionEnd));
}


__global__ void reloadParticlesSpecies_kernel(
    const double* interlockingFunctionY, 
    const ReloadParticlesData* reloadParticlesDataSpecies, 
    const Particle* reloadParticlesSourceSpecies, 
    unsigned long long reloadParticlesTotalNumSpecies, 
    Particle* particlesSpecies, 
    unsigned long long* particlesNumCounter, 
    const unsigned long long seed, 
    const float xminForProcs, const float xmaxForProcs, 
    const int localNxPIC, const int bufferPIC
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localNxPIC && j < PIC2DConst::device_ny) {
        int indexPIC = j + (i + bufferPIC) * PIC2DConst::device_ny; 
        int indexForReload = j + i * PIC2DConst::device_ny; 
        float u = reloadParticlesDataSpecies[indexForReload].u;
        float v = reloadParticlesDataSpecies[indexForReload].v;
        float w = reloadParticlesDataSpecies[indexForReload].w;
        float vth = reloadParticlesDataSpecies[indexForReload].vth;

        Particle particleSource, particleReload;
        float x, y, z, vx, vy, vz, gamma;

        curandState stateForReloadIndex;
        curand_init(seed, indexForReload, 0, &stateForReloadIndex);
        unsigned long long restartParticlesIndexSpecies = static_cast<unsigned long long>(
            curand_uniform(&stateForReloadIndex) * reloadParticlesTotalNumSpecies
        );

        curandState stateForReload; 
        curand_init(seed, indexForReload, 0, &stateForReload);
        for (unsigned long long k = 0; k < reloadParticlesDataSpecies[indexForReload].numAndIndex; k++) {
            float randomValue = curand_uniform(&stateForReload);

            if (randomValue < interlockingFunctionY[indexPIC]) {
                particleSource = reloadParticlesSourceSpecies[
                    (restartParticlesIndexSpecies + k) % reloadParticlesTotalNumSpecies
                ];

                x = particleSource.x; 
                x = (x + i) * PIC2DConst::device_dx + xminForProcs;
                y = particleSource.y; 
                y = (y + j) * PIC2DConst::device_dy + PIC2DConst::device_ymin;
                z = particleSource.z;
                
                vx = particleSource.vx; vx = u + vx * vth;
                vy = particleSource.vy; vy = v + vy * vth;
                vz = particleSource.vz; vz = w + vz * vth;
                if (1.0f - (vx * vx + vy * vy + vz * vz) / pow(PIC2DConst::device_c, 2) < 0.0f){
                    float normalizedVelocity = sqrt(vx * vx + vy * vy + vz * vz);
                    vx = vx / normalizedVelocity * 0.99f * PIC2DConst::device_c;
                    vy = vy / normalizedVelocity * 0.99f * PIC2DConst::device_c;
                    vz = vz / normalizedVelocity * 0.99f * PIC2DConst::device_c;
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
    unsigned long long seed 
)
{
    thrust::device_vector<unsigned long long> particlesNumCounter(1, 0);
    particlesNumCounter[0] = existNumSpeciesPerProcs;

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfoPIC.localNx + threadsPerBlock.x - 1) / threadsPerBlock.x,
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
        mPIInfoPIC.localNx, mPIInfoPIC.buffer
    );
    cudaDeviceSynchronize();

    existNumSpeciesPerProcs = particlesNumCounter[0];
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
    const int indexOfInterfaceStartInMHD, 
    const int localNxPIC, const int bufferPIC, const int bufferMHD
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localNxPIC && j < PIC2DConst::device_ny) {
        int indexForReload = j + i * PIC2DConst::device_ny;
        int indexPIC = j + (i + bufferPIC) * PIC2DConst::device_ny;
        int indexMHD = indexOfInterfaceStartInMHD + static_cast<int>(j / Interface2DConst::device_gridSizeRatio)
                     + (static_cast<int>(i / Interface2DConst::device_gridSizeRatio) + bufferMHD) * IdealMHD2DConst::device_ny;
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
        jXMHD  = (U[indexMHD + 1].bZ - U[indexMHD - 1].bZ) / (2.0 * dy);
        jYMHD  = -(U[indexMHD + IdealMHD2DConst::device_ny].bZ - U[indexMHD - IdealMHD2DConst::device_ny].bZ) / (2.0 * dx);
        jZMHD  = (U[indexMHD + IdealMHD2DConst::device_ny].bY - U[indexMHD - IdealMHD2DConst::device_ny].bY) / (2.0 * dx)
               - (U[indexMHD + 1].bX - U[indexMHD - 1].bX) / (2.0 * dy);

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
    }
}


void Interface2D::sendMHDtoPIC_particle(
    const thrust::device_vector<ConservationParameter>& U, 
    const thrust::device_vector<ZerothMoment>& zerothMomentIon, 
    const thrust::device_vector<ZerothMoment>& zerothMomentElectron, 
    const thrust::device_vector<FirstMoment>& firstMomentIon, 
    const thrust::device_vector<FirstMoment>& firstMomentElectron, 
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron, 
    unsigned long long seed
)
{
    thrust::fill(reloadParticlesDataIon.begin(), reloadParticlesDataIon.end(), ReloadParticlesData());
    thrust::fill(reloadParticlesDataElectron.begin(), reloadParticlesDataElectron.end(), ReloadParticlesData());

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfoPIC.localNx + threadsPerBlock.x - 1) / threadsPerBlock.x,
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
        mPIInfoPIC.localNx, mPIInfoPIC.buffer, mPIInfoMHD.buffer
    );
    cudaDeviceSynchronize();

    //Interface2DMPI::sendrecv_reloadParticlesData_x(reloadParticlesDataIon, mPIInfoInterface);
    //Interface2DMPI::sendrecv_reloadParticlesData_x(reloadParticlesDataElectron, mPIInfoInterface);
    
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

    if (mPIInfoPIC.existNumIonPerProcs > mPIInfoPIC.totalNumIonPerProcs) std::cout << "BROKEN" << std::endl;
    if (mPIInfoPIC.existNumElectronPerProcs > mPIInfoPIC.totalNumElectronPerProcs) std::cout << "BROKEN" << std::endl;
}


