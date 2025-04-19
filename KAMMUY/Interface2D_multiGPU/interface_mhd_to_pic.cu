#include "interface.hpp"


struct MagneticField_MHD {
    double bX;
    double bY;
    double bZ;
};

struct ElectricField_MHD {
    double eX;
    double eY;
    double eZ;
};

struct CurrentField_MHD {
    double jX;
    double jY;
    double jZ;
};


__device__ MagneticField_MHD getMagneticField_MHD(
    const ConservationParameter U
)
{
    MagneticField_MHD B_MHD; 

    B_MHD.bX = U.bX; 
    B_MHD.bY = U.bY; 
    B_MHD.bZ = U.bZ;

    return B_MHD; 
}

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
        int indexPIC = j + (i + bufferPIC) * PIC2DConst::device_ny;
        
        bXPIC = B[indexPIC].bX;
        bYPIC = B[indexPIC].bY;
        bZPIC = B[indexPIC].bZ;
        

        double bXMHD, bYMHD, bZMHD;
        int indexMHD = indexOfInterfaceStartInMHD + static_cast<int>(j / Interface2DConst::device_gridSizeRatio)
                     + (static_cast<int>(i / Interface2DConst::device_gridSizeRatio) + bufferMHD) * IdealMHD2DConst::device_ny;
        double cx1, cx2, cy1, cy2;  

        MagneticField_MHD B_MHD_x1y1 = getMagneticField_MHD(U[indexMHD]);
        MagneticField_MHD B_MHD_x2y1 = getMagneticField_MHD(U[indexMHD + IdealMHD2DConst::device_ny]);
        MagneticField_MHD B_MHD_x1y2 = getMagneticField_MHD(U[indexMHD + 1]);
        MagneticField_MHD B_MHD_x2y2 = getMagneticField_MHD(U[indexMHD + IdealMHD2DConst::device_ny + 1]);
        
        cx1 = static_cast<double>((i % Interface2DConst::device_gridSizeRatio)) / Interface2DConst::device_gridSizeRatio; 
        cx2 = 1.0 - cx1;
        cy1 = static_cast<double>(((j % Interface2DConst::device_gridSizeRatio) + 0.5)) / Interface2DConst::device_gridSizeRatio;
        cy2 = 1.0 - cy1; 
        bXMHD = B_MHD_x1y1.bX * cx2 * cy2 + B_MHD_x2y1.bX * cx1 * cy2 + B_MHD_x1y2.bX * cx2 * cy1 + B_MHD_x2y2.bX * cx1 * cy1;

        cx1 = static_cast<double>(((i % Interface2DConst::device_gridSizeRatio) + 0.5)) / Interface2DConst::device_gridSizeRatio; 
        cx2 = 1.0 - cx1;
        cy1 = static_cast<double>((j % Interface2DConst::device_gridSizeRatio)) / Interface2DConst::device_gridSizeRatio;
        cy2 = 1.0 - cy1; 
        bYMHD = B_MHD_x1y1.bY * cx2 * cy2 + B_MHD_x2y1.bY * cx1 * cy2 + B_MHD_x1y2.bY * cx2 * cy1 + B_MHD_x2y2.bY * cx1 * cy1;

        cx1 = static_cast<double>(((i % Interface2DConst::device_gridSizeRatio) + 0.5)) / Interface2DConst::device_gridSizeRatio; 
        cx2 = 1.0 - cx1;
        cy1 = static_cast<double>(((j % Interface2DConst::device_gridSizeRatio) + 0.5)) / Interface2DConst::device_gridSizeRatio;
        cy2 = 1.0 - cy1; 
        bZMHD = B_MHD_x1y1.bZ * cx2 * cy2 + B_MHD_x2y1.bZ * cx1 * cy2 + B_MHD_x1y2.bZ * cx2 * cy1 + B_MHD_x2y2.bZ * cx1 * cy1;
        
        
        double bXInterface, bYInterface, bZInterface;

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


__device__ ElectricField_MHD getElectricField_MHD(
    const ConservationParameter U
)
{
    double rho, u, v, w, bX, bY, bZ;
    ElectricField_MHD E_MHD; 

    rho = U.rho;
    u   = U.rhoU / rho;
    v   = U.rhoV / rho;
    w   = U.rhoW / rho; 
    bX  = U.bX; 
    bY  = U.bY; 
    bZ  = U.bZ;
    E_MHD.eX = -(v * bZ - w * bY);
    E_MHD.eY = -(w * bX - u * bZ);
    E_MHD.eZ = -(u * bY - v * bX);

    return E_MHD; 
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
        int indexPIC = j + (i + bufferPIC) * PIC2DConst::device_ny;

        eXPIC = E[indexPIC].eX;
        eYPIC = E[indexPIC].eY;
        eZPIC = E[indexPIC].eZ;


        double eXMHD, eYMHD, eZMHD;
        int indexMHD = indexOfInterfaceStartInMHD + static_cast<int>(j / Interface2DConst::device_gridSizeRatio)
                     + (static_cast<int>(i / Interface2DConst::device_gridSizeRatio) + bufferMHD) * IdealMHD2DConst::device_ny;
        double cx1, cx2, cy1, cy2;  

        ElectricField_MHD E_MHD_x1y1 = getElectricField_MHD(U[indexMHD]);
        ElectricField_MHD E_MHD_x2y1 = getElectricField_MHD(U[indexMHD + IdealMHD2DConst::device_ny]);
        ElectricField_MHD E_MHD_x1y2 = getElectricField_MHD(U[indexMHD + 1]);
        ElectricField_MHD E_MHD_x2y2 = getElectricField_MHD(U[indexMHD + IdealMHD2DConst::device_ny + 1]);
        
        cx1 = static_cast<double>(((i % Interface2DConst::device_gridSizeRatio) + 0.5)) / Interface2DConst::device_gridSizeRatio; 
        cx2 = 1.0 - cx1;
        cy1 = static_cast<double>((j % Interface2DConst::device_gridSizeRatio)) / Interface2DConst::device_gridSizeRatio;
        cy2 = 1.0 - cy1;  
        eXMHD = E_MHD_x1y1.eX * cx2 * cy2 + E_MHD_x2y1.eX * cx1 * cy2 + E_MHD_x1y2.eX * cx2 * cy1 + E_MHD_x2y2.eX * cx1 * cy1;
        
        cx1 = static_cast<double>((i % Interface2DConst::device_gridSizeRatio)) / Interface2DConst::device_gridSizeRatio; 
        cx2 = 1.0 - cx1;
        cy1 = static_cast<double>(((j % Interface2DConst::device_gridSizeRatio) + 0.5)) / Interface2DConst::device_gridSizeRatio;
        cy2 = 1.0 - cy1; 
        eYMHD = E_MHD_x1y1.eY * cx2 * cy2 + E_MHD_x2y1.eY * cx1 * cy2 + E_MHD_x1y2.eY * cx2 * cy1 + E_MHD_x2y2.eY * cx1 * cy1;
        
        cx1 = static_cast<double>((i % Interface2DConst::device_gridSizeRatio)) / Interface2DConst::device_gridSizeRatio; 
        cx2 = 1.0 - cx1;
        cy1 = static_cast<double>((j % Interface2DConst::device_gridSizeRatio)) / Interface2DConst::device_gridSizeRatio;
        cy2 = 1.0 - cy1; 
        eZMHD = E_MHD_x1y1.eZ * cx2 * cy2 + E_MHD_x2y1.eZ * cx1 * cy2 + E_MHD_x1y2.eZ * cx2 * cy1 + E_MHD_x2y2.eZ * cx1 * cy1;
        
        
        double eXInterface, eYInterface, eZInterface;

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


__device__ CurrentField_MHD getCurrentField_MHD(
    const ConservationParameter* U, int indexMHD
)
{
    CurrentField_MHD J_MHD; 

    J_MHD.jX = (U[indexMHD + 1].bZ - U[indexMHD - 1].bZ)
             / (2.0 * IdealMHD2DConst::device_dy);
    J_MHD.jY = -(U[indexMHD + IdealMHD2DConst::device_ny].bZ - U[indexMHD - IdealMHD2DConst::device_ny].bZ)
             / (2.0 * IdealMHD2DConst::device_dx);
    J_MHD.jZ = (U[indexMHD + IdealMHD2DConst::device_ny].bY - U[indexMHD - IdealMHD2DConst::device_ny].bY)
             / (2.0 * IdealMHD2DConst::device_dx)
             - (U[indexMHD + 1].bX - U[indexMHD - 1].bX)
             / (2.0 * IdealMHD2DConst::device_dy);

    return J_MHD; 
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

    if (i < localNxPIC && j < PIC2DConst::device_ny) {
        double jXPIC, jYPIC, jZPIC;
        int indexPIC = j + (i + bufferPIC) * PIC2DConst::device_ny;

        jXPIC = current[indexPIC].jX;
        jYPIC = current[indexPIC].jY;
        jZPIC = current[indexPIC].jZ;


        double jXMHD, jYMHD, jZMHD;
        int indexMHD = indexOfInterfaceStartInMHD + static_cast<int>(j / Interface2DConst::device_gridSizeRatio)
                     + (static_cast<int>(i / Interface2DConst::device_gridSizeRatio) + bufferMHD) * IdealMHD2DConst::device_ny;
        double cx1, cx2, cy1, cy2;  

        CurrentField_MHD J_MHD_x1y1 = getCurrentField_MHD(U, indexMHD);
        CurrentField_MHD J_MHD_x2y1 = getCurrentField_MHD(U, indexMHD + Ideal2DConst::device_ny);
        CurrentField_MHD J_MHD_x1y2 = getCurrentField_MHD(U, indexMHD + 1);
        CurrentField_MHD J_MHD_x2y2 = getCurrentField_MHD(U, indexMHD + Ideal2DConst::device_ny + 1);
        
        cx1 = static_cast<double>(((i % Interface2DConst::device_gridSizeRatio) + 0.5)) / Interface2DConst::device_gridSizeRatio; 
        cx2 = 1.0 - cx1;
        cy1 = static_cast<double>((j % Interface2DConst::device_gridSizeRatio)) / Interface2DConst::device_gridSizeRatio;
        cy2 = 1.0 - cy1;
        jXMHD = J_MHD_x1y1.jX * cx2 * cy2 + J_MHD_x2y1.jX * cx1 * cy2 + J_MHD_x1y2.jX * cx2 * cy1 + J_MHD_x2y2.jX * cx1 * cy1;
        
        cx1 = static_cast<double>((i % Interface2DConst::device_gridSizeRatio)) / Interface2DConst::device_gridSizeRatio; 
        cx2 = 1.0 - cx1;
        cy1 = static_cast<double>(((j % Interface2DConst::device_gridSizeRatio) + 0.5)) / Interface2DConst::device_gridSizeRatio;
        cy2 = 1.0 - cy1; 
        jYMHD = J_MHD_x1y1.jY * cx2 * cy2 + J_MHD_x2y1.jY * cx1 * cy2 + J_MHD_x1y2.jY * cx2 * cy1 + J_MHD_x2y2.jY * cx1 * cy1;
        
        cx1 = static_cast<double>((i % Interface2DConst::device_gridSizeRatio)) / Interface2DConst::device_gridSizeRatio; 
        cx2 = 1.0 - cx1;
        cy1 = static_cast<double>((j % Interface2DConst::device_gridSizeRatio)) / Interface2DConst::device_gridSizeRatio;
        cy2 = 1.0 - cy1; 
        jZMHD = J_MHD_x1y1.jZ * cx2 * cy2 + J_MHD_x2y1.jZ * cx1 * cy2 + J_MHD_x1y2.jZ * cx2 * cy1 + J_MHD_x2y2.jZ * cx1 * cy1;

        
        double jXInterface, jYInterface, jZInterface;
        
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
        float xmin = xminForProcs;
        float ymin = PIC2DConst::device_ymin;

        int i = floorf(x - xmin); 
        int j = floorf(y - ymin);
        if (i < 0 || i >= PIC2DConst::device_nx) {
            particlesSpecies[k].isExist = false; 
            return; 
        }

        if (interlockingFunctionY[j + (i + bufferPIC) * PIC2DConst::device_ny] < Interface2DConst::device_EPS) return;
        
        curandState state; 
        curand_init(seed, k, 0, &state);
        float randomValue = curand_uniform(&state);
        if (randomValue < interlockingFunctionY[j + (i + bufferPIC) * PIC2DConst::device_ny]) {
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

        if (interlockingFunctionY[indexPIC] < Interface2DConst::device_EPS) return;

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


template <typename MomentType>
__device__ MomentType getConvolvedMomentForMHDtoPIC(
    const MomentType* moment, 
    int indexPIC, 
    int j
)
{
    MomentType convolvedMoment; 

    if (1 <= j && j <= PIC2DConst::device_ny - 2) {
        for (int windowX = -1; windowX <= 1; windowX++) {
            for (int windowY = -1; windowY <= 1; windowY++) {
                int localIndex = indexPIC + windowY + windowX * PIC2DConst::device_ny; 
                convolvedMoment += moment[localIndex];
            }
        }
        convolvedMoment = convolvedMoment / 9.0; 
    } else {
        convolvedMoment = moment[indexPIC];
    }

    return convolvedMoment;
}


__device__ BasicParameter getBasicParameter_MHD(
    ConservationParameter U
)
{
    double rho, u, v, w, bX, bY, bZ, e, p;
    BasicParameter basicParameter; 

    rho = U.rho; 
    u   = U.rhoU / rho;
    v   = U.rhoV / rho;
    w   = U.rhoW / rho; 
    bX  = U.bX;
    bY  = U.bY;
    bZ  = U.bZ;
    e   = U.e;
    p   = (IdealMHD2DConst::device_gamma - 1.0)
        * (e - 0.5 * rho * (u * u + v * v + w * w)
        - 0.5 * (bX * bX + bY * bY + bZ * bZ));
    
    basicParameter.rho = rho; 
    basicParameter.u   = u; 
    basicParameter.v   = v; 
    basicParameter.w   = w; 
    basicParameter.bX  = bX; 
    basicParameter.bY  = bY; 
    basicParameter.bZ  = bZ;
    basicParameter.p   = p;

    return basicParameter; 
}

__global__ void sendMHDtoPIC_particle_y_kernel(
    const double* interlockingFunctionY, 
    const ZerothMoment* zerothMomentIon, 
    const ZerothMoment* zerothMomentElectron, 
    const FirstMoment* firstMomentIon, 
    const FirstMoment* firstMomentElectron, 
    const SecondMoment* secondMomentIon, 
    const SecondMoment* secondMomentElectron, 
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
        int indexPIC = j + (i + bufferPIC) * PIC2DConst::device_ny;

        ZerothMoment convolvedZerothMomentIon, convolvedZerothMomentElectron; 
        FirstMoment convolvedFirstMomentIon, convolvedFirstMomentElectron;
        SecondMoment convolvedSecondMomentIon, convolvedSecondMomentElectron;
        convolvedZerothMomentIon = getConvolvedMomentForMHDtoPIC(zerothMomentIon, indexPIC, j);
        convolvedZerothMomentElectron = getConvolvedMomentForMHDtoPIC(zerothMomentElectron, indexPIC, j);
        convolvedFirstMomentIon = getConvolvedMomentForMHDtoPIC(firstMomentIon, indexPIC, j);
        convolvedFirstMomentElectron = getConvolvedMomentForMHDtoPIC(firstMomentElectron, indexPIC, j);
        convolvedSecondMomentIon = getConvolvedMomentForMHDtoPIC(secondMomentIon, indexPIC, j);
        convolvedSecondMomentElectron = getConvolvedMomentForMHDtoPIC(secondMomentElectron, indexPIC, j);

        double niPIC, nePIC, rhoPIC, uPIC, vPIC, wPIC, jXPIC, jYPIC, jZPIC, piPIC, pePIC;

        niPIC  = convolvedZerothMomentIon.n;
        nePIC  = convolvedZerothMomentElectron.n;
        rhoPIC = PIC2DConst::device_mIon * niPIC + PIC2DConst::device_mElectron * nePIC;
        uPIC   = (PIC2DConst::device_mIon * convolvedFirstMomentIon.x  + PIC2DConst::device_mElectron * convolvedFirstMomentElectron.x) / rhoPIC;
        vPIC   = (PIC2DConst::device_mIon * convolvedFirstMomentIon.y  + PIC2DConst::device_mElectron * convolvedFirstMomentElectron.y) / rhoPIC;
        wPIC   = (PIC2DConst::device_mIon * convolvedFirstMomentIon.z  + PIC2DConst::device_mElectron * convolvedFirstMomentElectron.z) / rhoPIC;
        jXPIC  =  PIC2DConst::device_qIon * convolvedFirstMomentIon.x  + PIC2DConst::device_qElectron * convolvedFirstMomentElectron.x;
        jYPIC  =  PIC2DConst::device_qIon * convolvedFirstMomentIon.y  + PIC2DConst::device_qElectron * convolvedFirstMomentElectron.y;
        jZPIC  =  PIC2DConst::device_qIon * convolvedFirstMomentIon.z  + PIC2DConst::device_qElectron * convolvedFirstMomentElectron.z;
        piPIC  = PIC2DConst::device_mIon
               * (convolvedSecondMomentIon.xx + convolvedSecondMomentIon.yy + convolvedSecondMomentIon.zz
               - (pow(convolvedFirstMomentIon.x, 2) + pow(convolvedFirstMomentIon.y, 2) + pow(convolvedFirstMomentIon.z, 2))
               / (convolvedZerothMomentIon.n + Interface2DConst::device_EPS)) / 3.0;
        pePIC  = PIC2DConst::device_mElectron
               * (convolvedSecondMomentElectron.xx + convolvedSecondMomentElectron.yy + convolvedSecondMomentElectron.zz
               - (pow(convolvedFirstMomentElectron.x, 2) + pow(convolvedFirstMomentElectron.y, 2) + pow(convolvedFirstMomentElectron.z, 2))
               / (convolvedZerothMomentElectron.n + Interface2DConst::device_EPS)) / 3.0;

        int indexMHD = indexOfInterfaceStartInMHD + static_cast<int>(j / Interface2DConst::device_gridSizeRatio)
                     + (static_cast<int>(i / Interface2DConst::device_gridSizeRatio) + bufferMHD) * IdealMHD2DConst::device_ny;
        double rhoMHD, uMHD, vMHD, wMHD, jXMHD, jYMHD, jZMHD, pMHD;
        double niMHD, neMHD, piMHD, peMHD; 

        double cx1 = static_cast<double>((i % Interface2DConst::device_gridSizeRatio)) / Interface2DConst::device_gridSizeRatio; 
        double cx2 = 1.0 - cx1;
        double cy1 = static_cast<double>((j % Interface2DConst::device_gridSizeRatio)) / Interface2DConst::device_gridSizeRatio;
        double cy2 = 1.0 - cy1; 

        BasicParameter basicParameter_x1y1 = getBasicParameter_MHD(U[indexMHD]); 
        BasicParameter basicParameter_x2y1 = getBasicParameter_MHD(U[indexMHD + IdealMHD2DConst::device_ny]);
        BasicParameter basicParameter_x1y2 = getBasicParameter_MHD(U[indexMHD + 1]);
        BasicParameter basicParameter_x2y2 = getBasicParameter_MHD(U[indexMHD + IdealMHD2DConst::device_ny + 1]);

        CurrentField_MHD J_MHD_x1y1 = getCurrentField_MHD(U, indexMHD);
        CurrentField_MHD J_MHD_x2y1 = getCurrentField_MHD(U, indexMHD + Ideal2DConst::device_ny);
        CurrentField_MHD J_MHD_x1y2 = getCurrentField_MHD(U, indexMHD + 1);
        CurrentField_MHD J_MHD_x2y2 = getCurrentField_MHD(U, indexMHD + Ideal2DConst::device_ny + 1);

        rhoMHD = basicParameter_x1y1.rho * cx2 * cy2 + basicParameter_x2y1.rho * cx1 * cy2 + basicParameter_x1y2.rho * cx2 * cy1 + basicParameter_x2y2.rho * cx1 * cy1;
        uMHD   = basicParameter_x1y1.u   * cx2 * cy2 + basicParameter_x2y1.u   * cx1 * cy2 + basicParameter_x1y2.u   * cx2 * cy1 + basicParameter_x2y2.u   * cx1 * cy1;
        vMHD   = basicParameter_x1y1.v   * cx2 * cy2 + basicParameter_x2y1.v   * cx1 * cy2 + basicParameter_x1y2.v   * cx2 * cy1 + basicParameter_x2y2.v   * cx1 * cy1;
        wMHD   = basicParameter_x1y1.w   * cx2 * cy2 + basicParameter_x2y1.w   * cx1 * cy2 + basicParameter_x1y2.w   * cx2 * cy1 + basicParameter_x2y2.w   * cx1 * cy1;
        jXMHD  = J_MHD_x1y1.jX           * cx2 * cy2 + J_MHD_x2y1.jX           * cx1 * cy2 + J_MHD_x1y2.jX           * cx2 * cy1 + J_MHD_x2y2.jX           * cx1 * cy1;
        jYMHD  = J_MHD_x1y1.jY           * cx2 * cy2 + J_MHD_x2y1.jY           * cx1 * cy2 + J_MHD_x1y2.jY           * cx2 * cy1 + J_MHD_x2y2.jY           * cx1 * cy1;
        jZMHD  = J_MHD_x1y1.jZ           * cx2 * cy2 + J_MHD_x2y1.jZ           * cx1 * cy2 + J_MHD_x1y2.jZ           * cx2 * cy1 + J_MHD_x2y2.jZ           * cx1 * cy1;
        pMHD   = basicParameter_x1y1.p   * cx2 * cy2 + basicParameter_x2y1.p   * cx1 * cy2 + basicParameter_x1y2.p   * cx2 * cy1 + basicParameter_x2y2.p   * cx1 * cy1;
        
        niMHD = rhoMHD / (PIC2DConst::device_mIon + PIC2DConst::device_mElectron); 
        neMHD = niMHD; 
        //pressure ratio is assumed to be 1.0
        piMHD = pMHD / 2.0; 
        peMHD = pMHD / 2.0;

        niPIC = interlockingFunctionY[indexPIC] * niMHD + (1.0 - interlockingFunctionY[indexPIC]) * niPIC;
        nePIC = interlockingFunctionY[indexPIC] * neMHD + (1.0 - interlockingFunctionY[indexPIC]) * nePIC;
        uPIC  = interlockingFunctionY[indexPIC] * uMHD  + (1.0 - interlockingFunctionY[indexPIC]) * uPIC;
        vPIC  = interlockingFunctionY[indexPIC] * vMHD  + (1.0 - interlockingFunctionY[indexPIC]) * vPIC;
        wPIC  = interlockingFunctionY[indexPIC] * wMHD  + (1.0 - interlockingFunctionY[indexPIC]) * wPIC;
        jXPIC = interlockingFunctionY[indexPIC] * jXMHD + (1.0 - interlockingFunctionY[indexPIC]) * jXPIC;
        jYPIC = interlockingFunctionY[indexPIC] * jYMHD + (1.0 - interlockingFunctionY[indexPIC]) * jYPIC;
        jZPIC = interlockingFunctionY[indexPIC] * jZMHD + (1.0 - interlockingFunctionY[indexPIC]) * jZPIC;
        piPIC = interlockingFunctionY[indexPIC] * piMHD + (1.0 - interlockingFunctionY[indexPIC]) * piPIC;
        pePIC = interlockingFunctionY[indexPIC] * peMHD + (1.0 - interlockingFunctionY[indexPIC]) * pePIC;

        double vThiPIC, vThePIC;
        vThiPIC = sqrt(piPIC / round(niPIC) / PIC2DConst::device_mIon);
        vThePIC = sqrt(pePIC / round(nePIC) / PIC2DConst::device_mElectron);

        int indexForReload = j + i * PIC2DConst::device_ny;

        reloadParticlesDataIon     [indexForReload].numAndIndex = round(niPIC);
        reloadParticlesDataElectron[indexForReload].numAndIndex = round(nePIC);
        reloadParticlesDataIon     [indexForReload].u           = uPIC;
        reloadParticlesDataIon     [indexForReload].v           = vPIC;
        reloadParticlesDataIon     [indexForReload].w           = wPIC;
        reloadParticlesDataElectron[indexForReload].u           = uPIC - jXPIC / round(nePIC) / abs(PIC2DConst::device_qElectron); //niPIC = nePIC
        reloadParticlesDataElectron[indexForReload].v           = vPIC - jYPIC / round(nePIC) / abs(PIC2DConst::device_qElectron); //niPIC = nePIC
        reloadParticlesDataElectron[indexForReload].w           = wPIC - jZPIC / round(nePIC) / abs(PIC2DConst::device_qElectron); //niPIC = nePIC
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
    const thrust::device_vector<SecondMoment>& secondMomentIon, 
    const thrust::device_vector<SecondMoment>& secondMomentElectron, 
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron, 
    unsigned long long seed
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfoPIC.localNx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (PIC2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendMHDtoPIC_particle_y_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()), 
        thrust::raw_pointer_cast(zerothMomentIon.data()), 
        thrust::raw_pointer_cast(zerothMomentElectron.data()), 
        thrust::raw_pointer_cast(firstMomentIon.data()), 
        thrust::raw_pointer_cast(firstMomentElectron.data()), 
        thrust::raw_pointer_cast(secondMomentIon.data()), 
        thrust::raw_pointer_cast(secondMomentElectron.data()), 
        thrust::raw_pointer_cast(U.data()),  
        thrust::raw_pointer_cast(reloadParticlesDataIon.data()), 
        thrust::raw_pointer_cast(reloadParticlesDataElectron.data()), 
        indexOfInterfaceStartInMHD, 
        mPIInfoPIC.localNx, mPIInfoPIC.buffer, mPIInfoMHD.buffer
    );
    cudaDeviceSynchronize();
    
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


