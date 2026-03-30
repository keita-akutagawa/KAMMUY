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

__global__ void sendMHDtoPIC_magneticField_kernel(
    const double* interlockingFunction, 
    const ConservationParameter* U, 
    MagneticField* B, 
    const int indexOfInterfaceStartInMHD_x, 
    const int indexOfInterfaceStartInMHD_y
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < PIC2DConst::device_nx && j < PIC2DConst::device_ny) {
        double bXPIC, bYPIC, bZPIC;
        unsigned long long indexPIC = j + i * PIC2DConst::device_ny;
        
        bXPIC = B[indexPIC].bX;
        bYPIC = B[indexPIC].bY;
        bZPIC = B[indexPIC].bZ;
        

        double bXMHD, bYMHD, bZMHD;
        unsigned long long indexMHD = indexOfInterfaceStartInMHD_y + static_cast<int>(j / Interface2DConst::device_gridSizeRatio)
                                    + (indexOfInterfaceStartInMHD_x + static_cast<int>(i / Interface2DConst::device_gridSizeRatio))
                                    * IdealMHD2DConst::device_ny;
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

        bXInterface = interlockingFunction[indexPIC] * bXMHD + (1.0 - interlockingFunction[indexPIC]) * bXPIC;
        bYInterface = interlockingFunction[indexPIC] * bYMHD + (1.0 - interlockingFunction[indexPIC]) * bYPIC;
        bZInterface = interlockingFunction[indexPIC] * bZMHD + (1.0 - interlockingFunction[indexPIC]) * bZPIC;
        
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
    dim3 blocksPerGrid((PIC2DConst::nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (PIC2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendMHDtoPIC_magneticField_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunction.data()),
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(B.data()), 
        indexOfInterfaceStartInMHD_x, 
        indexOfInterfaceStartInMHD_y 
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


__global__ void sendMHDtoPIC_electricField_kernel(
    const double* interlockingFunction, 
    const ConservationParameter* U, 
    ElectricField* E, 
    const int indexOfInterfaceStartInMHD_x, 
    const int indexOfInterfaceStartInMHD_y
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < PIC2DConst::device_nx && j < PIC2DConst::device_ny) {
        double eXPIC, eYPIC, eZPIC;
        unsigned long long indexPIC = j + i * PIC2DConst::device_ny;

        eXPIC = E[indexPIC].eX;
        eYPIC = E[indexPIC].eY;
        eZPIC = E[indexPIC].eZ;


        double eXMHD, eYMHD, eZMHD;
        unsigned long long indexMHD = indexOfInterfaceStartInMHD_y + static_cast<int>(j / Interface2DConst::device_gridSizeRatio)
                                    + (indexOfInterfaceStartInMHD_x + static_cast<int>(i / Interface2DConst::device_gridSizeRatio))
                                    * IdealMHD2DConst::device_ny;
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

        eXInterface = interlockingFunction[indexPIC] * eXMHD + (1.0 - interlockingFunction[indexPIC]) * eXPIC;
        eYInterface = interlockingFunction[indexPIC] * eYMHD + (1.0 - interlockingFunction[indexPIC]) * eYPIC;
        eZInterface = interlockingFunction[indexPIC] * eZMHD + (1.0 - interlockingFunction[indexPIC]) * eZPIC;
         
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
    dim3 blocksPerGrid((PIC2DConst::nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (PIC2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendMHDtoPIC_electricField_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunction.data()), 
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(E.data()), 
        indexOfInterfaceStartInMHD_x, 
        indexOfInterfaceStartInMHD_y
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

__global__ void sendMHDtoPIC_currentField_kernel(
    const double* interlockingFunction, 
    const ConservationParameter* U, 
    CurrentField* current, 
    const int indexOfInterfaceStartInMHD_x, 
    const int indexOfInterfaceStartInMHD_y
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < PIC2DConst::device_nx && j < PIC2DConst::device_ny) {
        double jXPIC, jYPIC, jZPIC;
        unsigned long long indexPIC = j + i * PIC2DConst::device_ny;

        jXPIC = current[indexPIC].jX;
        jYPIC = current[indexPIC].jY;
        jZPIC = current[indexPIC].jZ;


        double jXMHD, jYMHD, jZMHD;
        unsigned long long indexMHD = indexOfInterfaceStartInMHD_y + static_cast<int>(j / Interface2DConst::device_gridSizeRatio)
                                    + (indexOfInterfaceStartInMHD_x + static_cast<int>(i / Interface2DConst::device_gridSizeRatio))
                                    * IdealMHD2DConst::device_ny;
        double cx1, cx2, cy1, cy2;  

        CurrentField_MHD J_MHD_x1y1 = getCurrentField_MHD(U, indexMHD);
        CurrentField_MHD J_MHD_x2y1 = getCurrentField_MHD(U, indexMHD + IdealMHD2DConst::device_ny);
        CurrentField_MHD J_MHD_x1y2 = getCurrentField_MHD(U, indexMHD + 1);
        CurrentField_MHD J_MHD_x2y2 = getCurrentField_MHD(U, indexMHD + IdealMHD2DConst::device_ny + 1);
        
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
        
        jXInterface = interlockingFunction[indexPIC] * jXMHD + (1.0 - interlockingFunction[indexPIC]) * jXPIC;
        jYInterface = interlockingFunction[indexPIC] * jYMHD + (1.0 - interlockingFunction[indexPIC]) * jYPIC;
        jZInterface = interlockingFunction[indexPIC] * jZMHD + (1.0 - interlockingFunction[indexPIC]) * jZPIC;
        
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
    dim3 blocksPerGrid((PIC2DConst::nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (PIC2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendMHDtoPIC_currentField_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunction.data()), 
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(current.data()), 
        indexOfInterfaceStartInMHD_x,
        indexOfInterfaceStartInMHD_y
    );
    cudaDeviceSynchronize();
}



__global__ void deleteParticles_kernel(
    const double* interlockingFunction, 
    Particle* particlesSpecies, 
    const unsigned long long existNumSpecies, 
    const unsigned long long seed
)
{
    unsigned long long k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < existNumSpecies) {
        double x = particlesSpecies[k].x;
        double y = particlesSpecies[k].y;

        int i = floorf(x - PIC2DConst::device_xmin); 
        int j = floorf(y - PIC2DConst::device_ymin);

        unsigned long long indexForInterlocking = j + i * PIC2DConst::device_ny;

        if (interlockingFunction[indexForInterlocking] < Interface2DConst::device_EPS) return;
        
        curandState state; 
        curand_init(seed, k, 0, &state);
        double randomValue = curand_uniform(&state);
        if (randomValue < interlockingFunction[indexForInterlocking]) {
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
        thrust::raw_pointer_cast(interlockingFunction.data()), 
        thrust::raw_pointer_cast(particlesSpecies.data()),
        existNumSpeciesPerProcs, 
        seed
    );
    cudaDeviceSynchronize();

    auto partitionEnd = thrust::partition(
        particlesSpecies.begin(), particlesSpecies.begin() + existNumSpeciesPerProcs, 
        [] __device__ (const Particle& p) { return p.isExist; }
    );

    existNumSpeciesPerProcs = static_cast<unsigned long long>(thrust::distance(particlesSpecies.begin(), partitionEnd));
}


__global__ void reloadParticlesSpecies_kernel(
    const double* interlockingFunction, 
    const ReloadParticlesData* reloadParticlesDataSpecies, 
    const Particle* reloadParticlesSourceSpecies, 
    unsigned long long reloadParticlesTotalNumSpecies, 
    Particle* particlesSpecies, 
    unsigned long long* particlesNumCounter, 
    const unsigned long long seed
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < PIC2DConst::device_nx && j < PIC2DConst::device_ny) {
        unsigned long long indexPIC = j + i * PIC2DConst::device_ny; 

        if (interlockingFunction[indexPIC] < Interface2DConst::device_EPS) return;

        unsigned long long indexForReload = j + i * PIC2DConst::device_ny; 
        double u = reloadParticlesDataSpecies[indexForReload].u;
        double v = reloadParticlesDataSpecies[indexForReload].v;
        double w = reloadParticlesDataSpecies[indexForReload].w;
        double vth = reloadParticlesDataSpecies[indexForReload].vth;

        Particle particleSource, particleReload;
        double x, y, z, vx, vy, vz, gamma;

        curandState stateForReloadIndex;
        curand_init(seed, indexForReload, 0, &stateForReloadIndex);
        unsigned long long restartParticlesIndexSpecies = static_cast<unsigned long long>(
            curand_uniform(&stateForReloadIndex) * reloadParticlesTotalNumSpecies
        );

        if (reloadParticlesDataSpecies[indexForReload].number < 0) printf("Minus reload!");
        curandState stateForReload; 
        curand_init(seed, indexForReload, 0, &stateForReload);
        for (int k = 0; k < reloadParticlesDataSpecies[indexForReload].number; k++) {
            double randomValue = curand_uniform(&stateForReload);

            if (randomValue < interlockingFunction[indexPIC]) {
                particleSource = reloadParticlesSourceSpecies[
                    (restartParticlesIndexSpecies + k) % reloadParticlesTotalNumSpecies
                ];

                x = particleSource.x; 
                x = (x + i) * PIC2DConst::device_dx + PIC2DConst::device_xmin;
                y = particleSource.y; 
                y = (y + j) * PIC2DConst::device_dy + PIC2DConst::device_ymin;
                z = particleSource.z;
                
                vx = particleSource.vx; vx = u + vx * vth;
                vy = particleSource.vy; vy = v + vy * vth;
                vz = particleSource.vz; vz = w + vz * vth;
                if (1.0 - (vx * vx + vy * vy + vz * vz) / pow(PIC2DConst::device_c, 2) < 0.0){
                    double normalizedVelocity = sqrt(vx * vx + vy * vy + vz * vz);
                    vx = vx / normalizedVelocity * 0.9 * PIC2DConst::device_c;
                    vy = vy / normalizedVelocity * 0.9 * PIC2DConst::device_c;
                    vz = vz / normalizedVelocity * 0.9 * PIC2DConst::device_c;
                };
                gamma = 1.0 / sqrt(1.0 - (vx * vx + vy * vy + vz * vz) / pow(PIC2DConst::device_c, 2));

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
    dim3 blocksPerGrid((PIC2DConst::nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (PIC2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    reloadParticlesSpecies_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunction.data()), 
        thrust::raw_pointer_cast(reloadParticlesDataSpecies.data()), 
        thrust::raw_pointer_cast(reloadParticlesSourceSpecies.data()), 
        Interface2DConst::reloadParticlesTotalNum, 
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        thrust::raw_pointer_cast(particlesNumCounter.data()), 
        seed
    );
    cudaDeviceSynchronize();

    existNumSpeciesPerProcs = particlesNumCounter[0];
}


template <typename MomentType>
__device__ MomentType getConvolvedMomentForMHDtoPIC(
    const MomentType* moment, 
    unsigned long long i, unsigned long long j
)
{
    MomentType convolvedMoment; 

    const int R = Interface2DConst::device_gridSizeRatio / 2;

    double weightSum = 0.0;
    for (int dx = -R; dx <= R; dx++) {
        for (int dy = -R; dy <= R; dy++) {
            int localI = i + dx;
            int localJ = j + dy;

            if (0 <= localI && localI < PIC2DConst::device_nx &&
                0 <= localJ && localJ < PIC2DConst::device_ny)
            {
                double weight = 1.0;

                unsigned long long index = localJ + localI * PIC2DConst::device_ny;
                convolvedMoment += moment[index] * weight;
                weightSum += weight;
            }
        }
    }
    convolvedMoment = convolvedMoment / weightSum;

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

__global__ void sendMHDtoPIC_particle_kernel(
    const double* interlockingFunction, 
    const ZerothMoment* zerothMomentIon, 
    const ZerothMoment* zerothMomentElectron, 
    const FirstMoment* firstMomentIon, 
    const FirstMoment* firstMomentElectron, 
    const SecondMoment* secondMomentIon, 
    const SecondMoment* secondMomentElectron, 
    const ConservationParameter* U, 
    ReloadParticlesData* reloadParticlesDataIon, 
    ReloadParticlesData* reloadParticlesDataElectron, 
    const int indexOfInterfaceStartInMHD_x, 
    const int indexOfInterfaceStartInMHD_y
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < PIC2DConst::device_nx && j < PIC2DConst::device_ny) {
        unsigned long long indexPIC = j + i * PIC2DConst::device_ny;

        ZerothMoment convolvedZerothMomentIon, convolvedZerothMomentElectron; 
        FirstMoment convolvedFirstMomentIon, convolvedFirstMomentElectron;
        SecondMoment convolvedSecondMomentIon, convolvedSecondMomentElectron;
        convolvedZerothMomentIon      = getConvolvedMomentForMHDtoPIC(zerothMomentIon, i, j);
        convolvedZerothMomentElectron = getConvolvedMomentForMHDtoPIC(zerothMomentElectron, i, j);
        convolvedFirstMomentIon       = getConvolvedMomentForMHDtoPIC(firstMomentIon, i, j);
        convolvedFirstMomentElectron  = getConvolvedMomentForMHDtoPIC(firstMomentElectron, i, j);
        convolvedSecondMomentIon      = getConvolvedMomentForMHDtoPIC(secondMomentIon, i, j);
        convolvedSecondMomentElectron = getConvolvedMomentForMHDtoPIC(secondMomentElectron, i, j);

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

        unsigned long long indexMHD = indexOfInterfaceStartInMHD_y + static_cast<int>(j / Interface2DConst::device_gridSizeRatio)
                                    + (indexOfInterfaceStartInMHD_x + static_cast<int>(i / Interface2DConst::device_gridSizeRatio))
                                    * IdealMHD2DConst::device_ny;
        
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
        CurrentField_MHD J_MHD_x2y1 = getCurrentField_MHD(U, indexMHD + IdealMHD2DConst::device_ny);
        CurrentField_MHD J_MHD_x1y2 = getCurrentField_MHD(U, indexMHD + 1);
        CurrentField_MHD J_MHD_x2y2 = getCurrentField_MHD(U, indexMHD + IdealMHD2DConst::device_ny + 1);

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

        niPIC = interlockingFunction[indexPIC] * niMHD + (1.0 - interlockingFunction[indexPIC]) * niPIC;
        nePIC = interlockingFunction[indexPIC] * neMHD + (1.0 - interlockingFunction[indexPIC]) * nePIC;
        uPIC  = interlockingFunction[indexPIC] * uMHD  + (1.0 - interlockingFunction[indexPIC]) * uPIC;
        vPIC  = interlockingFunction[indexPIC] * vMHD  + (1.0 - interlockingFunction[indexPIC]) * vPIC;
        wPIC  = interlockingFunction[indexPIC] * wMHD  + (1.0 - interlockingFunction[indexPIC]) * wPIC;
        jXPIC = interlockingFunction[indexPIC] * jXMHD + (1.0 - interlockingFunction[indexPIC]) * jXPIC;
        jYPIC = interlockingFunction[indexPIC] * jYMHD + (1.0 - interlockingFunction[indexPIC]) * jYPIC;
        jZPIC = interlockingFunction[indexPIC] * jZMHD + (1.0 - interlockingFunction[indexPIC]) * jZPIC;
        piPIC = interlockingFunction[indexPIC] * piMHD + (1.0 - interlockingFunction[indexPIC]) * piPIC;
        pePIC = interlockingFunction[indexPIC] * peMHD + (1.0 - interlockingFunction[indexPIC]) * pePIC;

        niPIC = thrust::max(niPIC, 1.0);
        nePIC = thrust::max(nePIC, 1.0);

        double vThiPIC, vThePIC;
        vThiPIC = sqrt(piPIC / static_cast<int>(round(niPIC)) / PIC2DConst::device_mIon);
        vThePIC = sqrt(pePIC / static_cast<int>(round(nePIC)) / PIC2DConst::device_mElectron);

        unsigned long long indexForReload = j + i * PIC2DConst::device_ny;

        reloadParticlesDataIon     [indexForReload].number = static_cast<int>(round(niPIC));
        reloadParticlesDataElectron[indexForReload].number = static_cast<int>(round(nePIC));
        reloadParticlesDataIon     [indexForReload].u      = uPIC;
        reloadParticlesDataIon     [indexForReload].v      = vPIC;
        reloadParticlesDataIon     [indexForReload].w      = wPIC;
        reloadParticlesDataElectron[indexForReload].u      = uPIC - jXPIC / static_cast<unsigned int>(round(nePIC)) / abs(PIC2DConst::device_qElectron); //niPIC = nePIC
        reloadParticlesDataElectron[indexForReload].v      = vPIC - jYPIC / static_cast<unsigned int>(round(nePIC)) / abs(PIC2DConst::device_qElectron); //niPIC = nePIC
        reloadParticlesDataElectron[indexForReload].w      = wPIC - jZPIC / static_cast<unsigned int>(round(nePIC)) / abs(PIC2DConst::device_qElectron); //niPIC = nePIC
        reloadParticlesDataIon     [indexForReload].vth    = vThiPIC;
        reloadParticlesDataElectron[indexForReload].vth    = vThePIC;

        if (reloadParticlesDataIon[indexForReload].number < 0) printf("%llu %d \n", indexForReload, reloadParticlesDataIon[indexForReload].number);
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
    dim3 blocksPerGrid((PIC2DConst::nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (PIC2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendMHDtoPIC_particle_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunction.data()), 
        thrust::raw_pointer_cast(zerothMomentIon.data()), 
        thrust::raw_pointer_cast(zerothMomentElectron.data()), 
        thrust::raw_pointer_cast(firstMomentIon.data()), 
        thrust::raw_pointer_cast(firstMomentElectron.data()), 
        thrust::raw_pointer_cast(secondMomentIon.data()), 
        thrust::raw_pointer_cast(secondMomentElectron.data()), 
        thrust::raw_pointer_cast(U.data()),  
        thrust::raw_pointer_cast(reloadParticlesDataIon.data()), 
        thrust::raw_pointer_cast(reloadParticlesDataElectron.data()), 
        indexOfInterfaceStartInMHD_x, 
        indexOfInterfaceStartInMHD_y
    );
    cudaDeviceSynchronize();
    
    deleteParticlesSpecies(
        particlesIon, PIC2DConst::existNumIon, seed + 100
    );
    deleteParticlesSpecies(
        particlesElectron, PIC2DConst::existNumElectron, seed + 200
    );

    reloadParticlesSpecies(
        particlesIon, reloadParticlesDataIon, reloadParticlesSourceIon, 
        PIC2DConst::existNumIon, seed + 300
    ); 
    reloadParticlesSpecies(
        particlesElectron, reloadParticlesDataElectron, reloadParticlesSourceElectron, 
        PIC2DConst::existNumElectron, seed + 400
    ); 

    if (PIC2DConst::existNumIon > PIC2DConst::totalNumIon) std::cout << "BROKEN" << std::endl;
    if (PIC2DConst::existNumElectron > PIC2DConst::totalNumElectron) std::cout << "BROKEN" << std::endl;
}


