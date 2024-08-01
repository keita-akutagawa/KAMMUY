#include "boundary.hpp"
#include <thrust/partition.h>


using namespace PIC2DConst;


__global__ void conductingWallBoundaryParticleX_kernel(
    Particle* particlesSpecies, unsigned long long existNumSpecies
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        if (particlesSpecies[i].x <= device_xmin_PIC) {
            particlesSpecies[i].x = 2.0f * device_xmin_PIC - particlesSpecies[i].x + device_EPS_PIC;
            particlesSpecies[i].vx = -1.0f * particlesSpecies[i].vx;
        }

        if (particlesSpecies[i].x >= device_xmax_PIC) {
            particlesSpecies[i].x = 2.0f * device_xmax_PIC - particlesSpecies[i].x - device_EPS_PIC;
            particlesSpecies[i].vx = -1.0f * particlesSpecies[i].vx;
        }
    }
}

void BoundaryPIC::conductingWallBoundaryParticleX(
    thrust::device_vector<Particle>& particlesIon,
    thrust::device_vector<Particle>& particlesElectron
)
{
    dim3 threadsPerBlockForIon(256);
    dim3 blocksPerGridForIon((existNumIon_PIC + threadsPerBlockForIon.x - 1) / threadsPerBlockForIon.x);

    conductingWallBoundaryParticleX_kernel<<<blocksPerGridForIon, threadsPerBlockForIon>>>(
        thrust::raw_pointer_cast(particlesIon.data()), existNumIon_PIC
    );

    cudaDeviceSynchronize();

    dim3 threadsPerBlockForElectron(256);
    dim3 blocksPerGridForElectron((existNumElectron_PIC + threadsPerBlockForElectron.x - 1) / threadsPerBlockForElectron.x);

    conductingWallBoundaryParticleX_kernel<<<blocksPerGridForElectron, threadsPerBlockForElectron>>>(
        thrust::raw_pointer_cast(particlesElectron.data()), existNumElectron_PIC
    );

    cudaDeviceSynchronize();
}


__global__ void conductingWallBoundaryParticleY_kernel(
    Particle* particlesSpecies, unsigned long long existNumSpecies
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        if (particlesSpecies[i].y <= device_ymin_PIC) {
            particlesSpecies[i].y = 2.0f * device_ymin_PIC - particlesSpecies[i].y + device_EPS_PIC;
            particlesSpecies[i].vy = -1.0f * particlesSpecies[i].vy;
        }

        if (particlesSpecies[i].y >= device_ymax_PIC) {
            particlesSpecies[i].y = 2.0f * device_ymax_PIC - particlesSpecies[i].y - device_EPS_PIC;
            particlesSpecies[i].vy = -1.0f * particlesSpecies[i].vy;
        }
    }
}

void BoundaryPIC::conductingWallBoundaryParticleY(
    thrust::device_vector<Particle>& particlesIon,
    thrust::device_vector<Particle>& particlesElectron
)
{
    dim3 threadsPerBlockForIon(256);
    dim3 blocksPerGridForIon((existNumIon_PIC + threadsPerBlockForIon.x - 1) / threadsPerBlockForIon.x);

    conductingWallBoundaryParticleY_kernel<<<blocksPerGridForIon, threadsPerBlockForIon>>>(
        thrust::raw_pointer_cast(particlesIon.data()), existNumIon_PIC
    );

    cudaDeviceSynchronize();

    dim3 threadsPerBlockForElectron(256);
    dim3 blocksPerGridForElectron((existNumElectron_PIC + threadsPerBlockForElectron.x - 1) / threadsPerBlockForElectron.x);

    conductingWallBoundaryParticleY_kernel<<<blocksPerGridForElectron, threadsPerBlockForElectron>>>(
        thrust::raw_pointer_cast(particlesElectron.data()), existNumElectron_PIC
    );

    cudaDeviceSynchronize();
}


__global__ void openBoundaryParticleY_kernel(
    Particle* particlesSpecies, 
    const unsigned long long existNumSpecies, 
    unsigned long long& existNumSpeciesNext
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        if (particlesSpecies[i].y <= device_ymin_PIC) {
            particlesSpecies[i].isExist = false;
            return;
        }

        if (particlesSpecies[i].y >= device_ymax_PIC) {
            particlesSpecies[i].isExist = false;
            return;
        }

        atomicAdd(&existNumSpeciesNext, 1);
    }
}

void BoundaryPIC::openBoundaryParticleY(
    thrust::device_vector<Particle>& particlesIon,
    thrust::device_vector<Particle>& particlesElectron
)
{
    existNumIonNext = 0;
    existNumElectronNext = 0;

    dim3 threadsPerBlockForIon(256);
    dim3 blocksPerGridForIon((existNumIon_PIC + threadsPerBlockForIon.x - 1) / threadsPerBlockForIon.x);

    openBoundaryParticleY_kernel<<<blocksPerGridForIon, threadsPerBlockForIon>>>(
        thrust::raw_pointer_cast(particlesIon.data()), existNumIon_PIC, existNumIonNext
    );

    cudaDeviceSynchronize();

    dim3 threadsPerBlockForElectron(256);
    dim3 blocksPerGridForElectron((existNumElectron_PIC + threadsPerBlockForElectron.x - 1) / threadsPerBlockForElectron.x);

    openBoundaryParticleY_kernel<<<blocksPerGridForElectron, threadsPerBlockForElectron>>>(
        thrust::raw_pointer_cast(particlesElectron.data()), existNumElectron_PIC, existNumElectronNext
    );

    cudaDeviceSynchronize();

    existNumIon_PIC = existNumIonNext;
    existNumElectron_PIC = existNumElectronNext;


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

//////////

void BoundaryPIC::periodicBoundaryBX(
    thrust::device_vector<MagneticField>& B
)
{
    return;
}

void BoundaryPIC::conductingWallBoundaryBX(
    thrust::device_vector<MagneticField>& B
)
{
    std::cout << "Not writtern yet. Finish your calculation now!" << std::endl;
}

__global__ void symmetricBoundaryBX_kernel(
    MagneticField* B
)
{
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < device_ny_PIC) {
        B[j + device_ny_PIC * 0].bX = B[j + device_ny_PIC * 1].bX;
        B[j + device_ny_PIC * 0].bY = 0.0f;
        B[j + device_ny_PIC * 0].bZ = B[j + device_ny_PIC * 1].bZ;

        B[j + device_ny_PIC * (device_nx_PIC - 1)].bX = B[j + device_ny_PIC * (device_nx_PIC - 2)].bX;
        B[j + device_ny_PIC * (device_nx_PIC - 2)].bY = 0.0f;
        B[j + device_ny_PIC * (device_nx_PIC - 1)].bY = 0.0f;
        B[j + device_ny_PIC * (device_nx_PIC - 2)].bZ = B[j + device_ny_PIC * (device_nx_PIC - 3)].bZ;
        B[j + device_ny_PIC * (device_nx_PIC - 1)].bZ = B[j + device_ny_PIC * (device_nx_PIC - 2)].bZ;
    }
}

void BoundaryPIC::symmetricBoundaryBX(
    thrust::device_vector<MagneticField>& B
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny_PIC + threadsPerBlock.y - 1) / threadsPerBlock.y);

    symmetricBoundaryBX_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(B.data())
    );

    cudaDeviceSynchronize();
}


void BoundaryPIC::periodicBoundaryBY(
    thrust::device_vector<MagneticField>& B
)
{
    return;
}


__global__ void conductingWallBoundaryBY_kernel(
    MagneticField* B
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < device_nx_PIC) {
        B[0 + device_ny_PIC * i].bX = B[1 + device_ny_PIC * i].bX;
        B[1 + device_ny_PIC * i].bY = 0.0f;
        B[0 + device_ny_PIC * i].bY = 0.0f;
        B[0 + device_ny_PIC * i].bZ = B[1 + device_ny_PIC * i].bZ;

        B[device_ny_PIC - 2 + device_ny_PIC * i].bX = B[device_ny_PIC - 3 + device_ny_PIC * i].bX;
        B[device_ny_PIC - 1 + device_ny_PIC * i].bX = B[device_ny_PIC - 2 + device_ny_PIC * i].bX;
        B[device_ny_PIC - 1 + device_ny_PIC * i].bY = -1.0f * B[device_ny_PIC - 2 + device_ny_PIC * i].bY;
        B[device_ny_PIC - 2 + device_ny_PIC * i].bZ = B[device_ny_PIC - 3 + device_ny_PIC * i].bZ;
        B[device_ny_PIC - 1 + device_ny_PIC * i].bZ = B[device_ny_PIC - 2 + device_ny_PIC * i].bZ;
    }
}


void BoundaryPIC::conductingWallBoundaryBY(
    thrust::device_vector<MagneticField>& B
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny_PIC + threadsPerBlock.y - 1) / threadsPerBlock.y);

    conductingWallBoundaryBY_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(B.data())
    );

    cudaDeviceSynchronize();
}


void BoundaryPIC::symmetricBoundaryBY(
    thrust::device_vector<MagneticField>& B
)
{
    std::cout << "Not writtern yet. Finish your calculation now!" << std::endl;
}


__global__ void freeBoundaryBY_kernel(
    MagneticField* B
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < device_nx_PIC) {
        B[0 + device_ny_PIC * i].bX = B[1 + device_ny_PIC * i].bX;
        B[0 + device_ny_PIC * i].bY = B[2 + device_ny_PIC * i].bY;
        B[0 + device_ny_PIC * i].bZ = B[1 + device_ny_PIC * i].bZ;

        B[device_ny_PIC - 1 + device_ny_PIC * i].bX = B[device_ny_PIC - 3 + device_ny_PIC * i].bX;
        B[device_ny_PIC - 1 + device_ny_PIC * i].bY = B[device_ny_PIC - 2 + device_ny_PIC * i].bY;
        B[device_ny_PIC - 1 + device_ny_PIC * i].bZ = B[device_ny_PIC - 3 + device_ny_PIC * i].bZ;
    }
}

void BoundaryPIC::freeBoundaryBY(
    thrust::device_vector<MagneticField>& B
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny_PIC + threadsPerBlock.y - 1) / threadsPerBlock.y);

    freeBoundaryBY_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(B.data())
    );

    cudaDeviceSynchronize();
}

//////////


void BoundaryPIC::periodicBoundaryEX(
    thrust::device_vector<ElectricField>& E
)
{
    return;
}


void BoundaryPIC::conductingWallBoundaryEX(
    thrust::device_vector<ElectricField>& E
)
{
    std::cout << "Not writtern yet. Finish your calculation now!" << std::endl;
}


__global__ void symmetricBoundaryEX_kernel(
    ElectricField* E
)
{
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < device_ny_PIC) {
        E[j + device_ny_PIC * 0].eX = 0.0f;
        E[j + device_ny_PIC * 0].eY = E[j + device_ny_PIC * 1].eY;
        E[j + device_ny_PIC * 0].eZ = E[j + device_ny_PIC * 1].eZ;

        E[j + device_ny_PIC * (device_nx_PIC - 1)].eX = 0.0f;
        E[j + device_ny_PIC * (device_nx_PIC - 2)].eX = 0.0f;
        E[j + device_ny_PIC * (device_nx_PIC - 1)].eY = E[j + device_ny_PIC * (device_nx_PIC - 2)].eY;
        E[j + device_ny_PIC * (device_nx_PIC - 1)].eZ = E[j + device_ny_PIC * (device_nx_PIC - 2)].eZ;
    }
}


void BoundaryPIC::symmetricBoundaryEX(
    thrust::device_vector<ElectricField>& E
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny_PIC + threadsPerBlock.y - 1) / threadsPerBlock.y);

    symmetricBoundaryEX_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(E.data())
    );

    cudaDeviceSynchronize();
}


void BoundaryPIC::periodicBoundaryEY(
    thrust::device_vector<ElectricField>& E
)
{
    return;
}


__global__ void conductingWallBoundaryEY_kernel(
    ElectricField* E
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < device_nx_PIC) {
        E[0 + device_ny_PIC * i].eX = 0.0f;
        E[1 + device_ny_PIC * i].eX = 0.0f;
        E[0 + device_ny_PIC * i].eY = -1.0f * E[1 + device_ny_PIC * i].eY;
        E[0 + device_ny_PIC * i].eZ = 0.0f;
        E[1 + device_ny_PIC * i].eZ = 0.0f;

        E[device_ny_PIC - 1 + device_ny_PIC * i].eX = -1.0f * E[device_ny_PIC - 2 + device_ny_PIC * i].eX;
        E[device_ny_PIC - 1 + device_ny_PIC * i].eY = 0.0f;
        E[device_ny_PIC - 2 + device_ny_PIC * i].eY = 0.0f;
        E[device_ny_PIC - 1 + device_ny_PIC * i].eZ = -1.0f * E[device_ny_PIC - 2 + device_ny_PIC * i].eZ;
    }
}


void BoundaryPIC::conductingWallBoundaryEY(
    thrust::device_vector<ElectricField>& E
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny_PIC + threadsPerBlock.y - 1) / threadsPerBlock.y);

    conductingWallBoundaryEY_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(E.data())
    );

    cudaDeviceSynchronize();
}


void BoundaryPIC::symmetricBoundaryEY(
    thrust::device_vector<ElectricField>& E
)
{
    std::cout << "Not writtern yet. Finish your calculation now!" << std::endl;
}


__global__ void freeBoundaryEY_kernel(
    ElectricField* E
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < device_nx_PIC) {
        E[0 + device_ny_PIC * i].eX = 0.0f;
        E[1 + device_ny_PIC * i].eX = 0.0f;
        E[0 + device_ny_PIC * i].eY = -1.0f * E[1 + device_ny_PIC * i].eY;
        E[0 + device_ny_PIC * i].eZ = 0.0f;
        E[1 + device_ny_PIC * i].eZ = 0.0f;

        E[device_ny_PIC - 1 + device_ny_PIC * i].eX = -1.0f * E[device_ny_PIC - 2 + device_ny_PIC * i].eX;
        E[device_ny_PIC - 1 + device_ny_PIC * i].eY = 0.0f;
        E[device_ny_PIC - 2 + device_ny_PIC * i].eY = 0.0f;
        E[device_ny_PIC - 1 + device_ny_PIC * i].eZ = -1.0f * E[device_ny_PIC - 2 + device_ny_PIC * i].eZ;
    }
}

void BoundaryPIC::freeBoundaryEY(
    thrust::device_vector<ElectricField>& E
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny_PIC + threadsPerBlock.y - 1) / threadsPerBlock.y);

    freeBoundaryEY_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(E.data())
    );

    cudaDeviceSynchronize();
}

//////////

void BoundaryPIC::periodicBoundaryCurrentX(
    thrust::device_vector<CurrentField>& current
)
{
    return;
}


void BoundaryPIC::conductingWallBoundaryCurrentX(
    thrust::device_vector<CurrentField>& current
)
{
    std::cout << "Not writtern yet. Finish your calculation now!" << std::endl;
}


__global__ void symmetricBoundaryCurrentX_kernel(
    CurrentField* current
)
{
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < device_ny_PIC) {
        current[j + device_ny_PIC * 0].jX = 0.0f;
        current[j + device_ny_PIC * 0].jY = current[j + device_ny_PIC * 1].jY;
        current[j + device_ny_PIC * 0].jZ = current[j + device_ny_PIC * 1].jZ;
        current[j + device_ny_PIC * (device_nx_PIC - 1)].jX = 0.0f;
        current[j + device_ny_PIC * (device_nx_PIC - 1)].jY = current[j + device_ny_PIC * (device_nx_PIC - 2)].jY;
        current[j + device_ny_PIC * (device_nx_PIC - 1)].jZ = current[j + device_ny_PIC * (device_nx_PIC - 2)].jZ;
    }
}


void BoundaryPIC::symmetricBoundaryCurrentX(
    thrust::device_vector<CurrentField>& current
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny_PIC + threadsPerBlock.y - 1) / threadsPerBlock.y);

    symmetricBoundaryCurrentX_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(current.data())
    );

    cudaDeviceSynchronize();
}


void BoundaryPIC::periodicBoundaryCurrentY(
    thrust::device_vector<CurrentField>& current
)
{
    return;
}


__global__ void conductingWallBoundaryCurrentY_kernel(
    CurrentField* current
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < device_nx_PIC) {
        current[0 + device_ny_PIC * i].jX = 0.0f;
        current[1 + device_ny_PIC * i].jX = 0.0f;
        current[0 + device_ny_PIC * i].jY = 0.0f;
        current[0 + device_ny_PIC * i].jZ = 0.0f;
        current[1 + device_ny_PIC * i].jZ = 0.0f;
        current[device_ny_PIC - 1 + device_ny_PIC * i].jX = 0.0f;
        current[device_ny_PIC - 2 + device_ny_PIC * i].jX = 0.0f;
        current[device_ny_PIC - 1 + device_ny_PIC * i].jY = 0.0f;
        current[device_ny_PIC - 2 + device_ny_PIC * i].jY = 0.0f;
        current[device_ny_PIC - 1 + device_ny_PIC * i].jZ = 0.0f;
        current[device_ny_PIC - 2 + device_ny_PIC * i].jZ = 0.0f;
    }
}


void BoundaryPIC::conductingWallBoundaryCurrentY(
    thrust::device_vector<CurrentField>& current
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny_PIC + threadsPerBlock.y - 1) / threadsPerBlock.y);

    conductingWallBoundaryCurrentY_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(current.data())
    );

    cudaDeviceSynchronize();
}


void BoundaryPIC::symmetricBoundaryCurrentY(
    thrust::device_vector<CurrentField>& current
)
{
    std::cout << "Not writtern yet. Finish your calculation now!" << std::endl;
}


__global__ void freeBoundaryCurrentY_kernel(
    CurrentField* current
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < device_nx_PIC) {
        current[0 + device_ny_PIC * i].jX = current[2 + device_ny_PIC * i].jX;
        current[0 + device_ny_PIC * i].jY = current[1 + device_ny_PIC * i].jY;
        current[0 + device_ny_PIC * i].jZ = current[2 + device_ny_PIC * i].jZ;

        current[device_ny_PIC - 1 + device_ny_PIC * i].jX = current[device_ny_PIC - 2 + device_ny_PIC * i].jX;
        current[device_ny_PIC - 1 + device_ny_PIC * i].jY = current[device_ny_PIC - 3 + device_ny_PIC * i].jY;
        current[device_ny_PIC - 1 + device_ny_PIC * i].jZ = current[device_ny_PIC - 2 + device_ny_PIC * i].jZ;
    }
}

void BoundaryPIC::freeBoundaryCurrentY(
    thrust::device_vector<CurrentField>& current
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny_PIC + threadsPerBlock.y - 1) / threadsPerBlock.y);

    freeBoundaryCurrentY_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(current.data())
    );

    cudaDeviceSynchronize();
}

