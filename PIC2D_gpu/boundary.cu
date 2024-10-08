#include "boundary.hpp"
#include <thrust/partition.h>
#include <thrust/transform_reduce.h>


using namespace PIC2DConst;


__global__ void periodicBoundaryParticleX_kernel(
    Particle* particlesSpecies, unsigned long long existNumSpecies
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        if (particlesSpecies[i].x <= device_xmin_PIC) {
            particlesSpecies[i].x += device_xmax_PIC - device_xmin_PIC - device_EPS_PIC;
        }

        if (particlesSpecies[i].x >= device_xmax_PIC) {
            particlesSpecies[i].x -= device_xmax_PIC - device_xmin_PIC + device_EPS_PIC;
        }
    }
}

void BoundaryPIC::periodicBoundaryParticleX(
    thrust::device_vector<Particle>& particlesIon,
    thrust::device_vector<Particle>& particlesElectron
)
{
    dim3 threadsPerBlockForIon(256);
    dim3 blocksPerGridForIon((existNumIon_PIC + threadsPerBlockForIon.x - 1) / threadsPerBlockForIon.x);

    periodicBoundaryParticleX_kernel<<<blocksPerGridForIon, threadsPerBlockForIon>>>(
        thrust::raw_pointer_cast(particlesIon.data()), existNumIon_PIC
    );

    cudaDeviceSynchronize();

    dim3 threadsPerBlockForElectron(256);
    dim3 blocksPerGridForElectron((existNumElectron_PIC + threadsPerBlockForElectron.x - 1) / threadsPerBlockForElectron.x);

    periodicBoundaryParticleX_kernel<<<blocksPerGridForElectron, threadsPerBlockForElectron>>>(
        thrust::raw_pointer_cast(particlesElectron.data()), existNumElectron_PIC
    );

    cudaDeviceSynchronize();
}


__global__ void conductingWallBoundaryParticleX_kernel(
    Particle* particlesSpecies, unsigned long long existNumSpecies
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        if (particlesSpecies[i].x <= device_xmin_PIC) {
            particlesSpecies[i].x = 2.0 * device_xmin_PIC - particlesSpecies[i].x + device_EPS_PIC;
            particlesSpecies[i].vx = -1.0 * particlesSpecies[i].vx;
        }

        if (particlesSpecies[i].x >= device_xmax_PIC) {
            particlesSpecies[i].x = 2.0 * device_xmax_PIC - particlesSpecies[i].x - device_EPS_PIC;
            particlesSpecies[i].vx = -1.0 * particlesSpecies[i].vx;
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


__global__ void openBoundaryParticleX_kernel(
    Particle* particlesSpecies, 
    const unsigned long long existNumSpecies
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        if (particlesSpecies[i].x <= device_xmin_PIC) {
            particlesSpecies[i].isExist = false;
        }

        if (particlesSpecies[i].x >= device_xmax_PIC) {
            particlesSpecies[i].isExist = false;
        }
    }
}


void BoundaryPIC::openBoundaryParticleX(
    thrust::device_vector<Particle>& particlesIon,
    thrust::device_vector<Particle>& particlesElectron
)
{

    dim3 threadsPerBlockForIon(256);
    dim3 blocksPerGridForIon((existNumIon_PIC + threadsPerBlockForIon.x - 1) / threadsPerBlockForIon.x);

    openBoundaryParticleX_kernel<<<blocksPerGridForIon, threadsPerBlockForIon>>>(
        thrust::raw_pointer_cast(particlesIon.data()), existNumIon_PIC
    );

    cudaDeviceSynchronize();

    dim3 threadsPerBlockForElectron(256);
    dim3 blocksPerGridForElectron((existNumElectron_PIC + threadsPerBlockForElectron.x - 1) / threadsPerBlockForElectron.x);

    openBoundaryParticleX_kernel<<<blocksPerGridForElectron, threadsPerBlockForElectron>>>(
        thrust::raw_pointer_cast(particlesElectron.data()), existNumElectron_PIC
    );

    cudaDeviceSynchronize();

    
    existNumIon_PIC = thrust::transform_reduce(
        particlesIon.begin(),
        particlesIon.end(),
        IsExistTransform(), 
        0,               
        thrust::plus<unsigned long long>()
    );

    cudaDeviceSynchronize();

    existNumElectron_PIC = thrust::transform_reduce(
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


__global__ void conductingWallBoundaryParticleY_kernel(
    Particle* particlesSpecies, unsigned long long existNumSpecies
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        if (particlesSpecies[i].y <= device_ymin_PIC) {
            particlesSpecies[i].y = 2.0 * device_ymin_PIC - particlesSpecies[i].y + device_EPS_PIC;
            particlesSpecies[i].vy = -1.0 * particlesSpecies[i].vy;
        }

        if (particlesSpecies[i].y >= device_ymax_PIC) {
            particlesSpecies[i].y = 2.0 * device_ymax_PIC - particlesSpecies[i].y - device_EPS_PIC;
            particlesSpecies[i].vy = -1.0 * particlesSpecies[i].vy;
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
    const unsigned long long existNumSpecies
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        if (particlesSpecies[i].y <= device_ymin_PIC) {
            particlesSpecies[i].isExist = false;
        }

        if (particlesSpecies[i].y >= device_ymax_PIC) {
            particlesSpecies[i].isExist = false;
        }
    }
}


void BoundaryPIC::openBoundaryParticleY(
    thrust::device_vector<Particle>& particlesIon,
    thrust::device_vector<Particle>& particlesElectron
)
{

    dim3 threadsPerBlockForIon(256);
    dim3 blocksPerGridForIon((existNumIon_PIC + threadsPerBlockForIon.x - 1) / threadsPerBlockForIon.x);

    openBoundaryParticleY_kernel<<<blocksPerGridForIon, threadsPerBlockForIon>>>(
        thrust::raw_pointer_cast(particlesIon.data()), existNumIon_PIC
    );

    cudaDeviceSynchronize();

    dim3 threadsPerBlockForElectron(256);
    dim3 blocksPerGridForElectron((existNumElectron_PIC + threadsPerBlockForElectron.x - 1) / threadsPerBlockForElectron.x);

    openBoundaryParticleY_kernel<<<blocksPerGridForElectron, threadsPerBlockForElectron>>>(
        thrust::raw_pointer_cast(particlesElectron.data()), existNumElectron_PIC
    );

    cudaDeviceSynchronize();

    
    existNumIon_PIC = thrust::transform_reduce(
        particlesIon.begin(),
        particlesIon.end(),
        IsExistTransform(), 
        0,               
        thrust::plus<unsigned long long>()
    );

    cudaDeviceSynchronize();

    existNumElectron_PIC = thrust::transform_reduce(
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

//////////


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
        B[j + device_ny_PIC * 0].bY = 0.0;
        B[j + device_ny_PIC * 0].bZ = B[j + device_ny_PIC * 1].bZ;

        B[j + device_ny_PIC * (device_nx_PIC - 1)].bX = B[j + device_ny_PIC * (device_nx_PIC - 2)].bX;
        B[j + device_ny_PIC * (device_nx_PIC - 2)].bY = 0.0;
        B[j + device_ny_PIC * (device_nx_PIC - 1)].bY = 0.0;
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


__global__ void freeBoundaryBX_kernel(
    MagneticField* B
)
{
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < device_ny_PIC) {
        B[j + 0 * device_ny_PIC] = B[j + 1 * device_ny_PIC];

        B[j + (device_nx_PIC - 1) * device_ny_PIC] = B[j + (device_nx_PIC - 2) * device_ny_PIC];
    }
}

void BoundaryPIC::freeBoundaryBX(
    thrust::device_vector<MagneticField>& B
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny_PIC + threadsPerBlock.y - 1) / threadsPerBlock.y);

    freeBoundaryBX_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(B.data())
    );

    cudaDeviceSynchronize();
}



__global__ void conductingWallBoundaryBY_kernel(
    MagneticField* B
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < device_nx_PIC) {
        B[0 + device_ny_PIC * i].bX = B[1 + device_ny_PIC * i].bX;
        B[1 + device_ny_PIC * i].bY = 0.0;
        B[0 + device_ny_PIC * i].bY = 0.0;
        B[0 + device_ny_PIC * i].bZ = B[1 + device_ny_PIC * i].bZ;

        B[device_ny_PIC - 2 + device_ny_PIC * i].bX = B[device_ny_PIC - 3 + device_ny_PIC * i].bX;
        B[device_ny_PIC - 1 + device_ny_PIC * i].bX = B[device_ny_PIC - 2 + device_ny_PIC * i].bX;
        B[device_ny_PIC - 1 + device_ny_PIC * i].bY = -1.0 * B[device_ny_PIC - 2 + device_ny_PIC * i].bY;
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
        B[0 + i * device_ny_PIC] = B[1 + i * device_ny_PIC];

        B[device_ny_PIC - 1 + i * device_ny_PIC] = B[device_ny_PIC - 2 + i * device_ny_PIC];
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


//////////////////////////////////////////////////


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
        E[j + device_ny_PIC * 0].eX = 0.0;
        E[j + device_ny_PIC * 0].eY = E[j + device_ny_PIC * 1].eY;
        E[j + device_ny_PIC * 0].eZ = E[j + device_ny_PIC * 1].eZ;

        E[j + device_ny_PIC * (device_nx_PIC - 1)].eX = 0.0;
        E[j + device_ny_PIC * (device_nx_PIC - 2)].eX = 0.0;
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


__global__ void freeBoundaryEX_kernel(
    ElectricField* E
)
{
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < device_ny_PIC) {
        E[j + 0 * device_ny_PIC] = E[j + 1 * device_ny_PIC];

        E[j + (device_nx_PIC - 1) * device_ny_PIC] = E[j + (device_nx_PIC - 2) * device_ny_PIC];
    }
}

void BoundaryPIC::freeBoundaryEX(
    thrust::device_vector<ElectricField>& E
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny_PIC + threadsPerBlock.y - 1) / threadsPerBlock.y);

    freeBoundaryEX_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(E.data())
    );

    cudaDeviceSynchronize();
}


__global__ void conductingWallBoundaryEY_kernel(
    ElectricField* E
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < device_nx_PIC) {
        E[0 + device_ny_PIC * i].eX = 0.0;
        E[1 + device_ny_PIC * i].eX = 0.0;
        E[0 + device_ny_PIC * i].eY = -1.0 * E[1 + device_ny_PIC * i].eY;
        E[0 + device_ny_PIC * i].eZ = 0.0;
        E[1 + device_ny_PIC * i].eZ = 0.0;

        E[device_ny_PIC - 1 + device_ny_PIC * i].eX = -1.0 * E[device_ny_PIC - 2 + device_ny_PIC * i].eX;
        E[device_ny_PIC - 1 + device_ny_PIC * i].eY = 0.0;
        E[device_ny_PIC - 2 + device_ny_PIC * i].eY = 0.0;
        E[device_ny_PIC - 1 + device_ny_PIC * i].eZ = -1.0 * E[device_ny_PIC - 2 + device_ny_PIC * i].eZ;
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
        E[0 + i * device_ny_PIC] = E[1 + i * device_ny_PIC];

        E[device_ny_PIC - 1 + i * device_ny_PIC] = E[device_ny_PIC - 2 + i * device_ny_PIC];
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


//////////////////////////////////////////////////


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
        current[j + device_ny_PIC * 0].jX = 0.0;
        current[j + device_ny_PIC * 0].jY = current[j + device_ny_PIC * 1].jY;
        current[j + device_ny_PIC * 0].jZ = current[j + device_ny_PIC * 1].jZ;
        current[j + device_ny_PIC * (device_nx_PIC - 1)].jX = 0.0;
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


__global__ void freeBoundaryCurrentX_kernel(
    CurrentField* current
)
{
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < device_ny_PIC) {
        current[j + 0 * device_ny_PIC] = current[j + 1 * device_ny_PIC];

        current[j + (device_nx_PIC - 1) * device_ny_PIC] = current[j + (device_nx_PIC - 2) * device_ny_PIC];
    }
}

void BoundaryPIC::freeBoundaryCurrentX(
    thrust::device_vector<CurrentField>& current
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny_PIC + threadsPerBlock.y - 1) / threadsPerBlock.y);

    freeBoundaryCurrentX_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(current.data())
    );

    cudaDeviceSynchronize();
}


__global__ void conductingWallBoundaryCurrentY_kernel(
    CurrentField* current
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < device_nx_PIC) {
        current[0 + device_ny_PIC * i].jX = 0.0;
        current[1 + device_ny_PIC * i].jX = 0.0;
        current[0 + device_ny_PIC * i].jY = 0.0;
        current[0 + device_ny_PIC * i].jZ = 0.0;
        current[1 + device_ny_PIC * i].jZ = 0.0;
        current[device_ny_PIC - 1 + device_ny_PIC * i].jX = 0.0;
        current[device_ny_PIC - 2 + device_ny_PIC * i].jX = 0.0;
        current[device_ny_PIC - 1 + device_ny_PIC * i].jY = 0.0;
        current[device_ny_PIC - 2 + device_ny_PIC * i].jY = 0.0;
        current[device_ny_PIC - 1 + device_ny_PIC * i].jZ = 0.0;
        current[device_ny_PIC - 2 + device_ny_PIC * i].jZ = 0.0;
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
        current[0 + i * device_ny_PIC] = current[1 + i * device_ny_PIC];

        current[device_ny_PIC - 1 + i * device_ny_PIC] = current[device_ny_PIC - 2 + i * device_ny_PIC];
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

