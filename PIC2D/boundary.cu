#include "boundary.hpp"


PICBoundary::PICBoundary(
    PICConstParameter& pICConstParameter, 
    const PICGridParameter& pICGridParameter
)
    : NX(pICGridParameter.NX), 
      NY(pICGridParameter.NY), 
      DX(pICGridParameter.DX), 
      DY(pICGridParameter.DY), 
      
      pICConstParameter(pICConstParameter), 
      pICGridParameter(pICGridParameter)
{
}


void PICBoundary::freeBoundaryParticleX(
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron
)
{   
    freeBoundaryParticleSpeciesX(particlesIon, pICConstParameter.EXIST_NUM_ION);
    freeBoundaryParticleSpeciesX(particlesElectron, pICConstParameter.EXIST_NUM_ELECTRON);

    if (pICConstParameter.EXIST_NUM_ION > pICConstParameter.TOTAL_NUM_ION) {
        std::cout << "exist number of ion particles exceeds total number (with buffer)" << std::endl;
    }
    if (pICConstParameter.EXIST_NUM_ELECTRON > pICConstParameter.TOTAL_NUM_ELECTRON) {
        std::cout << "exist number of electron particles exceeds total number (with buffer)" << std::endl;
    }
}


void PICBoundary::freeBoundaryParticleY(
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron
)
{   
    freeBoundaryParticleSpeciesY(particlesIon, pICConstParameter.EXIST_NUM_ION);
    freeBoundaryParticleSpeciesY(particlesElectron, pICConstParameter.EXIST_NUM_ELECTRON);

    if (pICConstParameter.EXIST_NUM_ION > pICConstParameter.TOTAL_NUM_ION) {
        std::cout << "exist number of ion particles exceeds total number (with buffer)" << std::endl;
    }
    if (pICConstParameter.EXIST_NUM_ELECTRON > pICConstParameter.TOTAL_NUM_ELECTRON) {
        std::cout << "exist number of electron particles exceeds total number (with buffer)" << std::endl;
    }
}


__global__ void freeBoundaryParticleSpeciesX_kernel(
    const PICFloat DX, 
    const PICFloat XMIN, const PICFloat XMAX, 
    const PICUnsignedLongLong EXIST_NUM, 
    Particle* particles, 
    PICUnsignedLongLong* countForFreeBoundaryParticles
)
{
    PICUnsignedLongLong i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < EXIST_NUM) {
        PICFloat x = particles[i].x; 
        
        if (x <= XMIN + DX) {
            particles[i].isExist = false; 
        }
        if (x >= XMAX - DX) {
            particles[i].isExist = false; 
        }
        
        if (x > XMIN + DX && x <= XMIN + 2 * DX) {
            PICUnsignedLongLong particleIndex = atomicAdd(&(countForFreeBoundaryParticles[0]), 1);
            Particle sendParticle = particles[i];
            sendParticle.x = sendParticle.x - DX; 
            particles[particleIndex] = sendParticle;
        }

        if (x < XMAX - DX && x >= XMAX - 2 * DX) {
            PICUnsignedLongLong particleIndex = atomicAdd(&(countForFreeBoundaryParticles[0]), 1);
            Particle sendParticle = particles[i];
            sendParticle.x = sendParticle.x + DX; 
            particles[particleIndex] = sendParticle;
        }
    }
}

void PICBoundary::freeBoundaryParticleSpeciesX(
    thrust::device_vector<Particle>& particles, 
    PICUnsignedLongLong& EXIST_NUM
)
{   
    thrust::device_vector<PICUnsignedLongLong> countForFreeBoundaryParticles(1, EXIST_NUM); 

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((EXIST_NUM + threadsPerBlock.x - 1) / threadsPerBlock.x);

    freeBoundaryParticleSpeciesX_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        DX, 
        pICGridParameter.XMIN, pICGridParameter.XMAX, 
        EXIST_NUM, 
        thrust::raw_pointer_cast(particles.data()), 
        thrust::raw_pointer_cast(countForFreeBoundaryParticles.data())
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed at freeBoundaryParticleSpeciesX_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed at freeBoundaryParticleSpeciesX_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }

    auto partitionEnd = thrust::partition(
        particles.begin(), particles.begin() + countForFreeBoundaryParticles[0], 
        [] __device__ (const Particle& p) { return p.isExist; }
    );
    EXIST_NUM = static_cast<PICUnsignedLongLong>(thrust::distance(particles.begin(), partitionEnd));
}


__global__ void freeBoundaryParticleSpeciesY_kernel(
    const PICFloat DY, 
    const PICFloat YMIN, const PICFloat YMAX, 
    const PICUnsignedLongLong EXIST_NUM, 
    Particle* particles, 
    PICUnsignedLongLong* countForFreeBoundaryParticles
)
{
    PICUnsignedLongLong i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < EXIST_NUM) {
        PICFloat y = particles[i].y; 
        
        if (y <= YMIN + DY) {
            particles[i].isExist = false; 
        }
        if (y >= YMAX - DY) {
            particles[i].isExist = false; 
        }
        
        if (y > YMIN + DY && y <= YMIN + 2 * DY) {
            PICUnsignedLongLong particleIndex = atomicAdd(&(countForFreeBoundaryParticles[0]), 1);
            Particle sendParticle = particles[i];
            sendParticle.y = sendParticle.y - DY; 
            particles[particleIndex] = sendParticle;
        }

        if (y < YMAX - DY && y >= YMAX - 2 * DY) {
            PICUnsignedLongLong particleIndex = atomicAdd(&(countForFreeBoundaryParticles[0]), 1);
            Particle sendParticle = particles[i];
            sendParticle.y = sendParticle.y + DY; 
            particles[particleIndex] = sendParticle;
        }
    }
}

void PICBoundary::freeBoundaryParticleSpeciesY(
    thrust::device_vector<Particle>& particles, 
    PICUnsignedLongLong& EXIST_NUM
)
{   
    thrust::device_vector<PICUnsignedLongLong> countForFreeBoundaryParticles(1, EXIST_NUM); 

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((EXIST_NUM + threadsPerBlock.x - 1) / threadsPerBlock.x);

    freeBoundaryParticleSpeciesY_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        DY, 
        pICGridParameter.YMIN, pICGridParameter.YMAX, 
        EXIST_NUM, 
        thrust::raw_pointer_cast(particles.data()), 
        thrust::raw_pointer_cast(countForFreeBoundaryParticles.data())
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed at freeBoundaryParticleSpeciesY_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed at freeBoundaryParticleSpeciesY_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }

    auto partitionEnd = thrust::partition(
        particles.begin(), particles.begin() + countForFreeBoundaryParticles[0], 
        [] __device__ (const Particle& p) { return p.isExist; }
    );
    EXIST_NUM = static_cast<PICUnsignedLongLong>(thrust::distance(particles.begin(), partitionEnd));
}



template <typename FieldType>
__global__ void freeBoundaryFieldX_kernel(
    const PICUnsignedInt NX, const PICUnsignedInt NY, 
    FieldType* field
)
{
    PICUnsignedInt j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < NY) {
        PICUnsignedLongLong index; 
        index = getIndex<PICUnsignedLongLong>(static_cast<PICUnsignedInt>(0), j, NX, NY);
        field[index] = field[index + NY];
        index = getIndex<PICUnsignedLongLong>(static_cast<PICUnsignedInt>(NX - 1), j, NX, NY);
        field[index] = field[index - NY];
    }
}

template <typename FieldType>
void PICBoundary::freeBoundaryFieldX(
    thrust::device_vector<FieldType>& field
)
{
    dim3 threadsPerBlock(1, 256);
    dim3 blocksPerGrid(1,
                       (NY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    freeBoundaryFieldX_kernel<FieldType><<<blocksPerGrid, threadsPerBlock>>>(
        NX, NY, 
        thrust::raw_pointer_cast(field.data())
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed at freeBoundaryFieldX_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed at freeBoundaryFieldX_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
}

template void PICBoundary::freeBoundaryFieldX<MagneticField>(thrust::device_vector<MagneticField>&);
template void PICBoundary::freeBoundaryFieldX<ElectricField>(thrust::device_vector<ElectricField>&);
template void PICBoundary::freeBoundaryFieldX<CurrentField>(thrust::device_vector<CurrentField>&);
template void PICBoundary::freeBoundaryFieldX<ZerothMoment>(thrust::device_vector<ZerothMoment>&);
template void PICBoundary::freeBoundaryFieldX<FirstMoment>(thrust::device_vector<FirstMoment>&);
template void PICBoundary::freeBoundaryFieldX<SecondMoment>(thrust::device_vector<SecondMoment>&);


template <typename FieldType>
__global__ void freeBoundaryFieldY_kernel(
    const PICUnsignedInt NX, const PICUnsignedInt NY, 
    FieldType* field
)
{
    PICUnsignedInt i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < NX) {
        PICUnsignedLongLong index; 
        index = getIndex<PICUnsignedLongLong>(i, static_cast<PICUnsignedInt>(0), NX, NY);
        field[index] = field[index + 1];
        index = getIndex<PICUnsignedLongLong>(i, static_cast<PICUnsignedInt>(NY - 1), NX, NY);
        field[index] = field[index - 1];
    }
}

template <typename FieldType>
void PICBoundary::freeBoundaryFieldY(
    thrust::device_vector<FieldType>& field
)
{
    dim3 threadsPerBlock(256, 1);
    dim3 blocksPerGrid((NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       1);

    freeBoundaryFieldY_kernel<FieldType><<<blocksPerGrid, threadsPerBlock>>>(
        NX, NY, 
        thrust::raw_pointer_cast(field.data())
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed at freeBoundaryFieldY_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed at freeBoundaryFieldY_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
}

template void PICBoundary::freeBoundaryFieldY<MagneticField>(thrust::device_vector<MagneticField>&);
template void PICBoundary::freeBoundaryFieldY<ElectricField>(thrust::device_vector<ElectricField>&);
template void PICBoundary::freeBoundaryFieldY<CurrentField>(thrust::device_vector<CurrentField>&);
template void PICBoundary::freeBoundaryFieldY<ZerothMoment>(thrust::device_vector<ZerothMoment>&);
template void PICBoundary::freeBoundaryFieldY<FirstMoment>(thrust::device_vector<FirstMoment>&);
template void PICBoundary::freeBoundaryFieldY<SecondMoment>(thrust::device_vector<SecondMoment>&);

