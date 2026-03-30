#include "boundary.hpp"


void BoundaryPIC::freeBoundaryParticle_x(
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron
)
{   
    freeBoundaryParticleOfOneSpecies_x(
        particlesIon,
        PIC2DConst::existNumIon
    );
    freeBoundaryParticleOfOneSpecies_x(
        particlesElectron, 
        PIC2DConst::existNumElectron
    );

    if (PIC2DConst::existNumIon > PIC2DConst::totalNumIon) std::cout << "BROKEN" << std::endl;
    if (PIC2DConst::existNumElectron > PIC2DConst::totalNumElectron) std::cout << "BROKEN" << std::endl;
}


__global__ void freeBoundaryParticleXLeft_kernel(
    Particle* particlesSpecies, 
    Particle* bufferParticlesSpeciesX, 
    unsigned long long* countForFreeBoundaryParticlesSpeciesXLeft, 
    const unsigned long long existNumSpecies
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        double x = particlesSpecies[i].x; 
        
        double boundaryXLeft = PIC2DConst::device_xmin; 
        
        if (x <= boundaryXLeft + PIC2DConst::device_dx) {
            particlesSpecies[i].isExist = false; 
        }
        
        if (x > boundaryXLeft + PIC2DConst::device_dx && x <= boundaryXLeft + 2 * PIC2DConst::device_dx) {
            unsigned long long particleIndex = atomicAdd(&(countForFreeBoundaryParticlesSpeciesXLeft[0]), 1);
            Particle tmpParticle = particlesSpecies[i];
            tmpParticle.x = tmpParticle.x - PIC2DConst::device_dx + PIC2DConst::device_EPS; 
            bufferParticlesSpeciesX[particleIndex] = tmpParticle;
        }
    }
}

__global__ void freeBoundaryParticleXRight_kernel(
    Particle* particlesSpecies, 
    Particle* bufferParticlesSpeciesX, 
    unsigned long long* countForFreeBoundaryParticlesSpeciesXRight, 
    const unsigned long long existNumSpecies
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        double x = particlesSpecies[i].x; 
        
        double boundaryXRight = PIC2DConst::device_xmax;
        
        if (x >= boundaryXRight - PIC2DConst::device_dx) {
            particlesSpecies[i].isExist = false; 
        }

        if (x < boundaryXRight - PIC2DConst::device_dx && x >= boundaryXRight - 2 * PIC2DConst::device_dx) {
            unsigned long long particleIndex = atomicAdd(&(countForFreeBoundaryParticlesSpeciesXRight[0]), 1);
            Particle tmpParticle = particlesSpecies[i];
            tmpParticle.x = tmpParticle.x + PIC2DConst::device_dx - PIC2DConst::device_EPS; 
            bufferParticlesSpeciesX[particleIndex] = tmpParticle;
        }
    }
}

void BoundaryPIC::freeBoundaryParticleOfOneSpecies_x(
    thrust::device_vector<Particle>& particlesSpecies, 
    unsigned long long& existNumSpecies
)
{   
    //boundary for x left

    dim3 threadsPerBlockForXLeft(256);
    dim3 blocksPerGridForXLeft((existNumSpecies + threadsPerBlockForXLeft.x - 1) / threadsPerBlockForXLeft.x);
    
    thrust::device_vector<unsigned long long> countForFreeBoundaryParticlesSpeciesXLeft(1, 0); 

    freeBoundaryParticleXLeft_kernel<<<blocksPerGridForXLeft, threadsPerBlockForXLeft>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        thrust::raw_pointer_cast(bufferParticlesSpeciesX.data()), 
        thrust::raw_pointer_cast(countForFreeBoundaryParticlesSpeciesXLeft.data()), 
        existNumSpecies
    );
    cudaDeviceSynchronize();

    auto partitionEndXLeft = thrust::partition(
        particlesSpecies.begin(), particlesSpecies.begin() + existNumSpecies, 
        [] __device__ (const Particle& p) { return p.isExist; }
    );
    existNumSpecies = static_cast<unsigned long long>(thrust::distance(particlesSpecies.begin(), partitionEndXLeft));

    thrust::copy(
        bufferParticlesSpeciesX.begin(), 
        bufferParticlesSpeciesX.begin() + countForFreeBoundaryParticlesSpeciesXLeft[0],
        particlesSpecies.begin() + existNumSpecies
    );
    existNumSpecies += countForFreeBoundaryParticlesSpeciesXLeft[0];

    
    //boundary for x right

    dim3 threadsPerBlockForXRight(256);
    dim3 blocksPerGridForXRight((existNumSpecies + threadsPerBlockForXRight.x - 1) / threadsPerBlockForXRight.x);
    
    thrust::device_vector<unsigned long long> countForFreeBoundaryParticlesSpeciesXRight(1, 0); 

    freeBoundaryParticleXRight_kernel<<<blocksPerGridForXRight, threadsPerBlockForXRight>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        thrust::raw_pointer_cast(bufferParticlesSpeciesX.data()), 
        thrust::raw_pointer_cast(countForFreeBoundaryParticlesSpeciesXRight.data()), 
        existNumSpecies
    );
    cudaDeviceSynchronize();

    auto partitionEndXRight = thrust::partition(
        particlesSpecies.begin(), particlesSpecies.begin() + existNumSpecies, 
        [] __device__ (const Particle& p) { return p.isExist; }
    );
    existNumSpecies = static_cast<unsigned long long>(thrust::distance(particlesSpecies.begin(), partitionEndXRight));

    thrust::copy(
        bufferParticlesSpeciesX.begin(), 
        bufferParticlesSpeciesX.begin() + countForFreeBoundaryParticlesSpeciesXRight[0],
        particlesSpecies.begin() + existNumSpecies
    );
    existNumSpecies += countForFreeBoundaryParticlesSpeciesXRight[0];
}


void BoundaryPIC::freeBoundaryParticle_y(
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron
)
{   
    freeBoundaryParticleOfOneSpecies_y(
        particlesIon,
        PIC2DConst::existNumIon
    );
    freeBoundaryParticleOfOneSpecies_y(
        particlesElectron, 
        PIC2DConst::existNumElectron
    );

    if (PIC2DConst::existNumIon > PIC2DConst::totalNumIon) std::cout << "BROKEN" << std::endl;
    if (PIC2DConst::existNumElectron > PIC2DConst::totalNumElectron) std::cout << "BROKEN" << std::endl;
}


__global__ void freeBoundaryParticleYDown_kernel(
    Particle* particlesSpecies, 
    Particle* bufferParticlesSpeciesY, 
    unsigned long long* countForFreeBoundaryParticlesSpeciesYDown, 
    const unsigned long long existNumSpecies
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        double y = particlesSpecies[i].y; 
        
        double boundaryYDown = PIC2DConst::device_ymin; 
        
        if (y <= boundaryYDown + PIC2DConst::device_dy) {
            particlesSpecies[i].isExist = false; 
        }
        
        if (y > boundaryYDown + PIC2DConst::device_dy && y <= boundaryYDown + 2 * PIC2DConst::device_dy) {
            unsigned long long particleIndex = atomicAdd(&(countForFreeBoundaryParticlesSpeciesYDown[0]), 1);
            Particle tmpParticle = particlesSpecies[i];
            tmpParticle.y = tmpParticle.y - PIC2DConst::device_dy + PIC2DConst::device_EPS; 
            bufferParticlesSpeciesY[particleIndex] = tmpParticle;
        }
    }
}

__global__ void freeBoundaryParticleYUp_kernel(
    Particle* particlesSpecies, 
    Particle* bufferParticlesSpeciesY, 
    unsigned long long* countForFreeBoundaryParticlesSpeciesYUp, 
    const unsigned long long existNumSpecies
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        double y = particlesSpecies[i].y; 
        
        double boundaryYUp = PIC2DConst::device_ymax;
        
        if (y >= boundaryYUp - PIC2DConst::device_dy) {
            particlesSpecies[i].isExist = false; 
        }

        if (y < boundaryYUp - PIC2DConst::device_dy && y >= boundaryYUp - 2 * PIC2DConst::device_dy) {
            unsigned long long particleIndex = atomicAdd(&(countForFreeBoundaryParticlesSpeciesYUp[0]), 1);
            Particle tmpParticle = particlesSpecies[i];
            tmpParticle.y = tmpParticle.y + PIC2DConst::device_dy - PIC2DConst::device_EPS; 
            bufferParticlesSpeciesY[particleIndex] = tmpParticle;
        }
    }
}

void BoundaryPIC::freeBoundaryParticleOfOneSpecies_y(
    thrust::device_vector<Particle>& particlesSpecies, 
    unsigned long long& existNumSpecies
)
{   
    //boundary for y down

    dim3 threadsPerBlockForYDown(256);
    dim3 blocksPerGridForYDown((existNumSpecies + threadsPerBlockForYDown.x - 1) / threadsPerBlockForYDown.x);
    
    thrust::device_vector<unsigned long long> countForFreeBoundaryParticlesSpeciesYDown(1, 0); 

    freeBoundaryParticleYDown_kernel<<<blocksPerGridForYDown, threadsPerBlockForYDown>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        thrust::raw_pointer_cast(bufferParticlesSpeciesY.data()), 
        thrust::raw_pointer_cast(countForFreeBoundaryParticlesSpeciesYDown.data()), 
        existNumSpecies
    );
    cudaDeviceSynchronize();

    auto partitionEndYDown = thrust::partition(
        particlesSpecies.begin(), particlesSpecies.begin() + existNumSpecies, 
        [] __device__ (const Particle& p) { return p.isExist; }
    );
    existNumSpecies = static_cast<unsigned long long>(thrust::distance(particlesSpecies.begin(), partitionEndYDown));

    thrust::copy(
        bufferParticlesSpeciesY.begin(), 
        bufferParticlesSpeciesY.begin() + countForFreeBoundaryParticlesSpeciesYDown[0],
        particlesSpecies.begin() + existNumSpecies
    );
    existNumSpecies += countForFreeBoundaryParticlesSpeciesYDown[0];

    
    //boundary for y up

    dim3 threadsPerBlockForYUp(256);
    dim3 blocksPerGridForYUp((existNumSpecies + threadsPerBlockForYUp.x - 1) / threadsPerBlockForYUp.x);
    
    thrust::device_vector<unsigned long long> countForFreeBoundaryParticlesSpeciesYUp(1, 0); 

    freeBoundaryParticleYUp_kernel<<<blocksPerGridForYUp, threadsPerBlockForYUp>>>(
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        thrust::raw_pointer_cast(bufferParticlesSpeciesY.data()), 
        thrust::raw_pointer_cast(countForFreeBoundaryParticlesSpeciesYUp.data()), 
        existNumSpecies
    );
    cudaDeviceSynchronize();

    auto partitionEndYUp = thrust::partition(
        particlesSpecies.begin(), particlesSpecies.begin() + existNumSpecies, 
        [] __device__ (const Particle& p) { return p.isExist; }
    );
    existNumSpecies = static_cast<unsigned long long>(thrust::distance(particlesSpecies.begin(), partitionEndYUp));

    thrust::copy(
        bufferParticlesSpeciesY.begin(), 
        bufferParticlesSpeciesY.begin() + countForFreeBoundaryParticlesSpeciesYUp[0],
        particlesSpecies.begin() + existNumSpecies
    );
    existNumSpecies += countForFreeBoundaryParticlesSpeciesYUp[0];
}
