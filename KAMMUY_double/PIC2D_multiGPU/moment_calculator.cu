#include "moment_calculator.hpp"


MomentCalculator::MomentCalculator(PIC2DMPI::MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo)
{
}


void MomentCalculator::resetZerothMomentOfOneSpecies(
    thrust::device_vector<ZerothMoment>& zerothMomentOfOneSpecies
)
{
    thrust::fill(
        zerothMomentOfOneSpecies.begin(), 
        zerothMomentOfOneSpecies.end(), 
        ZerothMoment()
    );
    cudaDeviceSynchronize();
}

void MomentCalculator::resetFirstMomentOfOneSpecies(
    thrust::device_vector<FirstMoment>& firstMomentOfOneSpecies
)
{
    thrust::fill(
        firstMomentOfOneSpecies.begin(), 
        firstMomentOfOneSpecies.end(), 
        FirstMoment()
    );
    cudaDeviceSynchronize();
}

void MomentCalculator::resetSecondMomentOfOneSpecies(
    thrust::device_vector<SecondMoment>& secondMomentOfOneSpecies
)
{
    thrust::fill(
        secondMomentOfOneSpecies.begin(), 
        secondMomentOfOneSpecies.end(), 
        SecondMoment()
    );
    cudaDeviceSynchronize();
}

//////////

__global__ void calculateZerothMomentOfOneSpecies_kernel(
    ZerothMoment* zerothMomentOfOneSpecies, 
    const Particle* particlesSpecies, 
    const unsigned long long existNumSpecies, 
    const int localNx, const int buffer, 
    const int localSizeX, 
    const double xminForProcs, const double xmaxForProcs
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {

        double cx1, cx2; 
        int xIndex1, xIndex2;
        double xOverDx;
        double cy1, cy2; 
        int yIndex1, yIndex2;
        double yOverDy;

        xOverDx = (particlesSpecies[i].x - xminForProcs + buffer * PIC2DConst::device_dx) / PIC2DConst::device_dx;
        yOverDy = (particlesSpecies[i].y - PIC2DConst::device_ymin) / PIC2DConst::device_dy;

        xIndex1 = floor(xOverDx);
        xIndex2 = xIndex1 + 1;
        xIndex2 = (xIndex2 == localSizeX) ? 0 : xIndex2;
        yIndex1 = floor(yOverDy);
        yIndex2 = yIndex1 + 1;
        yIndex2 = (yIndex2 == PIC2DConst::device_ny) ? 0 : yIndex2;

        if (xIndex1 < 0 || xIndex1 >= localSizeX) printf("x = %f, index = %d, ERROR\n", particlesSpecies[i].x, xIndex1); 
        if (yIndex1 < 0 || yIndex1 >= PIC2DConst::device_ny) printf("y = %f, index = %d, ERROR\n", particlesSpecies[i].y, yIndex1);

        cx1 = xOverDx - xIndex1;
        cx2 = 1.0 - cx1;
        cy1 = yOverDy - yIndex1;
        cy2 = 1.0 - cy1;

        unsigned long long index11 = yIndex1 + PIC2DConst::device_ny * xIndex1; 
        unsigned long long index12 = yIndex2 + PIC2DConst::device_ny * xIndex1; 
        unsigned long long index21 = yIndex1 + PIC2DConst::device_ny * xIndex2; 
        unsigned long long index22 = yIndex2 + PIC2DConst::device_ny * xIndex2; 
        atomicAdd(&(zerothMomentOfOneSpecies[index11].n), cx2 * cy2);
        atomicAdd(&(zerothMomentOfOneSpecies[index12].n), cx2 * cy1);
        atomicAdd(&(zerothMomentOfOneSpecies[index21].n), cx1 * cy2);
        atomicAdd(&(zerothMomentOfOneSpecies[index22].n), cx1 * cy1);
    }
};


void MomentCalculator::calculateZerothMomentOfOneSpecies(
    thrust::device_vector<ZerothMoment>& zerothMomentOfOneSpecies, 
    const thrust::device_vector<Particle>& particlesSpecies, 
    unsigned long long existNumSpecies
)
{
    resetZerothMomentOfOneSpecies(zerothMomentOfOneSpecies);

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((existNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);

    calculateZerothMomentOfOneSpecies_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(zerothMomentOfOneSpecies.data()), 
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        existNumSpecies, 
        mPIInfo.localNx, mPIInfo.buffer, 
        mPIInfo.localSizeX, 
        mPIInfo.xminForProcs, mPIInfo.xmaxForProcs
    );
    cudaDeviceSynchronize();
}



__global__ void calculateFirstMomentOfOneSpecies_kernel(
    FirstMoment* firstMomentOfOneSpecies, 
    const Particle* particlesSpecies, 
    const unsigned long long existNumSpecies, 
    const int localNx, const int buffer, 
    const int localSizeX, 
    const double xminForProcs, const double xmaxForProcs
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
    
        double cx1, cx2; 
        int xIndex1, xIndex2;
        double xOverDx;
        double cy1, cy2; 
        int yIndex1, yIndex2;
        double yOverDy;
        double vx, vy, vz;

        xOverDx = (particlesSpecies[i].x - xminForProcs + buffer * PIC2DConst::device_dx) / PIC2DConst::device_dx;
        yOverDy = (particlesSpecies[i].y - PIC2DConst::device_ymin) / PIC2DConst::device_dy;

        xIndex1 = floor(xOverDx);
        xIndex2 = xIndex1 + 1;
        xIndex2 = (xIndex2 == localSizeX) ? 0 : xIndex2;
        yIndex1 = floor(yOverDy);
        yIndex2 = yIndex1 + 1;
        yIndex2 = (yIndex2 == PIC2DConst::device_ny) ? 0 : yIndex2;
        
        if (xIndex1 < 0 || xIndex1 >= localSizeX) printf("x = %f, index = %d, ERROR\n", particlesSpecies[i].x, xIndex1); 
        if (yIndex1 < 0 || yIndex1 >= PIC2DConst::device_ny) printf("y = %f, index = %d, ERROR\n", particlesSpecies[i].y, yIndex1);

        cx1 = xOverDx - xIndex1;
        cx2 = 1.0 - cx1;
        cy1 = yOverDy - yIndex1;
        cy2 = 1.0 - cy1;

        vx = particlesSpecies[i].vx / particlesSpecies[i].gamma;
        vy = particlesSpecies[i].vy / particlesSpecies[i].gamma;
        vz = particlesSpecies[i].vz / particlesSpecies[i].gamma;

        unsigned long long index11 = yIndex1 + PIC2DConst::device_ny * xIndex1; 
        unsigned long long index12 = yIndex2 + PIC2DConst::device_ny * xIndex1; 
        unsigned long long index21 = yIndex1 + PIC2DConst::device_ny * xIndex2; 
        unsigned long long index22 = yIndex2 + PIC2DConst::device_ny * xIndex2; 

        atomicAdd(&(firstMomentOfOneSpecies[index11].x), vx * cx2 * cy2);
        atomicAdd(&(firstMomentOfOneSpecies[index12].x), vx * cx2 * cy1);
        atomicAdd(&(firstMomentOfOneSpecies[index21].x), vx * cx1 * cy2);
        atomicAdd(&(firstMomentOfOneSpecies[index22].x), vx * cx1 * cy1);

        atomicAdd(&(firstMomentOfOneSpecies[index11].y), vy * cx2 * cy2);
        atomicAdd(&(firstMomentOfOneSpecies[index12].y), vy * cx2 * cy1);
        atomicAdd(&(firstMomentOfOneSpecies[index21].y), vy * cx1 * cy2);
        atomicAdd(&(firstMomentOfOneSpecies[index22].y), vy * cx1 * cy1);

        atomicAdd(&(firstMomentOfOneSpecies[index11].z), vz * cx2 * cy2);
        atomicAdd(&(firstMomentOfOneSpecies[index12].z), vz * cx2 * cy1);
        atomicAdd(&(firstMomentOfOneSpecies[index21].z), vz * cx1 * cy2);
        atomicAdd(&(firstMomentOfOneSpecies[index22].z), vz * cx1 * cy1);
    }
};


void MomentCalculator::calculateFirstMomentOfOneSpecies(
    thrust::device_vector<FirstMoment>& firstMomentOfOneSpecies, 
    const thrust::device_vector<Particle>& particlesSpecies, 
    unsigned long long existNumSpecies
)
{
    resetFirstMomentOfOneSpecies(firstMomentOfOneSpecies);

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((existNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);

    calculateFirstMomentOfOneSpecies_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(firstMomentOfOneSpecies.data()), 
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        existNumSpecies, 
        mPIInfo.localNx, mPIInfo.buffer, 
        mPIInfo.localSizeX, 
        mPIInfo.xminForProcs, mPIInfo.xmaxForProcs
    );
    cudaDeviceSynchronize();
}


__global__ void calculateSecondMomentOfOneSpecies_kernel(
    SecondMoment* secondMomentOfOneSpecies, 
    const Particle* particlesSpecies, 
    const unsigned long long existNumSpecies, 
    const int localNx, const int buffer, 
    const int localSizeX,
    const double xminForProcs, const double xmaxForProcs
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
    
        double cx1, cx2; 
        int xIndex1, xIndex2;
        double xOverDx;
        double cy1, cy2; 
        int yIndex1, yIndex2;
        double yOverDy;
        double vx, vy, vz;

        xOverDx = (particlesSpecies[i].x - xminForProcs + buffer * PIC2DConst::device_dx) / PIC2DConst::device_dx;
        yOverDy = (particlesSpecies[i].y - PIC2DConst::device_ymin) / PIC2DConst::device_dy;

        xIndex1 = floor(xOverDx);
        xIndex2 = xIndex1 + 1;
        xIndex2 = (xIndex2 == localSizeX) ? 0 : xIndex2;
        yIndex1 = floor(yOverDy);
        yIndex2 = yIndex1 + 1;
        yIndex2 = (yIndex2 == PIC2DConst::device_ny) ? 0 : yIndex2;
        
        if (xIndex1 < 0 || xIndex1 >= localSizeX) printf("x = %f, index = %d, ERROR\n", particlesSpecies[i].x, xIndex1); 
        if (yIndex1 < 0 || yIndex1 >= PIC2DConst::device_ny) printf("y = %f, index = %d, ERROR\n", particlesSpecies[i].y, yIndex1);

        cx1 = xOverDx - xIndex1;
        cx2 = 1.0 - cx1;
        cy1 = yOverDy - yIndex1;
        cy2 = 1.0 - cy1;

        vx = particlesSpecies[i].vx / particlesSpecies[i].gamma;
        vy = particlesSpecies[i].vy / particlesSpecies[i].gamma;
        vz = particlesSpecies[i].vz / particlesSpecies[i].gamma;

        unsigned long long index11 = yIndex1 + PIC2DConst::device_ny * xIndex1; 
        unsigned long long index12 = yIndex2 + PIC2DConst::device_ny * xIndex1; 
        unsigned long long index21 = yIndex1 + PIC2DConst::device_ny * xIndex2; 
        unsigned long long index22 = yIndex2 + PIC2DConst::device_ny * xIndex2; 

        atomicAdd(&(secondMomentOfOneSpecies[index11].xx), vx * vx * cx2 * cy2);
        atomicAdd(&(secondMomentOfOneSpecies[index12].xx), vx * vx * cx2 * cy1);
        atomicAdd(&(secondMomentOfOneSpecies[index21].xx), vx * vx * cx1 * cy2);
        atomicAdd(&(secondMomentOfOneSpecies[index22].xx), vx * vx * cx1 * cy1);

        atomicAdd(&(secondMomentOfOneSpecies[index11].yy), vy * vy * cx2 * cy2);
        atomicAdd(&(secondMomentOfOneSpecies[index12].yy), vy * vy * cx2 * cy1);
        atomicAdd(&(secondMomentOfOneSpecies[index21].yy), vy * vy * cx1 * cy2);
        atomicAdd(&(secondMomentOfOneSpecies[index22].yy), vy * vy * cx1 * cy1);

        atomicAdd(&(secondMomentOfOneSpecies[index11].zz), vz * vz * cx2 * cy2);
        atomicAdd(&(secondMomentOfOneSpecies[index12].zz), vz * vz * cx2 * cy1);
        atomicAdd(&(secondMomentOfOneSpecies[index21].zz), vz * vz * cx1 * cy2);
        atomicAdd(&(secondMomentOfOneSpecies[index22].zz), vz * vz * cx1 * cy1);

        atomicAdd(&(secondMomentOfOneSpecies[index11].xy), vx * vy * cx2 * cy2);
        atomicAdd(&(secondMomentOfOneSpecies[index12].xy), vx * vy * cx2 * cy1);
        atomicAdd(&(secondMomentOfOneSpecies[index21].xy), vx * vy * cx1 * cy2);
        atomicAdd(&(secondMomentOfOneSpecies[index22].xy), vx * vy * cx1 * cy1);

        atomicAdd(&(secondMomentOfOneSpecies[index11].xz), vx * vz * cx2 * cy2);
        atomicAdd(&(secondMomentOfOneSpecies[index12].xz), vx * vz * cx2 * cy1);
        atomicAdd(&(secondMomentOfOneSpecies[index21].xz), vx * vz * cx1 * cy2);
        atomicAdd(&(secondMomentOfOneSpecies[index22].xz), vx * vz * cx1 * cy1);

        atomicAdd(&(secondMomentOfOneSpecies[index11].yz), vy * vz * cx2 * cy2);
        atomicAdd(&(secondMomentOfOneSpecies[index12].yz), vy * vz * cx2 * cy1);
        atomicAdd(&(secondMomentOfOneSpecies[index21].yz), vy * vz * cx1 * cy2);
        atomicAdd(&(secondMomentOfOneSpecies[index22].yz), vy * vz * cx1 * cy1);
    }
};


void MomentCalculator::calculateSecondMomentOfOneSpecies(
    thrust::device_vector<SecondMoment>& secondMomentOfOneSpecies, 
    const thrust::device_vector<Particle>& particlesSpecies, 
    unsigned long long existNumSpecies
)
{
    resetSecondMomentOfOneSpecies(secondMomentOfOneSpecies);

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((existNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);

    calculateSecondMomentOfOneSpecies_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(secondMomentOfOneSpecies.data()), 
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        existNumSpecies, 
        mPIInfo.localNx, mPIInfo.buffer, 
        mPIInfo.localSizeX, 
        mPIInfo.xminForProcs, mPIInfo.xmaxForProcs
    );
    cudaDeviceSynchronize();
}



