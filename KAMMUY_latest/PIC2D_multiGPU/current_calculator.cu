#include <cmath>
#include "current_calculator.hpp"
#include <thrust/fill.h>


CurrentCalculator::CurrentCalculator(PIC2DMPI::MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo)
{
}


void CurrentCalculator::resetCurrent(
    thrust::device_vector<CurrentField>& current
)
{
    thrust::fill(current.begin(), current.end(), CurrentField());
}


void CurrentCalculator::calculateCurrent(
    thrust::device_vector<CurrentField>& current, 
    const thrust::device_vector<Particle>& particlesIon, 
    const thrust::device_vector<Particle>& particlesElectron
)
{
    calculateCurrentOfOneSpecies(
        current, particlesIon, PIC2DConst::qIon, mPIInfo.existNumIonPerProcs
    );
    calculateCurrentOfOneSpecies(
        current, particlesElectron, PIC2DConst::qElectron, mPIInfo.existNumElectronPerProcs
    );
}


__global__ void calculateCurrentOfOneSpecies_kernel(
    CurrentField* current,
    const Particle* particlesSpecies, 
    const float q, const unsigned long long existNumSpecies, 
    const int localNx, const int buffer, 
    const int localSizeX, 
    const float xminForProcs, const float xmaxForProcs
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
    
        float cx1, cx2; 
        int xIndex1, xIndex2;
        float xOverDx;
        float cy1, cy2; 
        int yIndex1, yIndex2;
        float yOverDy;
        float qOverGamma, qVxOverGamma, qVyOverGamma, qVzOverGamma;

        xOverDx = (particlesSpecies[i].x - xminForProcs + buffer * PIC2DConst::device_dx) / PIC2DConst::device_dx;
        yOverDy = (particlesSpecies[i].y - PIC2DConst::device_ymin) / PIC2DConst::device_dy;

        xIndex1 = floorf(xOverDx);
        xIndex2 = xIndex1 + 1;
        xIndex2 = (xIndex2 == localSizeX) ? 0 : xIndex2;
        yIndex1 = floorf(yOverDy);
        yIndex2 = yIndex1 + 1;
        yIndex2 = (yIndex2 == PIC2DConst::device_ny) ? 0 : yIndex2;

        if (xIndex1 < 0 || xIndex1 >= localSizeX) {
            //printf("ERROR %f %d %d\n", particlesSpecies[i].x, xIndex1, particlesSpecies[i].isExist);
            return;
        }
        if (yIndex1 < 0 || yIndex1 >= PIC2DConst::device_ny) {
            //printf("ERROR %f %d %d\n", particlesSpecies[i].x, xIndex1, particlesSpecies[i].isExist);
            return;
        }
        
        cx1 = xOverDx - xIndex1;
        cx2 = 1.0f - cx1;
        cy1 = yOverDy - yIndex1;
        cy2 = 1.0f - cy1;

        qOverGamma = q / particlesSpecies[i].gamma;
        qVxOverGamma = qOverGamma * particlesSpecies[i].vx;
        qVyOverGamma = qOverGamma * particlesSpecies[i].vy;
        qVzOverGamma = qOverGamma * particlesSpecies[i].vz;

        atomicAdd(&(current[yIndex1 + PIC2DConst::device_ny * xIndex1].jX), qVxOverGamma * cx2 * cy2);
        atomicAdd(&(current[yIndex2 + PIC2DConst::device_ny * xIndex1].jX), qVxOverGamma * cx2 * cy1);
        atomicAdd(&(current[yIndex1 + PIC2DConst::device_ny * xIndex2].jX), qVxOverGamma * cx1 * cy2);
        atomicAdd(&(current[yIndex2 + PIC2DConst::device_ny * xIndex2].jX), qVxOverGamma * cx1 * cy1);

        atomicAdd(&(current[yIndex1 + PIC2DConst::device_ny * xIndex1].jY), qVyOverGamma * cx2 * cy2);
        atomicAdd(&(current[yIndex2 + PIC2DConst::device_ny * xIndex1].jY), qVyOverGamma * cx2 * cy1);
        atomicAdd(&(current[yIndex1 + PIC2DConst::device_ny * xIndex2].jY), qVyOverGamma * cx1 * cy2);
        atomicAdd(&(current[yIndex2 + PIC2DConst::device_ny * xIndex2].jY), qVyOverGamma * cx1 * cy1);

        atomicAdd(&(current[yIndex1 + PIC2DConst::device_ny * xIndex1].jZ), qVzOverGamma * cx2 * cy2);
        atomicAdd(&(current[yIndex2 + PIC2DConst::device_ny * xIndex1].jZ), qVzOverGamma * cx2 * cy1);
        atomicAdd(&(current[yIndex1 + PIC2DConst::device_ny * xIndex2].jZ), qVzOverGamma * cx1 * cy2);
        atomicAdd(&(current[yIndex2 + PIC2DConst::device_ny * xIndex2].jZ), qVzOverGamma * cx1 * cy1);
    }
};


void CurrentCalculator::calculateCurrentOfOneSpecies(
    thrust::device_vector<CurrentField>& current, 
    const thrust::device_vector<Particle>& particlesSpecies, 
    float q, unsigned long long existNumSpecies
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((existNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);

    calculateCurrentOfOneSpecies_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(current.data()), 
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        q, existNumSpecies, 
        mPIInfo.localNx, mPIInfo.buffer, 
        mPIInfo.localSizeX, 
        mPIInfo.xminForProcs, mPIInfo.xmaxForProcs
    );
    cudaDeviceSynchronize();
}


