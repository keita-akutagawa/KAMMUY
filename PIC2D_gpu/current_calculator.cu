#include <cmath>
#include "current_calculator.hpp"
#include <thrust/fill.h>


using namespace PIC2DConst;

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
        current, particlesIon, qIon_PIC, existNumIon_PIC
    );
    calculateCurrentOfOneSpecies(
        current, particlesElectron, qElectron_PIC, existNumElectron_PIC
    );
}


__global__ void calculateCurrentOfOneSpecies_kernel(
    CurrentField* current,
    const Particle* particlesSpecies, 
    const double q, const unsigned long long existNumSpecies
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
        double qOverGamma, qVxOverGamma, qVyOverGamma, qVzOverGamma;

        xOverDx = particlesSpecies[i].x / device_dx_PIC;
        yOverDy = particlesSpecies[i].y / device_dy_PIC;

        xIndex1 = floorf(xOverDx);
        xIndex2 = xIndex1 + 1;
        xIndex2 = (xIndex2 == device_nx_PIC) ? 0 : xIndex2;
        yIndex1 = floorf(yOverDy);
        yIndex2 = yIndex1 + 1;
        yIndex2 = (yIndex2 == device_ny_PIC) ? 0 : yIndex2;

        cx1 = xOverDx - xIndex1;
        cx2 = 1.0 - cx1;
        cy1 = yOverDy - yIndex1;
        cy2 = 1.0 - cy1;

        qOverGamma = q / particlesSpecies[i].gamma;
        qVxOverGamma = qOverGamma * particlesSpecies[i].vx;
        qVyOverGamma = qOverGamma * particlesSpecies[i].vy;
        qVzOverGamma = qOverGamma * particlesSpecies[i].vz;

        atomicAdd(&(current[yIndex1 + device_ny_PIC * xIndex1].jX), qVxOverGamma * cx2 * cy2);
        atomicAdd(&(current[yIndex2 + device_ny_PIC * xIndex1].jX), qVxOverGamma * cx2 * cy1 * min(1, yIndex2));
        atomicAdd(&(current[yIndex1 + device_ny_PIC * xIndex2].jX), qVxOverGamma * cx1 * cy2 * min(1, yIndex2));
        atomicAdd(&(current[yIndex2 + device_ny_PIC * xIndex2].jX), qVxOverGamma * cx1 * cy1 * min(1, yIndex2));

        atomicAdd(&(current[yIndex1 + device_ny_PIC * xIndex1].jY), qVyOverGamma * cx2 * cy2);
        atomicAdd(&(current[yIndex2 + device_ny_PIC * xIndex1].jY), qVyOverGamma * cx2 * cy1 * min(1, yIndex2));
        atomicAdd(&(current[yIndex1 + device_ny_PIC * xIndex2].jY), qVyOverGamma * cx1 * cy2 * min(1, yIndex2));
        atomicAdd(&(current[yIndex2 + device_ny_PIC * xIndex2].jY), qVyOverGamma * cx1 * cy1 * min(1, yIndex2));

        atomicAdd(&(current[yIndex1 + device_ny_PIC * xIndex1].jZ), qVzOverGamma * cx2 * cy2);
        atomicAdd(&(current[yIndex2 + device_ny_PIC * xIndex1].jZ), qVzOverGamma * cx2 * cy1 * min(1, yIndex2));
        atomicAdd(&(current[yIndex1 + device_ny_PIC * xIndex2].jZ), qVzOverGamma * cx1 * cy2 * min(1, yIndex2));
        atomicAdd(&(current[yIndex2 + device_ny_PIC * xIndex2].jZ), qVzOverGamma * cx1 * cy1 * min(1, yIndex2));
    }
};


void CurrentCalculator::calculateCurrentOfOneSpecies(
    thrust::device_vector<CurrentField>& current, 
    const thrust::device_vector<Particle>& particlesSpecies, 
    double q, int existNumSpecies
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((existNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);

    calculateCurrentOfOneSpecies_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(current.data()), 
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        q, existNumSpecies
    );

    cudaDeviceSynchronize();
}


