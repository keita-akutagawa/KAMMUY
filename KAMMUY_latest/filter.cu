#include "filter.hpp"
#include <thrust/fill.h>


Filter::Filter(PIC2DMPI::MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo), 
      rho(mPIInfo.localSizeX * PIC2DConst::ny), 
      F(mPIInfo.localSizeX * PIC2DConst::ny)
{
}


__global__ void calculateF_kernel(
    FilterField* F, ElectricField* E, RhoField* rho, 
    int localSizeX
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < localSizeX - 1) && (0 < j) && (j < PIC2DConst::device_ny - 1)) {
        int index = j + i * PIC2DConst::device_ny;

        F[index].F = ((E[index].eX - E[index - PIC2DConst::device_ny].eX) / PIC2DConst::device_dx 
                   + (E[index].eY - E[index - 1].eY) / PIC2DConst::device_dy)
                   - rho[index].rho / PIC2DConst::device_epsilon0;
    }
}

__global__ void correctE_kernel(
    ElectricField* E, FilterField* F, float dt, 
    int localSizeX
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < localSizeX - 1) && (0 < j) && (j < PIC2DConst::device_ny - 1)) {
        int index = j + i * PIC2DConst::device_ny;

        E[index].eX += PIC2DConst::device_dOfLangdonMarderTypeCorrection
                     * (F[index + PIC2DConst::device_ny].F - F[index].F) / PIC2DConst::device_dx * dt;
        E[index].eY += PIC2DConst::device_dOfLangdonMarderTypeCorrection
                     * (F[index + 1].F - F[index].F) / PIC2DConst::device_dy * dt;
    }
}


void Filter::langdonMarderTypeCorrection(
    thrust::device_vector<ElectricField>& E, 
    const thrust::device_vector<Particle>& particlesIon, 
    const thrust::device_vector<Particle>& particlesElectron, 
    const float dt
)
{
    calculateRho(particlesIon, particlesElectron);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (PIC2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    calculateF_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(F.data()), 
        thrust::raw_pointer_cast(E.data()), 
        thrust::raw_pointer_cast(rho.data()), 
        mPIInfo.localSizeX
    );
    cudaDeviceSynchronize();

    correctE_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(E.data()), 
        thrust::raw_pointer_cast(F.data()), 
        dt, 
        mPIInfo.localSizeX
    );
    cudaDeviceSynchronize();
}


void Filter::resetRho()
{
    thrust::fill(rho.begin(), rho.end(), RhoField());
}


void Filter::calculateRho(
    const thrust::device_vector<Particle>& particlesIon, 
    const thrust::device_vector<Particle>& particlesElectron
)
{
    resetRho();

    calculateRhoOfOneSpecies(particlesIon, PIC2DConst::qIon, mPIInfo.existNumIonPerProcs);
    calculateRhoOfOneSpecies(particlesElectron, PIC2DConst::qElectron, mPIInfo.existNumElectronPerProcs);
}


__global__ void calculateRhoOfOneSpecies_kernel(
    RhoField* rho, const Particle* particlesSpecies, 
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

        xOverDx = (particlesSpecies[i].x - xminForProcs + buffer * PIC2DConst::device_dx) / PIC2DConst::device_dx;
        yOverDy = (particlesSpecies[i].y - PIC2DConst::device_ymin) / PIC2DConst::device_dy;

        xIndex1 = floorf(xOverDx);
        xIndex2 = xIndex1 + 1;
        xIndex2 = (xIndex2 == localSizeX) ? 0 : xIndex2;
        yIndex1 = floorf(yOverDy);
        yIndex2 = yIndex1 + 1;
        yIndex2 = (yIndex2 == PIC2DConst::device_ny) ? 0 : yIndex2;

        cx1 = xOverDx - xIndex1;
        cx2 = 1.0f - cx1;
        cy1 = yOverDy - yIndex1;
        cy2 = 1.0f - cy1;

        atomicAdd(&(rho[yIndex1 + PIC2DConst::device_ny * xIndex1].rho), q * cx2 * cy2);
        atomicAdd(&(rho[yIndex2 + PIC2DConst::device_ny * xIndex1].rho), q * cx2 * cy1);
        atomicAdd(&(rho[yIndex1 + PIC2DConst::device_ny * xIndex2].rho), q * cx1 * cy2);
        atomicAdd(&(rho[yIndex2 + PIC2DConst::device_ny * xIndex2].rho), q * cx1 * cy1);
    }
};


void Filter::calculateRhoOfOneSpecies(
    const thrust::device_vector<Particle>& particlesSpecies, 
    float q, unsigned long long existNumSpecies
)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((existNumSpecies + threadsPerBlock.x - 1) / threadsPerBlock.x);

    calculateRhoOfOneSpecies_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(rho.data()), 
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        q, existNumSpecies, 
        mPIInfo.localNx, mPIInfo.buffer, 
        mPIInfo.localSizeX, 
        mPIInfo.xminForProcs, mPIInfo.xmaxForProcs
    );
    cudaDeviceSynchronize();
}



