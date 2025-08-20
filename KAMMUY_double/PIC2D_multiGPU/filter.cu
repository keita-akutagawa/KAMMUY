#include "filter.hpp"
#include <thrust/fill.h>


Filter::Filter(PIC2DMPI::MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo), 
      rho(mPIInfo.localSizeX * PIC2DConst::ny), 
      F(mPIInfo.localSizeX * PIC2DConst::ny), 
      momentCalculator(mPIInfo)
{
}


__global__ void calculateRho_kernel(
    RhoField* rho,  
    const ZerothMoment* zerothMomentIon, 
    const ZerothMoment* zerothMomentElectron, 
    const int localSizeX
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localSizeX && j < PIC2DConst::device_ny) {
        unsigned long long index = j + i * PIC2DConst::device_ny;

        rho[index].rho = PIC2DConst::device_qIon * zerothMomentIon[index].n
                       + PIC2DConst::device_qElectron * zerothMomentElectron[index].n; 

    }
}


void Filter::calculateRho(
    thrust::device_vector<ZerothMoment>& zerothMomentIon, 
    thrust::device_vector<ZerothMoment>& zerothMomentElectron, 
    const thrust::device_vector<Particle>& particlesIon, 
    const thrust::device_vector<Particle>& particlesElectron
)
{
    momentCalculator.calculateZerothMomentOfOneSpecies(
        zerothMomentIon, particlesIon, mPIInfo.existNumIonPerProcs
    ); 
    momentCalculator.calculateZerothMomentOfOneSpecies(
        zerothMomentElectron, particlesElectron, mPIInfo.existNumElectronPerProcs
    ); 


    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (PIC2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    calculateRho_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(rho.data()), 
        thrust::raw_pointer_cast(zerothMomentIon.data()), 
        thrust::raw_pointer_cast(zerothMomentElectron.data()), 
        mPIInfo.localSizeX
    );
    
}


__global__ void calculateF_kernel(
    FilterField* F, ElectricField* E, RhoField* rho, 
    int localSizeX
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < localSizeX - 1) && (0 < j) && (j < PIC2DConst::device_ny - 1)) {
        unsigned long long index = j + i * PIC2DConst::device_ny;

        F[index].F = ((E[index].eX - E[index - PIC2DConst::device_ny].eX) / PIC2DConst::device_dx 
                   + (E[index].eY - E[index - 1].eY) / PIC2DConst::device_dy)
                   - rho[index].rho / PIC2DConst::device_epsilon0;
    }
}

__global__ void correctE_kernel(
    ElectricField* E, FilterField* F, double dt, 
    int localSizeX
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < localSizeX - 1) && (0 < j) && (j < PIC2DConst::device_ny - 1)) {
        unsigned long long index = j + i * PIC2DConst::device_ny;

        E[index].eX += PIC2DConst::device_dOfLangdonMarderTypeCorrection
                     * (F[index + PIC2DConst::device_ny].F - F[index].F) / PIC2DConst::device_dx * dt;
        E[index].eY += PIC2DConst::device_dOfLangdonMarderTypeCorrection
                     * (F[index + 1].F - F[index].F) / PIC2DConst::device_dy * dt;
    }
}


void Filter::langdonMarderTypeCorrection(
    thrust::device_vector<ElectricField>& E, 
    const double dt
)
{
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


