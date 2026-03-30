#include "filter.hpp"
#include <thrust/fill.h>


Filter::Filter()
    : rho(PIC2DConst::nx * PIC2DConst::ny), 
      F(PIC2DConst::nx * PIC2DConst::ny), 
      momentCalculator()
{
}


__global__ void calculateRho_kernel(
    RhoField* rho,  
    const ZerothMoment* zerothMomentIon, 
    const ZerothMoment* zerothMomentElectron
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < PIC2DConst::device_nx && j < PIC2DConst::device_ny) {
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
        zerothMomentIon, particlesIon, PIC2DConst::existNumIon
    ); 
    momentCalculator.calculateZerothMomentOfOneSpecies(
        zerothMomentElectron, particlesElectron, PIC2DConst::existNumElectron
    ); 


    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((PIC2DConst::nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (PIC2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    calculateRho_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(rho.data()), 
        thrust::raw_pointer_cast(zerothMomentIon.data()), 
        thrust::raw_pointer_cast(zerothMomentElectron.data())
    );
    
}


__global__ void calculateF_kernel(
    FilterField* F, ElectricField* E, RhoField* rho
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < PIC2DConst::device_nx - 1) && (0 < j) && (j < PIC2DConst::device_ny - 1)) {
        unsigned long long index = j + i * PIC2DConst::device_ny;

        F[index].F = ((E[index].eX - E[index - PIC2DConst::device_ny].eX) / PIC2DConst::device_dx 
                   + (E[index].eY - E[index - 1].eY) / PIC2DConst::device_dy)
                   - rho[index].rho / PIC2DConst::device_epsilon0;
    }
}

__global__ void correctE_kernel(
    ElectricField* E, FilterField* F, double dt
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < PIC2DConst::device_nx - 1) && (0 < j) && (j < PIC2DConst::device_ny - 1)) {
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
    dim3 blocksPerGrid((PIC2DConst::nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (PIC2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    calculateF_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(F.data()), 
        thrust::raw_pointer_cast(E.data()), 
        thrust::raw_pointer_cast(rho.data())
    );
    cudaDeviceSynchronize();

    correctE_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(E.data()), 
        thrust::raw_pointer_cast(F.data()), 
        dt
    );
    cudaDeviceSynchronize();
}


