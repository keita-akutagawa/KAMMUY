#include "filter.hpp"
#include <thrust/fill.h>


using namespace PIC2DConst;

Filter::Filter()
    : rho(nx_PIC * ny_PIC), 
      F(nx_PIC * ny_PIC)
{
}


__global__ void calculateF_kernel(
    FilterField* F, ElectricField* E, RhoField* rho
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < device_nx_PIC - 1) && (0 < j) && (j < device_ny_PIC - 1)) {
        F[j + device_ny_PIC * i].F = ((E[j + device_ny_PIC * i].eX - E[j + device_ny_PIC * (i - 1)].eX) / device_dx_PIC 
                               + (E[j + device_ny_PIC * i].eY - E[j - 1 + device_ny_PIC * i].eY) / device_dy_PIC)
                               - rho[j + device_ny_PIC * i].rho / device_epsilon0_PIC;
    }
}

__global__ void correctE_kernel(
    ElectricField* E, FilterField* F, double dt
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < device_nx_PIC - 1) && (0 < j) && (j < device_ny_PIC - 1)) {
        E[j + device_ny_PIC * i].eX += device_dOfLangdonMarderTypeCorrection_PIC
                                 * (F[j + device_ny_PIC * (i + 1)].F - F[j + device_ny_PIC * i].F) / device_dx_PIC * dt;
        E[j + device_ny_PIC * i].eY += device_dOfLangdonMarderTypeCorrection_PIC
                                 * (F[j + 1 + device_ny_PIC * i].F - F[j + device_ny_PIC * i].F) / device_dy_PIC * dt;
    }
}


void Filter::langdonMarderTypeCorrection(
    thrust::device_vector<ElectricField>& E, 
    const thrust::device_vector<Particle>& particlesIon, 
    const thrust::device_vector<Particle>& particlesElectron, 
    const double dt
)
{
    calculateRho(particlesIon, particlesElectron);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny_PIC + threadsPerBlock.y - 1) / threadsPerBlock.y);

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

    calculateRhoOfOneSpecies(particlesIon, qIon_PIC, existNumIon_PIC);
    calculateRhoOfOneSpecies(particlesElectron, qElectron_PIC, existNumElectron_PIC);
}


struct CalculateRhoOfOneSpeciesFunctor {
    RhoField* rho;
    const Particle* particlesSpecies;
    const double q;

    __device__
    void operator()(const unsigned long long& i) const {
        double cx1, cx2; 
        int xIndex1, xIndex2;
        double xOverDx;
        double cy1, cy2; 
        int yIndex1, yIndex2;
        double yOverDy;

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

        atomicAdd(&(rho[yIndex1 + device_ny_PIC * xIndex1].rho), q * cx2 * cy2);
        atomicAdd(&(rho[yIndex2 + device_ny_PIC * xIndex1].rho), q * cx2 * cy1);
        atomicAdd(&(rho[yIndex1 + device_ny_PIC * xIndex2].rho), q * cx1 * cy2);
        atomicAdd(&(rho[yIndex2 + device_ny_PIC * xIndex2].rho), q * cx1 * cy1);
    }
};


void Filter::calculateRhoOfOneSpecies(
    const thrust::device_vector<Particle>& particlesSpecies, 
    double q, int existNumSpecies
)
{
    CalculateRhoOfOneSpeciesFunctor calculateRhoOfOneSpeciesFunctor{
        thrust::raw_pointer_cast(rho.data()), 
        thrust::raw_pointer_cast(particlesSpecies.data()), 
        q
    };

    thrust::for_each(
        thrust::counting_iterator<unsigned long long>(0), 
        thrust::counting_iterator<unsigned long long>(existNumSpecies), 
        calculateRhoOfOneSpeciesFunctor
    );

    cudaDeviceSynchronize();
}



