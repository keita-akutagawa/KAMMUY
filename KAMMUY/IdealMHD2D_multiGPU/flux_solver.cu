#include "flux_solver.hpp"


FluxSolver::FluxSolver(IdealMHD2DMPI::MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo), 
      hLLD(mPIInfo)
{
}


thrust::device_vector<Flux> FluxSolver::getFluxF(
    const thrust::device_vector<ConservationParameter>& U
)
{
    hLLD.calculateFluxF(U);
    flux = hLLD.getFlux();

    addResistiveTermToFluxF(U);

    return flux;
}


thrust::device_vector<Flux> FluxSolver::getFluxG(
    const thrust::device_vector<ConservationParameter>& U
)
{
    hLLD.calculateFluxG(U);
    flux = hLLD.getFlux();

    addResistiveTermToFluxG(U);

    return flux;
}


__global__ void addResistiveTermToFluxF_kernel(
    const ConservationParameter* U, Flux* flux, 
    int localSizeX
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < localSizeX - 2) && (0 < j) && (j < IdealMHD2DConst::device_ny - 1)) {
        unsigned long long index = j + i * IdealMHD2DConst::device_ny;

        double jY, jZ;
        double eta, etaJY, etaJZ, etaJYBZ, etaJZBY;

        jY = -(U[index + IdealMHD2DConst::device_ny].bZ - U[index].bZ) / IdealMHD2DConst::device_dx;
        jZ = 0.5 * (
             (U[index + IdealMHD2DConst::device_ny].bY - U[index - IdealMHD2DConst::device_ny].bY) / (2.0 * IdealMHD2DConst::device_dx)
           - (U[index + 1].bX - U[index - 1].bX) / (2.0 * IdealMHD2DConst::device_dy)
           + (U[index + 2 * IdealMHD2DConst::device_ny].bY - U[index].bY) / (2.0 * IdealMHD2DConst::device_dx)
           - (U[index + IdealMHD2DConst::device_ny + 1].bX - U[index + IdealMHD2DConst::device_ny - 1].bX) / (2.0 * IdealMHD2DConst::device_dy)
        );
           
        eta = IdealMHD2DConst::device_eta;
        etaJY = eta * jY; 
        etaJZ = eta * jZ;
        etaJYBZ = etaJY * 0.5 * (U[index].bZ + U[index + IdealMHD2DConst::device_ny].bZ);
        etaJZBY = etaJZ * 0.5 * (U[index].bY + U[index + IdealMHD2DConst::device_ny].bY);
  
        flux[index].f5 -= etaJZ;
        flux[index].f6 += etaJY;
        flux[index].f7 += etaJYBZ - etaJZBY;
    }
}

void FluxSolver::addResistiveTermToFluxF(
    const thrust::device_vector<ConservationParameter>& U
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (IdealMHD2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    addResistiveTermToFluxF_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(flux.data()), 
        mPIInfo.localSizeX
    );
    cudaDeviceSynchronize();
}


__global__ void addResistiveTermToFluxG_kernel(
    const ConservationParameter* U, Flux* flux, 
    int localSizeX
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < localSizeX - 1) && (0 < j) && (j < IdealMHD2DConst::device_ny - 2)) {
        unsigned long long index = j + i * IdealMHD2DConst::device_ny;

        double jX, jZ;
        double eta, etaJX, etaJZ, etaJZBX, etaJXBZ;

        jX = (U[index + 1].bZ - U[index].bZ) / IdealMHD2DConst::device_dy;
        jZ = 0.5 * (
             (U[index + IdealMHD2DConst::device_ny].bY - U[index - IdealMHD2DConst::device_ny].bY) / (2.0 * IdealMHD2DConst::device_dx)
           - (U[index + 1].bX - U[index - 1].bX) / (2.0 * IdealMHD2DConst::device_dy)
           + (U[index + IdealMHD2DConst::device_ny + 1].bY - U[index - IdealMHD2DConst::device_ny + 1].bY) / (2.0 * IdealMHD2DConst::device_dx)
           - (U[index + 2].bX - U[index].bX) / (2.0 * IdealMHD2DConst::device_dy)
        );
        
        eta = IdealMHD2DConst::device_eta;
        etaJX = eta * jX;
        etaJZ = eta * jZ;
        etaJXBZ = etaJX * 0.5 * (U[index].bZ + U[index + 1].bZ);
        etaJZBX = etaJZ * 0.5 * (U[index].bX + U[index + 1].bX);
  
        flux[index].f4 += etaJZ;
        flux[index].f6 -= etaJX;
        flux[index].f7 += etaJZBX - etaJXBZ;
    }
}

void FluxSolver::addResistiveTermToFluxG(
    const thrust::device_vector<ConservationParameter>& U
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (IdealMHD2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    addResistiveTermToFluxG_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(flux.data()), 
        mPIInfo.localSizeX
    );
    cudaDeviceSynchronize();
}

