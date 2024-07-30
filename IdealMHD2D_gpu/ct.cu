#include "ct.hpp"


using namespace IdealMHD2DConst;

CT::CT()
    : oldFluxF(nx_MHD * ny_MHD), 
      oldFluxG(nx_MHD * ny_MHD), 
      eZVector(nx_MHD * ny_MHD)
{
}


void CT::setOldFlux2D(
    const thrust::device_vector<Flux>& fluxF, 
    const thrust::device_vector<Flux>& fluxG
)
{
    thrust::copy(fluxF.begin(), fluxF.end(), oldFluxF.begin());
    thrust::copy(fluxG.begin(), fluxG.end(), oldFluxG.begin());
}


__global__ void getEZVector_kernel(
    const Flux* fluxF, 
    const Flux* fluxG, 
    double* eZVector
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < device_nx_MHD - 1 && j < device_ny_MHD - 1) {
        double eZF1, eZF2, eZG1, eZG2, eZ;

        eZG1 = fluxG[j + i * device_ny_MHD].f4;
        eZG2 = fluxG[j + (i + 1) * device_ny_MHD].f4;
        eZF1 = -1.0 * fluxF[j + i * device_ny_MHD].f5;
        eZF2 = -1.0 * fluxF[j + 1 + i * device_ny_MHD].f5;
        eZ = 0.25 * (eZG1 + eZG2 + eZF1 + eZF2);
        eZVector[j + i * device_ny_MHD] = eZ;
    }
}


__global__ void CT_kernel(
    const double* bXOld, const double* bYOld, 
    const double* eZVector, 
    ConservationParameter* U
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < device_nx_MHD) && (0 < j) && (j < device_ny_MHD)) {
        U[j + i * device_ny_MHD].bX = bXOld[j + i * device_ny_MHD]
                                - device_dt_MHD / device_dy_MHD * (eZVector[j + i * device_ny_MHD] - eZVector[j - 1 + i * device_ny_MHD]);
        U[j + i * device_ny_MHD].bY = bYOld[j + i * device_ny_MHD]
                                + device_dt_MHD / device_dx_MHD * (eZVector[j + i * device_ny_MHD] - eZVector[j + (i - 1) * device_ny_MHD]);
    }
}


void CT::divBClean(
    const thrust::device_vector<double>& bXOld, 
    const thrust::device_vector<double>& bYOld, 
    thrust::device_vector<ConservationParameter>& U
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx_MHD + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny_MHD + threadsPerBlock.y - 1) / threadsPerBlock.y);

    getEZVector_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(oldFluxF.data()), 
        thrust::raw_pointer_cast(oldFluxG.data()), 
        thrust::raw_pointer_cast(eZVector.data())
    );

    CT_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(bXOld.data()),
        thrust::raw_pointer_cast(bYOld.data()),
        thrust::raw_pointer_cast(eZVector.data()),
        thrust::raw_pointer_cast(U.data())
    );

}

