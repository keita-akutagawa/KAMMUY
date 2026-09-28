#include "filter.hpp"
#include <thrust/fill.h>


Filter::Filter(
    PICConstParameter& pICConstParameter, 
    const PICGridParameter& pICGridParameter
)
    : NX(pICGridParameter.NX), 
      NY(pICGridParameter.NY), 
      DX(pICGridParameter.DX), 
      DY(pICGridParameter.DY), 

      pICConstParameter(pICConstParameter), 
      pICGridParameter(pICGridParameter), 

      rho(NX * NY), 
      F_E(NX * NY), 
      F_B(NX * NY), 

      momentCalculator(
        pICConstParameter, pICGridParameter
      )
{
}


__global__ void calculateRho_kernel(
    const ZerothMoment* zerothMomentIon, 
    const ZerothMoment* zerothMomentElectron, 
    const PICUnsignedInt NX, PICUnsignedInt NY, 
    const PICFloat Q_ION, PICFloat Q_ELECTRON, 
    RhoField* rho
)
{
    PICUnsignedInt i = blockIdx.x * blockDim.x + threadIdx.x;
    PICUnsignedInt j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < NX && j < NY) {
        PICUnsignedLongLong index = getIndex<PICUnsignedLongLong>(i, j, NX, NY); 

        rho[index].rho = Q_ION * zerothMomentIon[index].n
                       + Q_ELECTRON * zerothMomentElectron[index].n; 

    }
}


void Filter::calculateRho(
    const thrust::device_vector<Particle>& particlesIon, 
    const thrust::device_vector<Particle>& particlesElectron, 
    thrust::device_vector<ZerothMoment>& zerothMomentIon, 
    thrust::device_vector<ZerothMoment>& zerothMomentElectron
)
{
    momentCalculator.calculateZerothMoment(
        particlesIon, pICConstParameter.EXIST_NUM_ION, zerothMomentIon
    ); 
    momentCalculator.calculateZerothMoment(
        particlesElectron, pICConstParameter.EXIST_NUM_ELECTRON, zerothMomentElectron
    ); 


    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (NY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    calculateRho_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(zerothMomentIon.data()), 
        thrust::raw_pointer_cast(zerothMomentElectron.data()), 
        NX, NY, 
        pICConstParameter.Q_ION, pICConstParameter.Q_ELECTRON, 
        thrust::raw_pointer_cast(rho.data())
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed at calculateRho_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed at calculateRho_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
}


__global__ void calculateF_E_kernel(
    const PICUnsignedInt NX, PICUnsignedInt NY, 
    const PICFloat DX, PICFloat DY, 
    const PICFloat EPSILON0, 
    const ElectricField* E, const RhoField* rho, 
    FilterField* F_E
)
{
    PICUnsignedInt i = blockIdx.x * blockDim.x + threadIdx.x;
    PICUnsignedInt j = blockIdx.y * blockDim.y + threadIdx.y;

    if (1 <= i && i < NX && 1 <= j && j < NY) {
        PICUnsignedLongLong index = getIndex<PICUnsignedLongLong>(i, j, NX, NY); 

        F_E[index].F = ((E[index].eX - E[index - NY].eX) / DX 
                      + (E[index].eY - E[index - 1].eY) / DY)
                     - rho[index].rho / EPSILON0;
    }
}

__global__ void correctE_kernel(
    const PICUnsignedInt NX, PICUnsignedInt NY, 
    const PICFloat DX, PICFloat DY, 
    const PICFloat DCOEF_LM, 
    const FilterField* F_E, 
    const PICFloat DT,  
    ElectricField* E
)
{
    PICUnsignedInt i = blockIdx.x * blockDim.x + threadIdx.x;
    PICUnsignedInt j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < NX - 1 && j < NY - 1) {
        PICUnsignedLongLong index = getIndex<PICUnsignedLongLong>(i, j, NX, NY); 

        E[index].eX += DCOEF_LM * (F_E[index + NY].F - F_E[index].F) / DX * DT;
        E[index].eY += DCOEF_LM * (F_E[index + 1].F - F_E[index].F) / DY * DT;
    }
}


void Filter::langdonMarderTypeCorrectionE(
    const PICFloat DT, 
    thrust::device_vector<ElectricField>& E
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (NY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    calculateF_E_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        NX, NY, 
        DX, DY, 
        pICConstParameter.EPSILON0, 
        thrust::raw_pointer_cast(E.data()), 
        thrust::raw_pointer_cast(rho.data()), 
        thrust::raw_pointer_cast(F_E.data())
    );
    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess) {
        printf("Kernel launch failed at calculateF_E_kernel: %s\n", cudaGetErrorString(err1));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err1 = cudaDeviceSynchronize();
    if (err1 != cudaSuccess) {
        printf("Kernel execution failed at calculateF_E_kernel: %s\n", cudaGetErrorString(err1));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }

    correctE_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        NX, NY, 
        DX, DY, 
        pICConstParameter.DCOEF_LM,  
        thrust::raw_pointer_cast(F_E.data()), 
        DT, 
        thrust::raw_pointer_cast(E.data())
    );
    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess) {
        printf("Kernel launch failed at correctE_kernel: %s\n", cudaGetErrorString(err2));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err2 = cudaDeviceSynchronize();
    if (err2 != cudaSuccess) {
        printf("Kernel execution failed at correctE_kernel: %s\n", cudaGetErrorString(err2));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
}


__global__ void calculateF_B_kernel(
    const PICUnsignedInt NX, PICUnsignedInt NY, 
    const PICFloat DX, PICFloat DY, 
    const MagneticField* B, 
    FilterField* F_B
)
{
    PICUnsignedInt i = blockIdx.x * blockDim.x + threadIdx.x;
    PICUnsignedInt j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < NX - 1 && j < NY - 1) {
        PICUnsignedLongLong index = getIndex<PICUnsignedLongLong>(i, j, NX, NY); 

        F_B[index].F = (B[index + NY].bX - B[index].bX) / DX 
                     + (B[index + 1].bY - B[index].bY) / DY;
    }
}

__global__ void correctB_kernel(
    const PICUnsignedInt NX, PICUnsignedInt NY, 
    const PICFloat DX, PICFloat DY, 
    const PICFloat DCOEF_LM, 
    const FilterField* F_B, 
    const PICFloat DT,  
    MagneticField* B
)
{
    PICUnsignedInt i = blockIdx.x * blockDim.x + threadIdx.x;
    PICUnsignedInt j = blockIdx.y * blockDim.y + threadIdx.y;

    if (1 <= i && i < NX && 1 <= j && j < NY) {
        PICUnsignedLongLong index = getIndex<PICUnsignedLongLong>(i, j, NX, NY); 

        B[index].bX += DCOEF_LM * (F_B[index].F - F_B[index - NY].F) / DX * DT;
        B[index].bY += DCOEF_LM * (F_B[index].F - F_B[index - 1].F) / DY * DT;
    }
}


void Filter::langdonMarderTypeCorrectionB(
    const PICFloat DT, 
    thrust::device_vector<MagneticField>& B
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (NY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    calculateF_B_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        NX, NY, 
        DX, DY, 
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(F_B.data())
    );
    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess) {
        printf("Kernel launch failed at calculateF_B_kernel: %s\n", cudaGetErrorString(err1));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err1 = cudaDeviceSynchronize();
    if (err1 != cudaSuccess) {
        printf("Kernel execution failed at calculateF_B_kernel: %s\n", cudaGetErrorString(err1));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }

    correctB_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        NX, NY, 
        DX, DY, 
        pICConstParameter.DCOEF_LM,  
        thrust::raw_pointer_cast(F_B.data()), 
        DT, 
        thrust::raw_pointer_cast(B.data())
    );
    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess) {
        printf("Kernel launch failed at correctB_kernel: %s\n", cudaGetErrorString(err2));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err2 = cudaDeviceSynchronize();
    if (err2 != cudaSuccess) {
        printf("Kernel execution failed at correctB_kernel: %s\n", cudaGetErrorString(err2));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
}
