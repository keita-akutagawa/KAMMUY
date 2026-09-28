#include "field_solver.hpp"


FieldSolver::FieldSolver(
    PICConstParameter& pICConstParameter, 
    const PICGridParameter& pICGridParameter
)
    : NX(pICGridParameter.NX), 
      NY(pICGridParameter.NY), 
      DX(pICGridParameter.DX), 
      DY(pICGridParameter.DY), 
      
      pICConstParameter(pICConstParameter), 
      pICGridParameter(pICGridParameter)
{
}


__global__ void timeEvolutionB_kernel(
    const PICUnsignedInt NX, PICUnsignedInt NY, 
    const PICFloat DX, PICFloat DY, 
    const ElectricField* E, 
    const PICFloat DT, 
    MagneticField* B
)
{
    PICUnsignedInt i = blockIdx.x * blockDim.x + threadIdx.x;
    PICUnsignedInt j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < NX - 1 && j < NY - 1) {
        PICUnsignedLongLong index = getIndex<PICUnsignedLongLong>(i, j, NX, NY); 

        B[index].bX += -(E[index + 1].eZ - E[index].eZ) / DY * DT;
        B[index].bY += (E[index + NY].eZ - E[index].eZ) / DX * DT;
        B[index].bZ += (-(E[index + NY].eY - E[index].eY) / DX
                     + (E[index + 1].eX - E[index].eX) / DY) * DT;
    }
}

void FieldSolver::timeEvolutionB(
    const thrust::device_vector<ElectricField>& E, 
    const PICFloat DT, 
    thrust::device_vector<MagneticField>& B
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (NY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    timeEvolutionB_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        NX, NY, 
        DX, DY, 
        thrust::raw_pointer_cast(E.data()), 
        DT, 
        thrust::raw_pointer_cast(B.data())
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed at timeEvolutionB_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed at timeEvolutionB_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
}



__global__ void timeEvolutionE_kernel(
    const PICUnsignedInt NX, PICUnsignedInt NY, 
    const PICFloat DX, PICFloat DY, 
    const PICFloat EPSILON0, const PICFloat C, 
    const MagneticField* B, const CurrentField* current, 
    const PICFloat DT,  
    ElectricField* E
)
{
    PICUnsignedInt i = blockIdx.x * blockDim.x + threadIdx.x;
    PICUnsignedInt j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < NX && 0 < j && j < NY) {
        PICUnsignedLongLong index = getIndex<PICUnsignedLongLong>(i, j, NX, NY); 

        E[index].eX += (-current[index].jX / EPSILON0
                     + C * C * (B[index].bZ - B[index - 1].bZ) / DY) * DT;
        E[index].eY += (-current[index].jY / EPSILON0 
                     - C * C * (B[index].bZ - B[index - NY].bZ) / DX) * DT;
        E[index].eZ += (-current[index].jZ / EPSILON0 
                     + C * C * ((B[index].bY - B[index - NY].bY) / DX
                     - (B[index].bX - B[index - 1].bX) / DY)) * DT;
    }
}

void FieldSolver::timeEvolutionE(
    const thrust::device_vector<MagneticField>& B, 
    const thrust::device_vector<CurrentField>& current, 
    const PICFloat DT,  
    thrust::device_vector<ElectricField>& E
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (NY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    timeEvolutionE_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        NX, NY, 
        DX, DY, 
        pICConstParameter.EPSILON0, pICConstParameter.C, 
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(current.data()), 
        DT, 
        thrust::raw_pointer_cast(E.data())
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed at timeEvolutionE_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed at timeEvolutionE_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
}


