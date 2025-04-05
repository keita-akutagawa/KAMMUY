#include "field_solver.hpp"


FieldSolver::FieldSolver(PIC2DMPI::MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo)
{
}


__global__ void timeEvolutionB_kernel(
    MagneticField* B, const ElectricField* E, const float dt, 
    int localSizeX
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localSizeX - 1 && j < PIC2DConst::device_ny - 1) {
        int index = j + i * PIC2DConst::device_ny;

        B[index].bX += -(E[index + 1].eZ - E[index].eZ) / PIC2DConst::device_dy * dt;
        B[index].bY += (E[index + PIC2DConst::device_ny].eZ - E[index].eZ) / PIC2DConst::device_dx * dt;
        B[index].bZ += (-(E[index + PIC2DConst::device_ny].eY - E[index].eY) / PIC2DConst::device_dx
                     + (E[index + 1].eX - E[index].eX) / PIC2DConst::device_dy) * dt;
    }
}

void FieldSolver::timeEvolutionB(
    thrust::device_vector<MagneticField>& B, 
    const thrust::device_vector<ElectricField>& E, 
    const float dt
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (PIC2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    timeEvolutionB_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(E.data()), 
        dt, 
        mPIInfo.localSizeX
    );
    cudaDeviceSynchronize();
}



__global__ void timeEvolutionE_kernel(
    ElectricField* E, const MagneticField* B, const CurrentField* current, 
    const float dt, 
    int localSizeX
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < localSizeX) && (0 < j) && (j < PIC2DConst::device_ny)) {
        int index = j + i * PIC2DConst::device_ny;

        E[index].eX += (-current[index].jX / PIC2DConst::device_epsilon0
                     + PIC2DConst::device_c * PIC2DConst::device_c * (B[index].bZ - B[index - 1].bZ) / PIC2DConst::device_dy) * dt;
        E[index].eY += (-current[index].jY / PIC2DConst::device_epsilon0 
                     - PIC2DConst::device_c * PIC2DConst::device_c * (B[index].bZ - B[index - PIC2DConst::device_ny].bZ) / PIC2DConst::device_dx) * dt;
        E[index].eZ += (-current[index].jZ / PIC2DConst::device_epsilon0 
                     + PIC2DConst::device_c * PIC2DConst::device_c * ((B[index].bY - B[index - PIC2DConst::device_ny].bY) / PIC2DConst::device_dx
                     - (B[index].bX - B[index - 1].bX) / PIC2DConst::device_dy)) * dt;
    }
}

void FieldSolver::timeEvolutionE(
    thrust::device_vector<ElectricField>& E, 
    const thrust::device_vector<MagneticField>& B, 
    const thrust::device_vector<CurrentField>& current, 
    const float dt
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (PIC2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    timeEvolutionE_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(E.data()), 
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(current.data()), 
        dt, 
        mPIInfo.localSizeX
    );
    cudaDeviceSynchronize();
}


