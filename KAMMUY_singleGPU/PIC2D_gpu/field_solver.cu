#include "field_solver.hpp"


using namespace PIC2DConst;

__global__ void timeEvolutionB_kernel(
    MagneticField* B, const ElectricField* E, const double dt
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < device_nx_PIC - 1 && j < device_ny_PIC - 1) {
        B[j + device_ny_PIC * i].bX += -(E[j + 1 + device_ny_PIC * i].eZ - E[j + device_ny_PIC * i].eZ) / device_dy_PIC * dt;
        B[j + device_ny_PIC * i].bY += (E[j + device_ny_PIC * (i + 1)].eZ - E[j + device_ny_PIC * i].eZ) / device_dx_PIC * dt;
        B[j + device_ny_PIC * i].bZ += (-(E[j + device_ny_PIC * (i + 1)].eY - E[j + device_ny_PIC * i].eY) / device_dx_PIC
                                 + (E[j + 1 + device_ny_PIC * i].eX - E[j + device_ny_PIC * i].eX) / device_dy_PIC) * dt;
    }

    if (i == device_nx_PIC - 1 && j < device_ny_PIC - 1) {
        B[j + device_ny_PIC * i].bX += -(E[j + 1 + device_ny_PIC * i].eZ - E[j + device_ny_PIC * i].eZ) / device_dy_PIC * dt;
        B[j + device_ny_PIC * i].bY += (E[j + device_ny_PIC * 0].eZ - E[j + device_ny_PIC * i].eZ) / device_dx_PIC * dt;
        B[j + device_ny_PIC * i].bZ += (-(E[j + device_ny_PIC * 0].eY - E[j + device_ny_PIC * i].eY) / device_dx_PIC
                                 + (E[j + 1 + device_ny_PIC * i].eX - E[j + device_ny_PIC * i].eX) / device_dy_PIC) * dt;
    }

    if (i < device_nx_PIC - 1 && j == device_ny_PIC - 1) {
        B[j + device_ny_PIC * i].bX += -(E[0 + device_ny_PIC * i].eZ - E[j + device_ny_PIC * i].eZ) / device_dy_PIC * dt;
        B[j + device_ny_PIC * i].bY += (E[j + device_ny_PIC * (i + 1)].eZ - E[j + device_ny_PIC * i].eZ) / device_dx_PIC * dt;
        B[j + device_ny_PIC * i].bZ += (-(E[j + device_ny_PIC * (i + 1)].eY - E[j + device_ny_PIC * i].eY) / device_dx_PIC
                                 + (E[0 + device_ny_PIC * i].eX - E[j + device_ny_PIC * i].eX) / device_dy_PIC) * dt;
    }

    if (i == device_nx_PIC - 1 && j == device_ny_PIC - 1) {
        B[j + device_ny_PIC * i].bX += -(E[0 + device_ny_PIC * i].eZ - E[j + device_ny_PIC * i].eZ) / device_dy_PIC * dt;
        B[j + device_ny_PIC * i].bY += (E[j + device_ny_PIC * 0].eZ - E[j + device_ny_PIC * i].eZ) / device_dx_PIC * dt;
        B[j + device_ny_PIC * i].bZ += (-(E[j + device_ny_PIC * 0].eY - E[j + device_ny_PIC * i].eY) / device_dx_PIC
                                 + (E[0 + device_ny_PIC * i].eX - E[j + device_ny_PIC * i].eX) / device_dy_PIC) * dt;
    }
}

void FieldSolver::timeEvolutionB(
    thrust::device_vector<MagneticField>& B, 
    const thrust::device_vector<ElectricField>& E, 
    const double dt
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny_PIC + threadsPerBlock.y - 1) / threadsPerBlock.y);

    timeEvolutionB_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(E.data()), 
        dt
    );

    cudaDeviceSynchronize();
}



__global__ void timeEvolutionE_kernel(
    ElectricField* E, const MagneticField* B, const CurrentField* current, const double dt
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < device_nx_PIC) && (0 < j) && (j < device_ny_PIC)) {
        E[j + device_ny_PIC * i].eX += (-current[j + device_ny_PIC * i].jX / device_epsilon0_PIC
                                 + device_c_PIC * device_c_PIC * (B[j + device_ny_PIC * i].bZ - B[j - 1 + device_ny_PIC * i].bZ) / device_dy_PIC) * dt;
        E[j + device_ny_PIC * i].eY += (-current[j + device_ny_PIC * i].jY / device_epsilon0_PIC 
                                 - device_c_PIC * device_c_PIC * (B[j + device_ny_PIC * i].bZ - B[j + device_ny_PIC * (i - 1)].bZ) / device_dx_PIC) * dt;
        E[j + device_ny_PIC * i].eZ += (-current[j + device_ny_PIC * i].jZ / device_epsilon0_PIC 
                                 + device_c_PIC * device_c_PIC * ((B[j + device_ny_PIC * i].bY - B[j + device_ny_PIC * (i - 1)].bY) / device_dx_PIC
                                 - (B[j + device_ny_PIC * i].bX - B[j - 1 + device_ny_PIC * i].bX) / device_dy_PIC)) * dt;
    }

    if ((i == 0) && (0 < j) && (j < device_ny_PIC)) {
        E[j + device_ny_PIC * i].eX += (-current[j + device_ny_PIC * i].jX / device_epsilon0_PIC
                                 + device_c_PIC * device_c_PIC * (B[j + device_ny_PIC * i].bZ - B[j - 1 + device_ny_PIC * i].bZ) / device_dy_PIC) * dt;
        E[j + device_ny_PIC * i].eY += (-current[j + device_ny_PIC * i].jY / device_epsilon0_PIC 
                                 - device_c_PIC * device_c_PIC * (B[j + device_ny_PIC * i].bZ - B[j + device_ny_PIC * (device_nx_PIC - 1)].bZ) / device_dx_PIC) * dt;
        E[j + device_ny_PIC * i].eZ += (-current[j + device_ny_PIC * i].jZ / device_epsilon0_PIC 
                                 + device_c_PIC * device_c_PIC * ((B[j + device_ny_PIC * i].bY - B[j + device_ny_PIC * (device_nx_PIC - 1)].bY) / device_dx_PIC
                                 - (B[j + device_ny_PIC * i].bX - B[j - 1 + device_ny_PIC * i].bX) / device_dy_PIC)) * dt;
    }

    if ((0 < i) && (i < device_nx_PIC) && (j == 0)) {
        E[j + device_ny_PIC * i].eX += (-current[j + device_ny_PIC * i].jX / device_epsilon0_PIC
                                 + device_c_PIC * device_c_PIC * (B[j + device_ny_PIC * i].bZ - B[device_ny_PIC - 1 + device_ny_PIC * i].bZ) / device_dy_PIC) * dt;
        E[j + device_ny_PIC * i].eY += (-current[j + device_ny_PIC * i].jY / device_epsilon0_PIC 
                                 - device_c_PIC * device_c_PIC * (B[j + device_ny_PIC * i].bZ - B[j + device_ny_PIC * (i - 1)].bZ) / device_dx_PIC) * dt;
        E[j + device_ny_PIC * i].eZ += (-current[j + device_ny_PIC * i].jZ / device_epsilon0_PIC 
                                 + device_c_PIC * device_c_PIC * ((B[j + device_ny_PIC * i].bY - B[j + device_ny_PIC * (i - 1)].bY) / device_dx_PIC
                                 - (B[j + device_ny_PIC * i].bX - B[device_ny_PIC - 1 + device_ny_PIC * i].bX) / device_dy_PIC)) * dt;
    }

    if (i == 0 && j == 0) {
        E[j + device_ny_PIC * i].eX += (-current[j + device_ny_PIC * i].jX / device_epsilon0_PIC
                                 + device_c_PIC * device_c_PIC * (B[j + device_ny_PIC * i].bZ - B[device_ny_PIC - 1 + device_ny_PIC * i].bZ) / device_dy_PIC) * dt;
        E[j + device_ny_PIC * i].eY += (-current[j + device_ny_PIC * i].jY / device_epsilon0_PIC 
                                 - device_c_PIC * device_c_PIC * (B[j + device_ny_PIC * i].bZ - B[j + device_ny_PIC * (device_nx_PIC - 1)].bZ) / device_dx_PIC) * dt;
        E[j + device_ny_PIC * i].eZ += (-current[j + device_ny_PIC * i].jZ / device_epsilon0_PIC 
                                 + device_c_PIC * device_c_PIC * ((B[j + device_ny_PIC * i].bY - B[j + device_ny_PIC * (device_nx_PIC - 1)].bY) / device_dx_PIC
                                 - (B[j + device_ny_PIC * i].bX - B[device_ny_PIC - 1 + device_ny_PIC * i].bX) / device_dy_PIC)) * dt;
    }
}

void FieldSolver::timeEvolutionE(
    thrust::device_vector<ElectricField>& E, 
    const thrust::device_vector<MagneticField>& B, 
    const thrust::device_vector<CurrentField>& current, 
    const double dt
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny_PIC + threadsPerBlock.y - 1) / threadsPerBlock.y);

    timeEvolutionE_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(E.data()), 
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(current.data()), 
        dt
    );

    cudaDeviceSynchronize();
}


