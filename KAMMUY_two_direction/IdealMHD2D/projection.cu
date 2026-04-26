#include "projection.hpp"


// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause]]]
//
//modified by Keita Akutagawa [2025.4.10]
//


Projection::Projection()
    : divB(IdealMHD2DConst::nx * IdealMHD2DConst::ny), 
      psi(IdealMHD2DConst::nx * IdealMHD2DConst::ny)
{
    AMGX_initialize();
    AMGX_config_create_from_file(&config, IdealMHD2DConst::jsonFilenameForSolver.c_str());
    AMGX_resources_create_simple(&resource, config);
    AMGX_solver_create(&solver, resource, AMGX_mode_dDDI, config);
    AMGX_matrix_create(&A, resource, AMGX_mode_dDDI);
    AMGX_vector_create(&amgx_sol, resource, AMGX_mode_dDDI);
    AMGX_vector_create(&amgx_rhs, resource, AMGX_mode_dDDI);
    AMGX_read_system(A, amgx_sol, amgx_rhs, IdealMHD2DConst::MTXfilename.c_str());
    AMGX_solver_setup(solver, A);
}


Projection::~Projection()
{
    AMGX_solver_destroy(solver);
    AMGX_vector_destroy(amgx_rhs);
    AMGX_vector_destroy(amgx_sol);
    AMGX_matrix_destroy(A);
    AMGX_resources_destroy(resource);
    AMGX_config_destroy(config);
    AMGX_finalize();
}


__global__ void calculateDivB_kernel(
    double* divB, 
    const ConservationParameter* U
)   
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < IdealMHD2DConst::device_nx - 1 && 0 < j && j < IdealMHD2DConst::device_ny - 1) {
        unsigned long long index  = j + i * IdealMHD2DConst::device_ny;
        
        divB[index] = (U[index + IdealMHD2DConst::device_ny].bX - U[index - IdealMHD2DConst::device_ny].bX) / (2.0 * IdealMHD2DConst::device_dx)
                    + (U[index + 1].bY - U[index - 1].bY) / (2.0 * IdealMHD2DConst::device_dy);
    }
}


__global__ void correctDivB_kernel(
    ConservationParameter* U, 
    const double* psi
)   
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < IdealMHD2DConst::device_nx - 1 && 0 < j && j < IdealMHD2DConst::device_ny - 1) {
        unsigned long long index = j + i * IdealMHD2DConst::device_ny;
        
        U[index].bX += (psi[index + IdealMHD2DConst::device_ny] - psi[index - IdealMHD2DConst::device_ny]) / (2.0 * IdealMHD2DConst::device_dx);
        U[index].bY += (psi[index + 1] - psi[index - 1]) / (2.0 * IdealMHD2DConst::device_dy);
    }
}



void Projection::correctB(
    thrust::device_vector<ConservationParameter>& U
)
{

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((IdealMHD2DConst::nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (IdealMHD2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    calculateDivB_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(divB.data()), 
        thrust::raw_pointer_cast(U.data())
    );
    cudaDeviceSynchronize();

    thrust::fill(psi.begin(), psi.end(), 0.0); 
    AMGX_vector_upload(amgx_sol, IdealMHD2DConst::nx * IdealMHD2DConst::ny, 1, thrust::raw_pointer_cast(psi.data()));
    AMGX_vector_upload(amgx_rhs, IdealMHD2DConst::nx * IdealMHD2DConst::ny, 1, thrust::raw_pointer_cast(divB.data()));
    AMGX_solver_solve(solver, amgx_rhs, amgx_sol);
    AMGX_vector_download(amgx_sol, thrust::raw_pointer_cast(psi.data()));
    
    correctDivB_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(psi.data())
    );
    cudaDeviceSynchronize();
}

