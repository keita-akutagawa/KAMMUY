#include "projection.hpp"


// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause]]]
//
//modified by Keita Akutagawa [2025.4.10]
//


Projection::Projection(IdealMHD2DMPI::MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo),
      
      //今のところは各プロセスが全領域を持っていて、ランク0だけが解くことにする
      divB(IdealMHD2DConst::nx * IdealMHD2DConst::ny), 
      sum_divB(IdealMHD2DConst::nx * IdealMHD2DConst::ny), 
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
    const ConservationParameter* U, 
    int localNx, int buffer, 
    int localSizeX, 
    int localGridX
)   
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localNx && j < IdealMHD2DConst::device_ny) {
        int indexLeft  = j + (i - 1 + buffer) * IdealMHD2DConst::device_ny;
        int indexRight = j + (i + 1 + buffer) * IdealMHD2DConst::device_ny;
        int indexDown  = (j - 1 + IdealMHD2DConst::device_ny) % IdealMHD2DConst::device_ny
                       + (i + buffer) * IdealMHD2DConst::device_ny; 
        int indexUp    = (j + 1 + IdealMHD2DConst::device_ny) % IdealMHD2DConst::device_ny
                       + (i + buffer) * IdealMHD2DConst::device_ny; 
        
        int indexForDivB = j + (i + localNx * localGridX) * IdealMHD2DConst::device_ny; 
        
        divB[indexForDivB] = (U[indexRight].bX - U[indexLeft].bX) / (2.0 * IdealMHD2DConst::device_dx)
                           + (U[indexUp].bY - U[indexDown].bY) / (2.0 * IdealMHD2DConst::device_dy);
    }
}


__global__ void correctDivB_kernel(
    ConservationParameter* U, 
    const double* psi, 
    int localNx, int buffer, 
    int localSizeX, 
    int localGridX
)   
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localNx && j < IdealMHD2DConst::device_ny) {
        int index = j + (i + buffer) * IdealMHD2DConst::device_ny;
        int indexForPsiLeft  = j + ((i - 1 + localNx * localGridX + IdealMHD2DConst::device_nx) % IdealMHD2DConst::device_nx) * IdealMHD2DConst::device_ny; 
        int indexForPsiRight = j + ((i + 1 + localNx * localGridX + IdealMHD2DConst::device_nx) % IdealMHD2DConst::device_nx) * IdealMHD2DConst::device_ny; 
        int indexForPsiDown  = (j - 1 + IdealMHD2DConst::device_ny) % IdealMHD2DConst::device_ny
                             + (i + localNx * localGridX) * IdealMHD2DConst::device_ny; 
        int indexForPsiUp    = (j + 1 + IdealMHD2DConst::device_ny) % IdealMHD2DConst::device_ny
                             + (i + localNx * localGridX) * IdealMHD2DConst::device_ny; 
        
        U[index].bX += (psi[indexForPsiRight] - psi[indexForPsiLeft]) / (2.0 * IdealMHD2DConst::device_dx);
        U[index].bY += (psi[indexForPsiUp] - psi[indexForPsiDown]) / (2.0 * IdealMHD2DConst::device_dy);
    }
}



void Projection::correctB(
    thrust::device_vector<ConservationParameter>& U
)
{

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localNx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (IdealMHD2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    calculateDivB_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(divB.data()), 
        thrust::raw_pointer_cast(U.data()),
        mPIInfo.localNx, mPIInfo.buffer, 
        mPIInfo.localSizeX, 
        mPIInfo.localGridX
    );
    cudaDeviceSynchronize();

    thrust::fill(sum_divB.begin(), sum_divB.end(), 0.0);
    MPI_Reduce(thrust::raw_pointer_cast(divB.data()), thrust::raw_pointer_cast(sum_divB.data()), IdealMHD2DConst::nx * IdealMHD2DConst::ny, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (mPIInfo.rank == 0) {
        AMGX_vector_upload(amgx_rhs, IdealMHD2DConst::nx * IdealMHD2DConst::ny, 1, thrust::raw_pointer_cast(sum_divB.data()));
        AMGX_solver_solve(solver, amgx_rhs, amgx_sol);
        AMGX_vector_download(amgx_sol, thrust::raw_pointer_cast(psi.data()));
    }
    MPI_Bcast(thrust::raw_pointer_cast(psi.data()), IdealMHD2DConst::nx * IdealMHD2DConst::ny, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    correctDivB_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(psi.data()),
        mPIInfo.localNx, mPIInfo.buffer, 
        mPIInfo.localSizeX,
        mPIInfo.localGridX
    );
    cudaDeviceSynchronize();
}

