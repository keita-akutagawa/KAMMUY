#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <thrust/extrema.h>
#include "const.hpp"
#include "idealMHD2D.hpp"


IdealMHD2D::IdealMHD2D(IdealMHD2DMPI::MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo), 

      fluxSolver(mPIInfo), 

      fluxF    (mPIInfo.localSizeX * IdealMHD2DConst::ny),
      fluxG    (mPIInfo.localSizeX * IdealMHD2DConst::ny),
      U        (mPIInfo.localSizeX * IdealMHD2DConst::ny),
      UBar     (mPIInfo.localSizeX * IdealMHD2DConst::ny), 
      UPast    (mPIInfo.localSizeX * IdealMHD2DConst::ny), 
      tmpVector(mPIInfo.localSizeX * IdealMHD2DConst::ny),
      host_U   (mPIInfo.localSizeX * IdealMHD2DConst::ny), 

      dtVector(mPIInfo.localNx * IdealMHD2DConst::ny), 

      boundaryMHD(mPIInfo), 
      projection(mPIInfo)
{

    cudaMalloc(&device_mPIInfo, sizeof(IdealMHD2DMPI::MPIInfo));
    cudaMemcpy(device_mPIInfo, &mPIInfo, sizeof(IdealMHD2DMPI::MPIInfo), cudaMemcpyHostToDevice);
    
}


void IdealMHD2D::setPastU()
{
    thrust::copy(U.begin(), U.end(), UPast.begin());
}


__global__ void oneStepFirst_kernel(
    const ConservationParameter* U, 
    const Flux* fluxF, const Flux* fluxG, 
    ConservationParameter* UBar, 
    const double dt, 
    int localSizeX
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < localSizeX - 1) && (0 < j) && (j < IdealMHD2DConst::device_ny - 1)) {
        unsigned long long index = j + i * IdealMHD2DConst::device_ny;

        double rho, rhoU, rhoV, rhoW, bX, bY, bZ, e; 

        rho  = U[index].rho  
             - dt / IdealMHD2DConst::device_dx * (fluxF[index].f0 - fluxF[index - IdealMHD2DConst::device_ny].f0)
             - dt / IdealMHD2DConst::device_dy * (fluxG[index].f0 - fluxG[index - 1].f0);
        rhoU = U[index].rhoU 
             - dt / IdealMHD2DConst::device_dx * (fluxF[index].f1 - fluxF[index - IdealMHD2DConst::device_ny].f1)
             - dt / IdealMHD2DConst::device_dy * (fluxG[index].f1 - fluxG[index - 1].f1);
        rhoV = U[index].rhoV
             - dt / IdealMHD2DConst::device_dx * (fluxF[index].f2 - fluxF[index - IdealMHD2DConst::device_ny].f2)
             - dt / IdealMHD2DConst::device_dy * (fluxG[index].f2 - fluxG[index - 1].f2);
        rhoW = U[index].rhoW
             - dt / IdealMHD2DConst::device_dx * (fluxF[index].f3 - fluxF[index - IdealMHD2DConst::device_ny].f3)
             - dt / IdealMHD2DConst::device_dy * (fluxG[index].f3 - fluxG[index - 1].f3);
        bX   = U[index].bX 
             - dt / IdealMHD2DConst::device_dx * (fluxF[index].f4 - fluxF[index - IdealMHD2DConst::device_ny].f4)
             - dt / IdealMHD2DConst::device_dy * (fluxG[index].f4 - fluxG[index - 1].f4);
        bY   = U[index].bY 
             - dt / IdealMHD2DConst::device_dx * (fluxF[index].f5 - fluxF[index - IdealMHD2DConst::device_ny].f5)
             - dt / IdealMHD2DConst::device_dy * (fluxG[index].f5 - fluxG[index - 1].f5);
        bZ   = U[index].bZ 
             - dt / IdealMHD2DConst::device_dx * (fluxF[index].f6 - fluxF[index - IdealMHD2DConst::device_ny].f6)
             - dt / IdealMHD2DConst::device_dy * (fluxG[index].f6 - fluxG[index - 1].f6);
        e    = U[index].e 
             - dt / IdealMHD2DConst::device_dx * (fluxF[index].f7 - fluxF[index - IdealMHD2DConst::device_ny].f7)
             - dt / IdealMHD2DConst::device_dy * (fluxG[index].f7 - fluxG[index - 1].f7);
        
        if (!(isnan(rho) || isnan(rhoU) || isnan(rhoV) || isnan(rhoW) || isnan(bX) || isnan(bY) || isnan(bZ) || isnan(e))) {
            UBar[index].rho  = rho; 
            UBar[index].rhoU = rhoU; 
            UBar[index].rhoV = rhoV; 
            UBar[index].rhoW = rhoW; 
            UBar[index].bX   = bX; 
            UBar[index].bY   = bY; 
            UBar[index].bZ   = bZ; 
            UBar[index].e    = e; 
        }
    }
}


__global__ void oneStepSecond_kernel(
    const ConservationParameter* UBar, 
    const Flux* fluxF, const Flux* fluxG, 
    ConservationParameter* U, 
    int localSizeX
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < localSizeX - 1) && (0 < j) && (j < IdealMHD2DConst::device_ny - 1)) {
        unsigned long long index = j + i * IdealMHD2DConst::device_ny;
        double rho, rhoU, rhoV, rhoW, bX, bY, bZ, e;

        rho  = 0.5 * (U[index].rho + UBar[index].rho
             - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dx * (fluxF[index].f0 - fluxF[index - IdealMHD2DConst::device_ny].f0)
             - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dy * (fluxG[index].f0 - fluxG[index - 1].f0));
        rhoU = 0.5 * (U[index].rhoU + UBar[index].rhoU
             - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dx * (fluxF[index].f1 - fluxF[index - IdealMHD2DConst::device_ny].f1)
             - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dy * (fluxG[index].f1 - fluxG[index - 1].f1));
        rhoV = 0.5 * (U[index].rhoV + UBar[index].rhoV
             - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dx * (fluxF[index].f2 - fluxF[index - IdealMHD2DConst::device_ny].f2)
             - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dy * (fluxG[index].f2 - fluxG[index - 1].f2));
        rhoW = 0.5 * (U[index].rhoW + UBar[index].rhoW
             - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dx * (fluxF[index].f3 - fluxF[index - IdealMHD2DConst::device_ny].f3)
             - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dy * (fluxG[index].f3 - fluxG[index - 1].f3));
        bX   = 0.5 * (U[index].bX + UBar[index].bX
             - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dx * (fluxF[index].f4 - fluxF[index - IdealMHD2DConst::device_ny].f4)
             - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dy * (fluxG[index].f4 - fluxG[index - 1].f4));
        bY   = 0.5 * (U[index].bY + UBar[index].bY
             - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dx * (fluxF[index].f5 - fluxF[index - IdealMHD2DConst::device_ny].f5)
             - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dy * (fluxG[index].f5 - fluxG[index - 1].f5));
        bZ   = 0.5 * (U[index].bZ + UBar[index].bZ
             - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dx * (fluxF[index].f6 - fluxF[index - IdealMHD2DConst::device_ny].f6)
             - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dy * (fluxG[index].f6 - fluxG[index - 1].f6));
        e    = 0.5 * (U[index].e + UBar[index].e
             - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dx * (fluxF[index].f7 - fluxF[index - IdealMHD2DConst::device_ny].f7)
             - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dy * (fluxG[index].f7 - fluxG[index - 1].f7));

        if (!(isnan(rho) || isnan(rhoU) || isnan(rhoV) || isnan(rhoW) || isnan(bX) || isnan(bY) || isnan(bZ) || isnan(e))) {
            U[index].rho  = rho; 
            U[index].rhoU = rhoU; 
            U[index].rhoV = rhoV; 
            U[index].rhoW = rhoW; 
            U[index].bX   = bX; 
            U[index].bY   = bY; 
            U[index].bZ   = bZ; 
            U[index].e    = e; 
        }
    }
}


void IdealMHD2D::oneStepRK2_periodicXY_predictor()
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (IdealMHD2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    thrust::copy(U.begin(), U.end(), UBar.begin());

    fluxF = fluxSolver.getFluxF(U);
    fluxG = fluxSolver.getFluxG(U);

    oneStepFirst_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(fluxF.data()), 
        thrust::raw_pointer_cast(fluxG.data()), 
        thrust::raw_pointer_cast(UBar.data()), 
        IdealMHD2DConst::dt, 
        mPIInfo.localSizeX
    );
    cudaDeviceSynchronize();

    boundaryMHD.periodicBoundaryX2nd_U(UBar);
    boundaryMHD.periodicBoundaryY2nd_U(UBar);
    MPI_Barrier(MPI_COMM_WORLD);

    checkAndResetExtremeValues(UBar);

    fluxF = fluxSolver.getFluxF(UBar);
    fluxG = fluxSolver.getFluxG(UBar);

    oneStepSecond_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(UBar.data()), 
        thrust::raw_pointer_cast(fluxF.data()), 
        thrust::raw_pointer_cast(fluxG.data()), 
        thrust::raw_pointer_cast(U.data()), 
        mPIInfo.localSizeX
    );
    cudaDeviceSynchronize();

    boundaryMHD.periodicBoundaryX2nd_U(U);
    boundaryMHD.periodicBoundaryY2nd_U(U);
    MPI_Barrier(MPI_COMM_WORLD);

    checkAndResetExtremeValues(U);
}


void IdealMHD2D::oneStepRK2_periodicXY_corrector(
    thrust::device_vector<ConservationParameter>& UHalf
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (IdealMHD2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    checkAndResetExtremeValues(UHalf);

    fluxF = fluxSolver.getFluxF(UHalf);
    fluxG = fluxSolver.getFluxG(UHalf);

    oneStepFirst_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(UPast.data()), 
        thrust::raw_pointer_cast(fluxF.data()), 
        thrust::raw_pointer_cast(fluxG.data()), 
        thrust::raw_pointer_cast(U.data()), 
        IdealMHD2DConst::dt, 
        mPIInfo.localSizeX
    );
    cudaDeviceSynchronize();

    boundaryMHD.periodicBoundaryX2nd_U(U);
    boundaryMHD.periodicBoundaryY2nd_U(U);
    MPI_Barrier(MPI_COMM_WORLD);

    checkAndResetExtremeValues(U);
}


void IdealMHD2D::oneStepRK2_periodicXSymmetricY_predictor()
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (IdealMHD2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    thrust::copy(U.begin(), U.end(), UBar.begin());

    fluxF = fluxSolver.getFluxF(U);
    fluxG = fluxSolver.getFluxG(U);

    oneStepFirst_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(fluxF.data()), 
        thrust::raw_pointer_cast(fluxG.data()), 
        thrust::raw_pointer_cast(UBar.data()), 
        IdealMHD2DConst::dt, 
        mPIInfo.localSizeX
    );
    cudaDeviceSynchronize();

    boundaryMHD.periodicBoundaryX2nd_U(UBar);
    boundaryMHD.symmetricBoundaryY2nd_U(UBar);
    MPI_Barrier(MPI_COMM_WORLD);

    checkAndResetExtremeValues(UBar);

    fluxF = fluxSolver.getFluxF(UBar);
    fluxG = fluxSolver.getFluxG(UBar);

    oneStepSecond_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(UBar.data()), 
        thrust::raw_pointer_cast(fluxF.data()), 
        thrust::raw_pointer_cast(fluxG.data()), 
        thrust::raw_pointer_cast(U.data()), 
        mPIInfo.localSizeX
    );
    cudaDeviceSynchronize();

    boundaryMHD.periodicBoundaryX2nd_U(U);
    boundaryMHD.symmetricBoundaryY2nd_U(U);
    MPI_Barrier(MPI_COMM_WORLD);

    checkAndResetExtremeValues(U);
}


void IdealMHD2D::oneStepRK2_periodicXSymmetricY_corrector(
    thrust::device_vector<ConservationParameter>& UHalf
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (IdealMHD2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    checkAndResetExtremeValues(UHalf);

    fluxF = fluxSolver.getFluxF(UHalf);
    fluxG = fluxSolver.getFluxG(UHalf);

    oneStepFirst_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(UPast.data()), 
        thrust::raw_pointer_cast(fluxF.data()), 
        thrust::raw_pointer_cast(fluxG.data()), 
        thrust::raw_pointer_cast(U.data()), 
        IdealMHD2DConst::dt, 
        mPIInfo.localSizeX
    );
    cudaDeviceSynchronize();

    checkAndResetExtremeValues(U);

    boundaryMHD.periodicBoundaryX2nd_U(U);
    boundaryMHD.symmetricBoundaryY2nd_U(U);
    MPI_Barrier(MPI_COMM_WORLD);
}


/*
void IdealMHD2D::oneStepRK2_periodicXSymmetricY_corrector(
    thrust::device_vector<ConservationParameter>& UHalf
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (IdealMHD2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    checkAndResetExtremeValues(UHalf);

    fluxF = fluxSolver.getFluxF(UHalf);
    fluxG = fluxSolver.getFluxG(UHalf);

    oneStepFirst_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(UHalf.data()), 
        thrust::raw_pointer_cast(fluxF.data()), 
        thrust::raw_pointer_cast(fluxG.data()), 
        thrust::raw_pointer_cast(U.data()), 
        IdealMHD2DConst::dt / 2.0, 
        mPIInfo.localSizeX
    );
    cudaDeviceSynchronize();

    checkAndResetExtremeValues(U);

    boundaryMHD.periodicBoundaryX2nd_U(U);
    boundaryMHD.symmetricBoundaryY2nd_U(U);
    MPI_Barrier(MPI_COMM_WORLD);
}
*/


void IdealMHD2D::save(
    std::string directoryname, 
    std::string filenameWithoutStep, 
    int step
)
{
    host_U = U;

    std::string filename;
    filename = directoryname + "/"
             + filenameWithoutStep + "_" + std::to_string(step)
             + "_" + std::to_string(mPIInfo.rank)
             + ".bin";

    std::ofstream ofs(filename, std::ios::binary);
    ofs << std::fixed << std::setprecision(6);

    for (int i = 0; i < mPIInfo.localSizeX; i++) {
        for (int j = 0; j < IdealMHD2DConst::ny; j++) {
            ofs.write(reinterpret_cast<const char*>(&host_U[j + i * IdealMHD2DConst::ny].rho),  sizeof(double));
            ofs.write(reinterpret_cast<const char*>(&host_U[j + i * IdealMHD2DConst::ny].rhoU), sizeof(double));
            ofs.write(reinterpret_cast<const char*>(&host_U[j + i * IdealMHD2DConst::ny].rhoV), sizeof(double));
            ofs.write(reinterpret_cast<const char*>(&host_U[j + i * IdealMHD2DConst::ny].rhoW), sizeof(double));
            ofs.write(reinterpret_cast<const char*>(&host_U[j + i * IdealMHD2DConst::ny].bX),   sizeof(double));
            ofs.write(reinterpret_cast<const char*>(&host_U[j + i * IdealMHD2DConst::ny].bY),   sizeof(double));
            ofs.write(reinterpret_cast<const char*>(&host_U[j + i * IdealMHD2DConst::ny].bZ),   sizeof(double));
            ofs.write(reinterpret_cast<const char*>(&host_U[j + i * IdealMHD2DConst::ny].e),    sizeof(double));
        }
    }
}


__global__ void calculateDtVector_kernel(
    const ConservationParameter* U, 
    double* dtVector, 
    int localNx, int buffer
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localNx && j < IdealMHD2DConst::device_ny) {
        unsigned long long indexForU  = j + (i + buffer) * IdealMHD2DConst::device_ny;
        unsigned long long indexForDt = j + i            * IdealMHD2DConst::device_ny;

        double rho, u, v, w, bX, bY, bZ, e, p, cs, ca;
        double maxSpeedX, maxSpeedY;

        rho = U[indexForU].rho;
        u   = U[indexForU].rhoU / rho;
        v   = U[indexForU].rhoV / rho;
        w   = U[indexForU].rhoW / rho;
        bX  = U[indexForU].bX;
        bY  = U[indexForU].bY;
        bZ  = U[indexForU].bZ;
        e   = U[indexForU].e;
        p   = (IdealMHD2DConst::device_gamma - 1.0)
            * (e - 0.5 * rho * (u * u + v * v + w * w)
            - 0.5 * (bX * bX + bY * bY + bZ * bZ));
        
        cs = sqrt(IdealMHD2DConst::device_gamma * p / rho);
        ca = sqrt((bX * bX + bY * bY + bZ * bZ) / rho);

        maxSpeedX = std::abs(u) + sqrt(cs * cs + ca * ca);
        maxSpeedY = std::abs(v) + sqrt(cs * cs + ca * ca);

        dtVector[indexForDt] = 1.0 / (maxSpeedX / IdealMHD2DConst::device_dx + maxSpeedY / IdealMHD2DConst::device_dy + IdealMHD2DConst::device_EPS);
    
    }
}


void IdealMHD2D::calculateDt()
{
    // localSizeではないので注意
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localNx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (IdealMHD2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    calculateDtVector_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(dtVector.data()), 
        mPIInfo.localNx, mPIInfo.buffer
    );
    cudaDeviceSynchronize();

    thrust::device_vector<double>::iterator dtMin = thrust::min_element(dtVector.begin(), dtVector.end());
    
    IdealMHD2DConst::dt = (*dtMin) * IdealMHD2DConst::CFL;
    
    double dtLocal = IdealMHD2DConst::dt;
    double dtCommon;
    
    MPI_Allreduce(&dtLocal, &dtCommon, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    IdealMHD2DConst::dt = dtCommon;

    cudaMemcpyToSymbol(IdealMHD2DConst::device_dt, &IdealMHD2DConst::dt, sizeof(double));
    cudaDeviceSynchronize();
}


__global__ void checkAndResetExtremeValues_kernel(
    ConservationParameter* U, 
    int localSizeX
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localSizeX && j < IdealMHD2DConst::device_ny) {
        unsigned long long index = j + i * IdealMHD2DConst::device_ny;
        double rho, u, v, w, bX, bY, bZ, e, p;

        rho = U[index].rho;
        u   = U[index].rhoU / rho;
        v   = U[index].rhoV / rho;
        w   = U[index].rhoW / rho;
        bX  = U[index].bX;
        bY  = U[index].bY;
        bZ  = U[index].bZ;
        e   = U[index].e;
        p   = (IdealMHD2DConst::device_gamma - 1.0)
            * (e - 0.5 * rho * (u * u + v * v + w * w)
            - 0.5 * (bX * bX + bY * bY + bZ * bZ));

        rho = thrust::max(rho, 0.1 * IdealMHD2DConst::device_rho0);
        p   = thrust::max(p,   0.1 * IdealMHD2DConst::device_p0);
        rho = thrust::min(rho, 10.0 * IdealMHD2DConst::device_rho0);
        p   = thrust::min(p,   10.0 * IdealMHD2DConst::device_p0);

        double VA = IdealMHD2DConst::device_B0 / sqrt(IdealMHD2DConst::device_rho0);
        u = thrust::max(-10.0 * VA, thrust::min(u, 10.0 * VA));
        v = thrust::max(-10.0 * VA, thrust::min(v, 10.0 * VA));
        w = thrust::max(-10.0 * VA, thrust::min(w, 10.0 * VA));

        bX = thrust::max(-10.0 * IdealMHD2DConst::device_B0, thrust::min(bX, 10.0 * IdealMHD2DConst::device_B0));
        bY = thrust::max(-10.0 * IdealMHD2DConst::device_B0, thrust::min(bY, 10.0 * IdealMHD2DConst::device_B0));
        bZ = thrust::max(-10.0 * IdealMHD2DConst::device_B0, thrust::min(bZ, 10.0 * IdealMHD2DConst::device_B0));

        e = p / (IdealMHD2DConst::device_gamma - 1.0)
          + 0.5 * rho * (u * u + v * v + w * w)
          + 0.5 * (bX * bX + bY * bY + bZ * bZ);

        U[index].rho  = rho;
        U[index].rhoU = rho * u; 
        U[index].rhoV = rho * v;
        U[index].rhoW = rho * w; 
        U[index].bX   = bX;
        U[index].bY   = bY;
        U[index].bZ   = bZ;
        U[index].e    = e; 
        
    }
}

void IdealMHD2D::checkAndResetExtremeValues(
    thrust::device_vector<ConservationParameter>& U
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (IdealMHD2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    checkAndResetExtremeValues_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        mPIInfo.localSizeX
    );
    cudaDeviceSynchronize(); 
}


struct IsNan
{
    __device__ 
    bool operator()(const ConservationParameter U) const {
        return isnan(U.e); // 何かが壊れたらeは壊れるから
    }
};


bool IdealMHD2D::checkCalculationIsCrashed()
{
    bool result = thrust::transform_reduce(
        U.begin(), U.end(), IsNan(), false, thrust::logical_or<bool>()
    );

    bool global_result;
    MPI_Allreduce(&result, &global_result, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);

    if (IdealMHD2DConst::dt < 0) global_result = true;

    return global_result;
}


thrust::host_vector<ConservationParameter>& IdealMHD2D::getHostURef()
{
    return host_U;
}


thrust::device_vector<ConservationParameter>& IdealMHD2D::getURef()
{
    return U;
}


thrust::device_vector<ConservationParameter>& IdealMHD2D::getUPastRef()
{
    return UPast;
}

BoundaryMHD& IdealMHD2D::getBoundaryMHDRef()
{
    return boundaryMHD; 
}

Projection& IdealMHD2D::getProjectionRef()
{
    return projection; 
}
