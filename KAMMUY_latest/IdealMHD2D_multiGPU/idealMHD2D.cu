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
      hU       (mPIInfo.localSizeX * IdealMHD2DConst::ny), 

      dtVector(mPIInfo.localNx * IdealMHD2DConst::ny), 

      boundaryMHD(mPIInfo)
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
    int localSizeX
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < localSizeX) && (0 < j) && (j < IdealMHD2DConst::device_ny)) {
        int index = j + i * IdealMHD2DConst::device_ny;

        UBar[index].rho  = U[index].rho  
                         - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dx * (fluxF[index].f0 - fluxF[index - IdealMHD2DConst::device_ny].f0)
                         - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dy * (fluxG[index].f0 - fluxG[index - 1].f0);
        UBar[index].rhoU = U[index].rhoU 
                         - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dx * (fluxF[index].f1 - fluxF[index - IdealMHD2DConst::device_ny].f1)
                         - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dy * (fluxG[index].f1 - fluxG[index - 1].f1);
        UBar[index].rhoV = U[index].rhoV
                         - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dx * (fluxF[index].f2 - fluxF[index - IdealMHD2DConst::device_ny].f2)
                         - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dy * (fluxG[index].f2 - fluxG[index - 1].f2);
        UBar[index].rhoW = U[index].rhoW
                         - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dx * (fluxF[index].f3 - fluxF[index - IdealMHD2DConst::device_ny].f3)
                         - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dy * (fluxG[index].f3 - fluxG[index - 1].f3);
        UBar[index].bX   = U[index].bX 
                         - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dx * (fluxF[index].f4 - fluxF[index - IdealMHD2DConst::device_ny].f4)
                         - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dy * (fluxG[index].f4 - fluxG[index - 1].f4);
        UBar[index].bY   = U[index].bY 
                         - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dx * (fluxF[index].f5 - fluxF[index - IdealMHD2DConst::device_ny].f5)
                         - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dy * (fluxG[index].f5 - fluxG[index - 1].f5);
        UBar[index].bZ   = U[index].bZ 
                         - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dx * (fluxF[index].f6 - fluxF[index - IdealMHD2DConst::device_ny].f6)
                         - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dy * (fluxG[index].f6 - fluxG[index - 1].f6);
        UBar[index].e    = U[index].e 
                         - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dx * (fluxF[index].f7 - fluxF[index - IdealMHD2DConst::device_ny].f7)
                         - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dy * (fluxG[index].f7 - fluxG[index - 1].f7);
        UBar[index].psi  = U[index].psi 
                         - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dx * (fluxF[index].f8 - fluxF[index - IdealMHD2DConst::device_ny].f8)
                         - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dy * (fluxG[index].f8 - fluxG[index - 1].f8)
                         - IdealMHD2DConst::device_dt * pow(IdealMHD2DConst::device_ch / IdealMHD2DConst::device_cp, 2) * U[index].psi;
    }
}


__global__ void oneStepSecond_kernel(
    const ConservationParameter* UBar, 
    const Flux* fluxF, const Flux* fluxG, 
    ConservationParameter* U, 
    int localSizeX
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < localSizeX) && (0 < j) && (j < IdealMHD2DConst::device_ny)) {
        int index = j + i * IdealMHD2DConst::device_ny;

        U[index].rho  = 0.5 * (U[index].rho + UBar[index].rho
                      - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dx * (fluxF[index].f0 - fluxF[index - IdealMHD2DConst::device_ny].f0)
                      - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dy * (fluxG[index].f0 - fluxG[index - 1].f0));
        U[index].rhoU = 0.5 * (U[index].rhoU + UBar[index].rhoU
                      - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dx * (fluxF[index].f1 - fluxF[index - IdealMHD2DConst::device_ny].f1)
                      - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dy * (fluxG[index].f1 - fluxG[index - 1].f1));
        U[index].rhoV = 0.5 * (U[index].rhoV + UBar[index].rhoV
                      - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dx * (fluxF[index].f2 - fluxF[index - IdealMHD2DConst::device_ny].f2)
                      - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dy * (fluxG[index].f2 - fluxG[index - 1].f2));
        U[index].rhoW = 0.5 * (U[index].rhoW + UBar[index].rhoW
                      - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dx * (fluxF[index].f3 - fluxF[index - IdealMHD2DConst::device_ny].f3)
                      - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dy * (fluxG[index].f3 - fluxG[index - 1].f3));
        U[index].bX   = 0.5 * (U[index].bX + UBar[index].bX
                      - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dx * (fluxF[index].f4 - fluxF[index - IdealMHD2DConst::device_ny].f4)
                      - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dy * (fluxG[index].f4 - fluxG[index - 1].f4));
        U[index].bY   = 0.5 * (U[index].bY + UBar[index].bY
                      - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dx * (fluxF[index].f5 - fluxF[index - IdealMHD2DConst::device_ny].f5)
                      - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dy * (fluxG[index].f5 - fluxG[index - 1].f5));
        U[index].bZ   = 0.5 * (U[index].bZ + UBar[index].bZ
                      - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dx * (fluxF[index].f6 - fluxF[index - IdealMHD2DConst::device_ny].f6)
                      - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dy * (fluxG[index].f6 - fluxG[index - 1].f6));
        U[index].e    = 0.5 * (U[index].e + UBar[index].e
                      - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dx * (fluxF[index].f7 - fluxF[index - IdealMHD2DConst::device_ny].f7)
                      - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dy * (fluxG[index].f7 - fluxG[index - 1].f7));
        U[index].psi  = 0.5 * (U[index].psi + UBar[index].psi
                      - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dx * (fluxF[index].f8 - fluxF[index - IdealMHD2DConst::device_ny].f8)
                      - IdealMHD2DConst::device_dt / IdealMHD2DConst::device_dy * (fluxG[index].f8 - fluxG[index - 1].f8))
                      - IdealMHD2DConst::device_dt / 2.0 * pow(IdealMHD2DConst::device_ch / IdealMHD2DConst::device_cp, 2) * 0.5 * (U[index].psi + UBar[index].psi);
    }
}


void IdealMHD2D::oneStepRK2_periodicXSymmetricY_predictor()
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (IdealMHD2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);


    MPI_Barrier(MPI_COMM_WORLD);
    
    thrust::copy(U.begin(), U.end(), UBar.begin());
    cudaDeviceSynchronize();

    //calculateDt();
    IdealMHD2DConst::ch = IdealMHD2DConst::CFL / 2.0 * min(IdealMHD2DConst::dx, IdealMHD2DConst::dy) / IdealMHD2DConst::dt; 
    IdealMHD2DConst::cp = sqrt(IdealMHD2DConst::cr * IdealMHD2DConst::ch); 
    cudaMemcpyToSymbol(IdealMHD2DConst::device_ch, &IdealMHD2DConst::ch, sizeof(double));
    cudaMemcpyToSymbol(IdealMHD2DConst::device_cp, &IdealMHD2DConst::cp, sizeof(double));

    fluxF = fluxSolver.getFluxF(U);
    fluxG = fluxSolver.getFluxG(U);

    oneStepFirst_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(fluxF.data()), 
        thrust::raw_pointer_cast(fluxG.data()), 
        thrust::raw_pointer_cast(UBar.data()), 
        mPIInfo.localSizeX
    );
    cudaDeviceSynchronize();

    boundaryMHD.periodicBoundaryX2nd_U(UBar);
    boundaryMHD.symmetricBoundaryY2nd_U(UBar);
    MPI_Barrier(MPI_COMM_WORLD);

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

    checkAndResetExtremeValues();
}


void IdealMHD2D::oneStepRK2_periodicXSymmetricY_corrector(
    thrust::device_vector<ConservationParameter>& UHalf
)
{
    MPI_Barrier(MPI_COMM_WORLD);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (IdealMHD2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    fluxF = fluxSolver.getFluxF(UHalf);
    fluxG = fluxSolver.getFluxG(UHalf);

    oneStepFirst_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(UPast.data()), 
        thrust::raw_pointer_cast(fluxF.data()), 
        thrust::raw_pointer_cast(fluxG.data()), 
        thrust::raw_pointer_cast(U.data()), 
        mPIInfo.localSizeX
    );
    cudaDeviceSynchronize();

    boundaryMHD.periodicBoundaryX2nd_U(U);
    boundaryMHD.symmetricBoundaryY2nd_U(U);
    MPI_Barrier(MPI_COMM_WORLD);

    checkAndResetExtremeValues();
}


void IdealMHD2D::save(
    std::string directoryname, 
    std::string filenameWithoutStep, 
    int step
)
{
    hU = U;

    std::string filename;
    filename = directoryname + "/"
             + filenameWithoutStep + "_" + std::to_string(step)
             + "_" + std::to_string(mPIInfo.rank)
             + ".bin";

    std::ofstream ofs(filename, std::ios::binary);
    ofs << std::fixed << std::setprecision(6);

    for (int i = 0; i < mPIInfo.localSizeX; i++) {
        for (int j = 0; j < IdealMHD2DConst::ny; j++) {
            ofs.write(reinterpret_cast<const char*>(&hU[j + i * IdealMHD2DConst::ny].rho),  sizeof(double));
            ofs.write(reinterpret_cast<const char*>(&hU[j + i * IdealMHD2DConst::ny].rhoU), sizeof(double));
            ofs.write(reinterpret_cast<const char*>(&hU[j + i * IdealMHD2DConst::ny].rhoV), sizeof(double));
            ofs.write(reinterpret_cast<const char*>(&hU[j + i * IdealMHD2DConst::ny].rhoW), sizeof(double));
            ofs.write(reinterpret_cast<const char*>(&hU[j + i * IdealMHD2DConst::ny].bX),   sizeof(double));
            ofs.write(reinterpret_cast<const char*>(&hU[j + i * IdealMHD2DConst::ny].bY),   sizeof(double));
            ofs.write(reinterpret_cast<const char*>(&hU[j + i * IdealMHD2DConst::ny].bZ),   sizeof(double));
            ofs.write(reinterpret_cast<const char*>(&hU[j + i * IdealMHD2DConst::ny].e),    sizeof(double));
        }
    }
}


__global__ void calculateDtVector_kernel(
    const ConservationParameter* U, 
    double* dtVector, 
    int localNx, int buffer
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localNx && j < IdealMHD2DConst::device_ny) {
        int indexForU  = j + (i + buffer) * IdealMHD2DConst::device_ny;
        int indexForDt = j + i            * IdealMHD2DConst::device_ny;

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
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localSizeX && j < IdealMHD2DConst::device_ny) {
        int index = j + i * IdealMHD2DConst::device_ny;
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

        if (rho < IdealMHD2DConst::device_rho0 * 0.1) {
            rho = IdealMHD2DConst::device_rho0 * 0.1; 
        }
        if (p < IdealMHD2DConst::device_p0 * 0.1) {
            p = IdealMHD2DConst::device_p0 * 0.1; 
        }

        if (rho > IdealMHD2DConst::device_rho0 * 10.0) {
            rho = IdealMHD2DConst::device_rho0 * 10.0; 
        }
        if (p > IdealMHD2DConst::device_p0 * 10.0) {
            p = IdealMHD2DConst::device_p0 * 10.0; 
        }

        double VA = IdealMHD2DConst::device_B0 / sqrt(IdealMHD2DConst::device_rho0); 
        if (u > VA * 10.0) {
            u = VA * 10.0; 
        }
        if (v > VA * 10.0) {
            v = VA * 10.0; 
        }
        if (w > VA * 10.0) {
            w = VA * 10.0; 
        }
        if (u < -VA * 10.0) {
            u = -VA * 10.0; 
        }
        if (v < -VA * 10.0) {
            v = -VA * 10.0; 
        }
        if (w < -VA * 10.0) {
            w = -VA * 10.0; 
        }

        if (bX > IdealMHD2DConst::device_B0 * 10.0) {
            bX = IdealMHD2DConst::device_B0 * 10.0; 
        }
        if (bY > IdealMHD2DConst::device_B0 * 10.0) {
            bY = IdealMHD2DConst::device_B0 * 10.0; 
        }
        if (bZ > IdealMHD2DConst::device_B0 * 10.0) {
            bZ = IdealMHD2DConst::device_B0 * 10.0; 
        }
        if (bX < -IdealMHD2DConst::device_B0 * 10.0) {
            bX = -IdealMHD2DConst::device_B0 * 10.0; 
        }
        if (bY < -IdealMHD2DConst::device_B0 * 10.0) {
            bY = -IdealMHD2DConst::device_B0 * 10.0; 
        }
        if (bZ < -IdealMHD2DConst::device_B0 * 10.0) {
            bZ = -IdealMHD2DConst::device_B0 * 10.0; 
        }


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

void IdealMHD2D::checkAndResetExtremeValues()
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (IdealMHD2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    checkAndResetExtremeValues_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        mPIInfo.localSizeX
    );
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
    return hU;
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


