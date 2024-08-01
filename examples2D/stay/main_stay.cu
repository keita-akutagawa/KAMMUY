#include "../../IdealMHD2D_gpu/IdealMHD2D.hpp"
#include "../../PIC2D_gpu_single/PIC2D.hpp"
#include "../../Interface2D/interface.hpp"
#include "main_stay_const.hpp"
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>



__global__ void initializePICField_kernel(
    ElectricField* E, MagneticField* B
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < device_nx_PIC && j < device_ny_PIC) {
        E[j + i * device_ny_PIC].eX = 0.0f;
        E[j + i * device_ny_PIC].eY = 0.0f;
        E[j + i * device_ny_PIC].eZ = 0.0f;
        B[j + i * device_ny_PIC].bX = 0.0f;
        B[j + i * device_ny_PIC].bY = 0.0f; 
        B[j + i * device_ny_PIC].bZ = 0.0f;
    }
}

void PIC2D::initialize()
{
    initializeParticle.uniformForPositionX(
        0, existNumIon_PIC, 0, particlesIon
    );
    initializeParticle.uniformForPositionX(
        0, existNumElectron_PIC, 100, particlesElectron
    );
    initializeParticle.uniformForPositionY(
        0, existNumIon_PIC, 200, particlesIon
    );
    initializeParticle.uniformForPositionY(
        0, existNumElectron_PIC, 300, particlesElectron
    );

    initializeParticle.maxwellDistributionForVelocity(
        bulkVxIon_PIC, bulkVyIon_PIC, bulkVzIon_PIC, vThIon_PIC, vThIon_PIC, vThIon_PIC, 
        0, existNumIon_PIC, 400, particlesIon
    );
    initializeParticle.maxwellDistributionForVelocity(
        bulkVxElectron_PIC, bulkVyElectron_PIC, bulkVzElectron_PIC, vThElectron_PIC, vThElectron_PIC, vThElectron_PIC, 
        0, existNumElectron_PIC, 500, particlesElectron
    );


    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny_PIC + threadsPerBlock.y - 1) / threadsPerBlock.y);

    initializePICField_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(E.data()), thrust::raw_pointer_cast(B.data())
    );

    cudaDeviceSynchronize();
}


__global__ void initializeU_kernel(
    ConservationParameter* U
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < device_nx_MHD && j < device_ny_MHD) {
        U[j + i * device_ny_MHD].rho = device_rho0_MHD;
        U[j + i * device_ny_MHD].rhoU = device_rho0_MHD * device_u0_MHD;
        U[j + i * device_ny_MHD].rhoV = device_rho0_MHD * device_v0_MHD;
        U[j + i * device_ny_MHD].rhoW = device_rho0_MHD * device_w0_MHD;
        U[j + i * device_ny_MHD].bX = device_bX0_MHD;
        U[j + i * device_ny_MHD].bY = device_bY0_MHD;
        U[j + i * device_ny_MHD].bZ = device_bZ0_MHD;
        U[j + i * device_ny_MHD].e = device_e0_MHD;
    }
}

void IdealMHD2D::initializeU()
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx_MHD + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny_MHD + threadsPerBlock.y - 1) / threadsPerBlock.y);

    initializeU_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data())
    );

    cudaDeviceSynchronize();
}


int main()
{
    initializeDeviceConstants_PIC();
    initializeDeviceConstants_MHD();
    initializeDeviceConstants_Interface();

    IdealMHD2D idealMHD2D;
    PIC2D pIC2D;
    Interface2D interface2D(
        indexOfInterfaceStartInMHD, 
        indexOfInterfaceStartInPIC, 
        interfaceLength
    );

    size_t free_mem = 0;
    size_t total_mem = 0;
    cudaError_t status = cudaMemGetInfo(&free_mem, &total_mem);

    std::cout << "Free memory: " << free_mem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Total memory: " << total_mem / (1024 * 1024) << " MB" << std::endl;


    idealMHD2D.initializeU();
    pIC2D.initialize();

    for (int step = 0; step < IdealMHD2DConst::totalStep_MHD; step++) {
        
        idealMHD2D.setPastU();
        thrust::device_vector<ConservationParameter>& UPast = idealMHD2D.getURef();
        idealMHD2D.oneStepRK2_predictor();
        thrust::device_vector<ConservationParameter>& UNext = idealMHD2D.getURef();

        int substeps = round(sqrt(PIC2DConst::mRatio_PIC));
        PIC2DConst::dt_PIC = IdealMHD2DConst::dt_MHD / substeps;
        interface2D.resetTimeAveParameters();
        for (int substep = 0; substep < substeps; substep++) {
            pIC2D.oneStepWallXFreeY();

            thrust::device_vector<MagneticField>& B = pIC2D.getBRef();
            thrust::device_vector<ElectricField>& E = pIC2D.getERef();
            thrust::device_vector<CurrentField>& current = pIC2D.getCurrentRef();
            thrust::device_vector<Particle>& particlesIon = pIC2D.getParticlesIonRef();
            thrust::device_vector<Particle>& particlesElectron = pIC2D.getParticlesElectronRef();

            float mixingRatio = (substeps - substep) / substeps;
            thrust::device_vector<ConservationParameter>& USub = interface2D.calculateAndGetSubU(UPast, UNext, mixingRatio);

            interface2D.sendMHDtoPIC_magneticField_yDirection(USub, B);
            interface2D.sendMHDtoPIC_electricField_yDirection(USub, E);
            interface2D.sendMHDtoPIC_currentField_yDirection(USub, current);
            interface2D.sendMHDtoPIC_particle(USub, particlesIon, particlesElectron, step * substeps + substep);

            interface2D.sumUpTimeAveParameters(B, particlesIon, particlesElectron);
        }
        interface2D.calculateTimeAveParameters(substeps);

        interface2D.sendPICtoMHD(UPast, UNext);
        thrust::device_vector<ConservationParameter>& UHalf = interface2D.getUHalfRef();

        idealMHD2D.oneStepRK2_corrector(UHalf);

    }

    return 0;
}


