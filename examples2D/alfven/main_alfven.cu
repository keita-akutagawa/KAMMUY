#include "main_alfven_const.hpp"
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
        E[j + i * device_ny_PIC].eX = 0.0;
        E[j + i * device_ny_PIC].eY = 0.0;
        E[j + i * device_ny_PIC].eZ = 0.0;
        B[j + i * device_ny_PIC].bX = -device_waveAmp * device_b0_MHD * cos(device_waveNumber * (j * PIC2DConst::device_dy_PIC + 950 * IdealMHD2DConst::device_dy_MHD));
        B[j + i * device_ny_PIC].bY = device_b0_PIC; 
        B[j + i * device_ny_PIC].bZ = device_waveAmp * device_b0_MHD * sin(device_waveNumber * (j * PIC2DConst::device_dy_PIC + 950 * IdealMHD2DConst::device_dy_MHD));
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

    for (int j = 0; j < PIC2DConst::ny_PIC; j++) {
        double u, v, w;
        u = waveAmp * VA * cos(waveNumber * (j * PIC2DConst::dy_PIC + 950 * IdealMHD2DConst::dy_MHD));
        v = 0.0;
        w = -waveAmp * VA * sin(waveNumber * (j * PIC2DConst::dy_PIC + 950 * IdealMHD2DConst::dy_MHD));

        initializeParticle.maxwellDistributionForVelocity(
            u, v, w, vThIon_PIC, vThIon_PIC, vThIon_PIC, 
            j * PIC2DConst::nx_PIC * numberDensityIon_PIC, (j + 1) * PIC2DConst::nx_PIC * numberDensityIon_PIC, j * 100 + 400, particlesIon
        );
        initializeParticle.maxwellDistributionForVelocity(
            u, v, w, vThElectron_PIC, vThElectron_PIC, vThElectron_PIC, 
            j * PIC2DConst::nx_PIC * numberDensityElectron_PIC, (j + 1) * PIC2DConst::nx_PIC * numberDensityElectron_PIC, j * 100 + 400 + totalNumIon_PIC, particlesElectron
        );
    }
    

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
        double rho, u, v, w, bX, bY, bZ, e, p;
        
        rho = device_rho0_MHD;
        u   = device_waveAmp * device_VA * cos(device_waveNumber * j * IdealMHD2DConst::device_dy_MHD);
        v   = 0.0;
        w   = -device_waveAmp * device_VA * sin(device_waveNumber * j * IdealMHD2DConst::device_dy_MHD);
        bX  = -device_waveAmp * device_b0_MHD * cos(device_waveNumber * j * IdealMHD2DConst::device_dy_MHD);
        bY  = device_b0_MHD;
        bZ  = device_waveAmp * device_b0_MHD * sin(device_waveNumber * j * IdealMHD2DConst::device_dy_MHD);
        p   = device_p0_MHD;
        e   = p / (IdealMHD2DConst::device_gamma_MHD - 1.0)
            + 0.5 * rho * (u * u + v * v + w * w)
            + 0.5 * (bX * bX + bY * bY + bZ * bZ);

        U[j + i * device_ny_MHD].rho  = rho;
        U[j + i * device_ny_MHD].rhoU = rho * u;
        U[j + i * device_ny_MHD].rhoV = rho * v;
        U[j + i * device_ny_MHD].rhoW = rho * w;
        U[j + i * device_ny_MHD].bX   = bX;
        U[j + i * device_ny_MHD].bY   = bY;
        U[j + i * device_ny_MHD].bZ   = bZ;
        U[j + i * device_ny_MHD].e    = e;
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
    cudaMemcpyToSymbol(device_VA, &VA, sizeof(double));
    cudaMemcpyToSymbol(device_waveAmp, &waveAmp, sizeof(double));
    cudaMemcpyToSymbol(device_waveLength, &waveLength, sizeof(double));
    cudaMemcpyToSymbol(device_waveNumber, &waveNumber, sizeof(double));

    initializeDeviceConstants_PIC();
    initializeDeviceConstants_MHD();
    initializeDeviceConstants_Interface();
    for (int i = 0; i < interfaceLength; i++) {
        host_interlockingFunctionY[i] = max(
            0.5 * (1.0 + cos(Interface2DConst::PI * (i - 0.0) / (interfaceLength - 0.0))), 
            1e-20
        );
    }
    for (int i = 0; i < interfaceLength - 1; i++) {
        host_interlockingFunctionYHalf[i] = max(
            0.5 * (1.0 + cos(Interface2DConst::PI * (i + 0.5 - 0.0) / (interfaceLength - 0.0))), 
            1e-20
        );
    }


    IdealMHD2D idealMHD2D;
    PIC2D pIC2D;
    InterfaceNoiseRemover2D interfaceNoiseRemover2D(
        indexOfInterfaceStartInMHD, 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        windowSizeForConvolution
    );
    Interface2D interface2D(
        indexOfInterfaceStartInMHD, 
        indexOfInterfaceStartInPIC, 
        interfaceLength, 
        host_interlockingFunctionY, 
        host_interlockingFunctionYHalf, 
        interfaceNoiseRemover2D
    );
    BoundaryPIC boundaryPIC;
    BoundaryMHD boundaryMHD;
    

    size_t free_mem = 0;
    size_t total_mem = 0;
    cudaError_t status = cudaMemGetInfo(&free_mem, &total_mem);

    std::cout << "Free memory: " << free_mem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Total memory: " << total_mem / (1024 * 1024) << " MB" << std::endl;

    std::cout << "total number of particles is " 
              << PIC2DConst::totalNumIon_PIC + PIC2DConst::totalNumElectron_PIC << std::endl;


    idealMHD2D.initializeU();
    pIC2D.initialize();

    const int substeps = int(round(sqrt(PIC2DConst::mRatio_PIC)));
    for (int step = 0; step < IdealMHD2DConst::totalStep_MHD + 1; step++) {
        if (step % recordStep == 0) {
            std::cout << std::to_string(step) << " step done : total time is "
                      << std::setprecision(4) << step * substeps * PIC2DConst::dt_PIC * PIC2DConst::omegaPe_PIC
                      << " [omega_pe * t]"
                      << std::endl;
            logfile << std::setprecision(6) << PIC2DConst::totalTime_PIC << std::endl;
            pIC2D.saveFields(
                directoryname, filenameWithoutStep, step
            );
            pIC2D.saveZerothMoments(
                directoryname, filenameWithoutStep, step
            );
            pIC2D.saveFirstMoments(
                directoryname, filenameWithoutStep, step
            );
            idealMHD2D.save(
                directoryname, filenameWithoutStep, step
            );
        }
        if (isParticleRecord && step % particleRecordStep == 0) {
            pIC2D.saveParticle(
                directoryname, filenameWithoutStep, step
            );
        }

        
        idealMHD2D.calculateDt();
        double dtCommon = min(IdealMHD2DConst::dt_MHD / substeps, 0.5 * PIC2DConst::c_PIC);
        PIC2DConst::dt_PIC = dtCommon;
        IdealMHD2DConst::dt_MHD = substeps * dtCommon;

        idealMHD2D.setPastU();
        thrust::device_vector<ConservationParameter>& UPast = idealMHD2D.getUPastRef();
        idealMHD2D.oneStepRK2_predictor();
        thrust::device_vector<ConservationParameter>& UNext = idealMHD2D.getURef();


        interface2D.resetTimeAveParameters();
        for (int substep = 1; substep <= substeps; substep++) {
            pIC2D.oneStepPeriodicXFreeY();

            thrust::device_vector<MagneticField>& B = pIC2D.getBRef();
            thrust::device_vector<ElectricField>& E = pIC2D.getERef();
            thrust::device_vector<CurrentField>& current = pIC2D.getCurrentRef();
            thrust::device_vector<Particle>& particlesIon = pIC2D.getParticlesIonRef();
            thrust::device_vector<Particle>& particlesElectron = pIC2D.getParticlesElectronRef();

            double mixingRatio = (substeps - substep) / substeps;
            thrust::device_vector<ConservationParameter>& USub = interface2D.calculateAndGetSubU(UPast, UNext, mixingRatio);
            
            interface2D.sendMHDtoPIC_magneticField_yDirection(USub, B);
            interface2D.sendMHDtoPIC_electricField_yDirection(USub, E);
            interface2D.sendMHDtoPIC_currentField_yDirection(USub, current);
            interface2D.sendMHDtoPIC_particle(USub, particlesIon, particlesElectron, step * substeps + substep);

            interfaceNoiseRemover2D.convolveFields(B, E, current);

            boundaryPIC.freeBoundaryBY(B);
            boundaryPIC.freeBoundaryEY(E);
            boundaryPIC.freeBoundaryCurrentY(current); 
            boundaryPIC.periodicBoundaryParticleX(particlesIon, particlesElectron);
            boundaryPIC.openBoundaryParticleY(particlesIon, particlesElectron);

            interface2D.sumUpTimeAveParameters(B, particlesIon, particlesElectron);

        }

        interface2D.calculateTimeAveParameters(substeps);


        interface2D.sendPICtoMHD(UPast, UNext);
        thrust::device_vector<ConservationParameter>& UHalf = interface2D.getUHalfRef();
        boundaryMHD.periodicBoundaryX2nd(UHalf);
        boundaryMHD.symmetricBoundaryY2nd(UHalf);

        idealMHD2D.oneStepRK2_corrector(UHalf);
        thrust::device_vector<ConservationParameter>& U = idealMHD2D.getURef();
        interfaceNoiseRemover2D.convolveU(U);
        boundaryMHD.periodicBoundaryX2nd(U);
        boundaryMHD.symmetricBoundaryY2nd(U);

        if (idealMHD2D.checkCalculationIsCrashed()) {
            std::cout << "Calculation stopped! : " << step << " steps" << std::endl;
            return 0;
        }

        IdealMHD2DConst::totalTime_MHD += IdealMHD2DConst::dt_MHD;
    }

    return 0;
}


