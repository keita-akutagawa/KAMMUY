#include "main_alfven_const.hpp"
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>


// 別にinitializeUを作ることにする。
void IdealMHD2D::initializeU()
{
}


__global__ void initializeU_Lower_kernel(
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


__global__ void initializeU_Upper_kernel(
    ConservationParameter* U
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < device_nx_MHD && j < device_ny_MHD) {
        double rho, u, v, w, bX, bY, bZ, e, p;
        double y = j * PIC2DConst::device_dy_PIC + 9500 * IdealMHD2DConst::device_dy_MHD + 950 * PIC2DConst::device_dy_PIC;
        
        rho = device_rho0_MHD;
        u   = device_waveAmp * device_VA * cos(device_waveNumber * y);
        v   = 0.0;
        w   = -device_waveAmp * device_VA * sin(device_waveNumber * y);
        bX  = -device_waveAmp * device_b0_MHD * cos(device_waveNumber * y);
        bY  = device_b0_MHD;
        bZ  = device_waveAmp * device_b0_MHD * sin(device_waveNumber * y);
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


void initializeU(
    thrust::device_vector<ConservationParameter>& U_Lower, 
    thrust::device_vector<ConservationParameter>& U_Upper
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx_MHD + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny_MHD + threadsPerBlock.y - 1) / threadsPerBlock.y);

    initializeU_Lower_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U_Lower.data())
    );

    cudaDeviceSynchronize();


    initializeU_Upper_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U_Upper.data())
    );

    cudaDeviceSynchronize();
}


__global__ void initializePICField_kernel(
    ElectricField* E, MagneticField* B
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < device_nx_PIC && j < device_ny_PIC) {
        double u, v, w, bX, bY, bZ, eX, eY, eZ;
        double y = j * PIC2DConst::device_dy_PIC + 9500 * IdealMHD2DConst::device_dy_MHD;

        bX = -device_waveAmp * device_b0_PIC * cos(device_waveNumber * y);
        bY = device_b0_PIC; 
        bZ = device_waveAmp * device_b0_PIC * sin(device_waveNumber * y);
        u = device_waveAmp * device_VA * cos(device_waveNumber * y);
        v = 0.0;
        w = -device_waveAmp * device_VA * sin(device_waveNumber * y);
        eX = -(v * bZ - w * bY);
        eY = -(w * bX - u * bZ);
        eZ = -(u * bY - v * bX);

        E[j + i * device_ny_PIC].eX = eX;
        E[j + i * device_ny_PIC].eY = eY;
        E[j + i * device_ny_PIC].eZ = eZ;
        B[j + i * device_ny_PIC].bX = bX;
        B[j + i * device_ny_PIC].bY = bY; 
        B[j + i * device_ny_PIC].bZ = bZ;
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
        host_interlockingFunctionY_Lower[i] = max(
            0.5 * (1.0 + cos(Interface2DConst::PI * (i - 0.0) / (interfaceLength - 0.0))), 
            1e-20
        );
        host_interlockingFunctionY_Upper[i] = max(
            0.5 * (1.0 - cos(Interface2DConst::PI * (i - 0.0) / (interfaceLength - 0.0))), 
            Interface2DConst::EPS
        );
    }
    for (int i = 0; i < interfaceLength - 1; i++) {
        host_interlockingFunctionYHalf_Lower[i] = max(
            0.5 * (1.0 + cos(Interface2DConst::PI * (i + 0.5 - 0.0) / (interfaceLength - 0.0))), 
            1e-20
        );
        host_interlockingFunctionYHalf_Upper[i] = max(
            0.5 * (1.0 - cos(Interface2DConst::PI * (i + 0.5 - 0.0) / (interfaceLength - 0.0))), 
            Interface2DConst::EPS
        );
    }


    IdealMHD2D idealMHD2D_Lower;
    IdealMHD2D idealMHD2D_Upper;
    PIC2D pIC2D;
    InterfaceNoiseRemover2D interfaceNoiseRemover2D_Lower(
        indexOfInterfaceStartInMHD_Lower, 
        indexOfInterfaceStartInPIC_Lower, 
        interfaceLength, 
        windowSizeForConvolution
    );
    InterfaceNoiseRemover2D interfaceNoiseRemover2D_Upper(
        indexOfInterfaceStartInMHD_Upper, 
        indexOfInterfaceStartInPIC_Upper, 
        interfaceLength, 
        windowSizeForConvolution
    );
    Interface2D interface2D_Lower(
        indexOfInterfaceStartInMHD_Lower, 
        indexOfInterfaceStartInPIC_Lower, 
        interfaceLength, 
        host_interlockingFunctionY_Lower, 
        host_interlockingFunctionYHalf_Lower, 
        interfaceNoiseRemover2D_Lower
    );
    Interface2D interface2D_Upper(
        indexOfInterfaceStartInMHD_Upper, 
        indexOfInterfaceStartInPIC_Upper, 
        interfaceLength, 
        host_interlockingFunctionY_Upper, 
        host_interlockingFunctionYHalf_Upper, 
        interfaceNoiseRemover2D_Upper
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



    thrust::device_vector<ConservationParameter>& U_Lower = idealMHD2D_Lower.getURef();
    thrust::device_vector<ConservationParameter>& U_Upper = idealMHD2D_Upper.getURef();
    initializeU(U_Lower, U_Upper);
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
            idealMHD2D_Lower.save(
                directoryname, filenameWithoutStep + "_lower", step
            );
            idealMHD2D_Upper.save(
                directoryname, filenameWithoutStep + "_upper", step
            );
        }
        if (isParticleRecord && step % particleRecordStep == 0) {
            pIC2D.saveParticle(
                directoryname, filenameWithoutStep, step
            );
        }

        
        idealMHD2D_Lower.calculateDt();
        double dt_Lower_MHD = IdealMHD2DConst::dt_MHD;
        idealMHD2D_Upper.calculateDt();
        double dt_Upper_MHD = IdealMHD2DConst::dt_MHD;
        double dtCommon = min(min(dt_Lower_MHD / substeps, dt_Upper_MHD / substeps), min(0.7 * PIC2DConst::c_PIC, 0.1 * 1.0 / PIC2DConst::omegaPe_PIC));
        PIC2DConst::dt_PIC = dtCommon;
        IdealMHD2DConst::dt_MHD = substeps * dtCommon;

        idealMHD2D_Lower.setPastU();
        idealMHD2D_Upper.setPastU();
        thrust::device_vector<ConservationParameter>& UPast_Lower = idealMHD2D_Lower.getUPastRef();
        thrust::device_vector<ConservationParameter>& UPast_Upper = idealMHD2D_Upper.getUPastRef();
        idealMHD2D_Lower.oneStepRK2_predictor();
        idealMHD2D_Upper.oneStepRK2_predictor();
        thrust::device_vector<ConservationParameter>& UNext_Lower = idealMHD2D_Lower.getURef();
        thrust::device_vector<ConservationParameter>& UNext_Upper = idealMHD2D_Upper.getURef();


        interface2D_Lower.resetTimeAveParameters();
        interface2D_Upper.resetTimeAveParameters();
        for (int substep = 1; substep <= substeps; substep++) {
            pIC2D.oneStepPeriodicXFreeY();

            thrust::device_vector<MagneticField>& B = pIC2D.getBRef();
            thrust::device_vector<ElectricField>& E = pIC2D.getERef();
            thrust::device_vector<CurrentField>& current = pIC2D.getCurrentRef();
            thrust::device_vector<Particle>& particlesIon = pIC2D.getParticlesIonRef();
            thrust::device_vector<Particle>& particlesElectron = pIC2D.getParticlesElectronRef();

            double mixingRatio = (substeps - substep) / substeps;
            thrust::device_vector<ConservationParameter>& USub_Lower = interface2D_Lower.calculateAndGetSubU(UPast_Lower, UNext_Lower, mixingRatio);
            thrust::device_vector<ConservationParameter>& USub_Upper = interface2D_Upper.calculateAndGetSubU(UPast_Upper, UNext_Upper, mixingRatio);
            
            interface2D_Lower.sendMHDtoPIC_magneticField_yDirection(USub_Lower, B);
            interface2D_Lower.sendMHDtoPIC_electricField_yDirection(USub_Lower, E);
            interface2D_Lower.sendMHDtoPIC_currentField_yDirection(USub_Lower, current);
            interface2D_Lower.sendMHDtoPIC_particle(USub_Lower, particlesIon, particlesElectron, step * substeps + substep);
            interface2D_Upper.sendMHDtoPIC_magneticField_yDirection(USub_Upper, B);
            interface2D_Upper.sendMHDtoPIC_electricField_yDirection(USub_Upper, E);
            interface2D_Upper.sendMHDtoPIC_currentField_yDirection(USub_Upper, current);
            interface2D_Upper.sendMHDtoPIC_particle(USub_Upper, particlesIon, particlesElectron, step * substeps + substep);

            interfaceNoiseRemover2D_Lower.convolveFields(B, E, current);
            interfaceNoiseRemover2D_Upper.convolveFields(B, E, current);

            boundaryPIC.freeBoundaryBY(B);
            boundaryPIC.freeBoundaryEY(E);
            boundaryPIC.freeBoundaryCurrentY(current); 
            boundaryPIC.periodicBoundaryParticleX(particlesIon, particlesElectron);
            boundaryPIC.openBoundaryParticleY(particlesIon, particlesElectron);

            interface2D_Lower.sumUpTimeAveParameters(B, particlesIon, particlesElectron);
            interface2D_Upper.sumUpTimeAveParameters(B, particlesIon, particlesElectron);
        }

        interface2D_Lower.calculateTimeAveParameters(substeps);
        interface2D_Upper.calculateTimeAveParameters(substeps);


        interface2D_Lower.sendPICtoMHD(UPast_Lower, UNext_Lower);
        interface2D_Upper.sendPICtoMHD(UPast_Upper, UNext_Upper);
        thrust::device_vector<ConservationParameter>& UHalf_Lower = interface2D_Lower.getUHalfRef();
        thrust::device_vector<ConservationParameter>& UHalf_Upper = interface2D_Upper.getUHalfRef();
        boundaryMHD.periodicBoundaryX2nd(UHalf_Lower);
        boundaryMHD.symmetricBoundaryY2nd(UHalf_Lower);
        boundaryMHD.periodicBoundaryX2nd(UHalf_Upper);
        boundaryMHD.symmetricBoundaryY2nd(UHalf_Upper);

        idealMHD2D_Lower.oneStepRK2_corrector(UHalf_Lower);
        idealMHD2D_Upper.oneStepRK2_corrector(UHalf_Upper);
        U_Lower = idealMHD2D_Lower.getURef();
        U_Upper = idealMHD2D_Upper.getURef();
        interfaceNoiseRemover2D_Lower.convolveU(U_Lower);
        interfaceNoiseRemover2D_Upper.convolveU(U_Upper);
        boundaryMHD.periodicBoundaryX2nd(U_Lower);
        boundaryMHD.symmetricBoundaryY2nd(U_Lower);
        boundaryMHD.periodicBoundaryX2nd(U_Upper);
        boundaryMHD.symmetricBoundaryY2nd(U_Upper);

        if (idealMHD2D_Lower.checkCalculationIsCrashed() || idealMHD2D_Upper.checkCalculationIsCrashed()) {
            std::cout << "Calculation stopped! : " << step << " steps" << std::endl;
            return 0;
        }

        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(error));
            exit(-1);
        }

        IdealMHD2DConst::totalTime_MHD += IdealMHD2DConst::dt_MHD;
    }

    return 0;
}


