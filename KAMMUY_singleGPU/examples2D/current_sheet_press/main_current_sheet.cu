#include "main_current_sheet_const.hpp"



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
        double VA = device_b0_MHD / sqrt(device_rho0_MHD);
        
        rho = device_rho0_MHD * sqrt(device_betaUpstream);
        u   = 0.0;
        v   = 1.0 * VA * 0.5 * (-tanh((j - 0.8 * device_ny_MHD) / 100) + 1.0);
        w   = 0.0;
        bX  = -1.0 * device_b0_MHD;
        bY  = 0.0;
        bZ  = 0.0;
        p   = device_p0_MHD * device_betaUpstream;
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
        double VA = device_b0_MHD / sqrt(device_rho0_MHD);
        
        rho = device_rho0_MHD * sqrt(device_betaUpstream);
        u   = 0.0;
        v   = 1.0 * VA * 0.5 * (-tanh((j - 0.2 * device_ny_MHD) / 100) - 1.0);
        w   = 0.0;
        bX  = 1.0 * device_b0_MHD;
        bY  = 0.0;
        bZ  = 0.0;
        p   = device_p0_MHD * device_betaUpstream;
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
        double bX, bY, bZ, eX, eY, eZ;
        double y = j * device_dy_PIC;
        double xCenter = 0.5 * (device_xmax_PIC - device_xmin_PIC);
        double yCenter = 0.5 * (device_ymax_PIC - device_ymin_PIC);

        bX = device_b0_PIC * tanh((y - yCenter) / device_sheatThickness)
           - device_b0_PIC * device_triggerRatio * (j * device_dy_PIC - yCenter) / device_sheatThickness
           * exp(-(pow((i * device_dx_PIC - xCenter), 2) + pow((j * device_dy_PIC - yCenter), 2))
           / pow(2.0f * device_sheatThickness, 2));;
        bY = device_b0_PIC * device_triggerRatio * (i * device_dx_PIC - xCenter) / device_sheatThickness
           * exp(-(pow((i * device_dx_PIC - xCenter), 2) + pow((j * device_dy_PIC - yCenter), 2))
           / pow(2.0f * device_sheatThickness, 2)); 
        bZ = 0.0;
        eX = 0.0;
        eY = 0.0;
        eZ = 0.0;

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

    initializeParticle.harrisForPositionY(
        0, harrisNumIon_PIC, 200, sheatThickness, particlesIon
    );
    initializeParticle.uniformForPositionY(
        harrisNumIon_PIC, existNumIon_PIC, 300, particlesIon
    );
    initializeParticle.harrisForPositionY(
        0, harrisNumElectron_PIC, 400, sheatThickness, particlesElectron
    );
    initializeParticle.uniformForPositionY(
        harrisNumElectron_PIC, existNumElectron_PIC, 500, particlesElectron
    );

    initializeParticle.maxwellDistributionForVelocity(
        bulkVxIon_PIC, bulkVyIon_PIC, bulkVzIon_PIC, vThIon_PIC, vThIon_PIC, vThIon_PIC, 
        0, harrisNumIon_PIC, 600, particlesIon
    );
    initializeParticle.maxwellDistributionForVelocity(
        bulkVxIonB_PIC, bulkVyIonB_PIC, bulkVzIonB_PIC, vThIonB_PIC, vThIonB_PIC, vThIonB_PIC, 
        harrisNumIon_PIC, existNumIon_PIC, 700, particlesIon
    );
    initializeParticle.maxwellDistributionForVelocity(
        bulkVxElectron_PIC, bulkVyElectron_PIC, bulkVzElectron_PIC, vThElectron_PIC, vThElectron_PIC, vThElectron_PIC, 
        0, harrisNumElectron_PIC, 800, particlesElectron
    );
    initializeParticle.maxwellDistributionForVelocity(
        bulkVxElectronB_PIC, bulkVyElectronB_PIC, bulkVzElectronB_PIC, vThElectronB_PIC, vThElectronB_PIC, vThElectronB_PIC, 
        harrisNumElectron_PIC, existNumElectron_PIC, 900, particlesElectron
    );
    

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
    cudaMemcpyToSymbol(device_sheatThickness, &sheatThickness, sizeof(double));
    cudaMemcpyToSymbol(device_betaUpstream, &betaUpstream, sizeof(double));
    cudaMemcpyToSymbol(device_triggerRatio, &triggerRatio, sizeof(double));
     
    initializeDeviceConstants_PIC();
    initializeDeviceConstants_MHD();
    initializeDeviceConstants_Interface();
    for (int i = 0; i < interfaceLength; i++) {
        host_interlockingFunctionY_Lower[i] = max(
            0.5 * (1.0 + cos(Interface2DConst::PI * (i - 0.0) / (interfaceLength - 0.0))), 
            Interface2DConst::EPS
        );
        host_interlockingFunctionY_Upper[interfaceLength - 1 - i] = max(
            0.5 * (1.0 + cos(Interface2DConst::PI * (i - 0.0) / (interfaceLength - 0.0))), 
            Interface2DConst::EPS
        );
    }
    for (int i = 0; i < interfaceLength; i++) {
        host_interlockingFunctionYHalf_Lower[i] = max(
            0.5 * (1.0 + cos(Interface2DConst::PI * (i + 0.5 - 0.0) / (interfaceLength - 0.0))), 
            Interface2DConst::EPS
        );
        host_interlockingFunctionYHalf_Upper[interfaceLength - 1 - i] = max(
            0.5 * (1.0 + cos(Interface2DConst::PI * (i + 0.5 - 0.0) / (interfaceLength - 0.0))), 
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
        windowSizeForConvolution, 
        nx_Interface, ny_Interface
    );
    InterfaceNoiseRemover2D interfaceNoiseRemover2D_Upper(
        indexOfInterfaceStartInMHD_Upper, 
        indexOfInterfaceStartInPIC_Upper, 
        interfaceLength, 
        windowSizeForConvolution, 
        nx_Interface, ny_Interface
    );
    Interface2D interface2D_Lower(
        indexOfInterfaceStartInMHD_Lower, 
        indexOfInterfaceStartInPIC_Lower, 
        interfaceLength, 
        host_interlockingFunctionY_Lower, 
        host_interlockingFunctionYHalf_Lower, 
        interfaceNoiseRemover2D_Lower, 
        interfaceNoiseRemover2D_Upper
    );
    Interface2D interface2D_Upper(
        indexOfInterfaceStartInMHD_Upper, 
        indexOfInterfaceStartInPIC_Upper, 
        interfaceLength, 
        host_interlockingFunctionY_Upper, 
        host_interlockingFunctionYHalf_Upper,
        interfaceNoiseRemover2D_Lower,  
        interfaceNoiseRemover2D_Upper
    );
    //BoundaryPIC boundaryPIC;
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


    const int totalSubstep = int(round(sqrt(PIC2DConst::mRatio_PIC)));
    for (int step = 0; step < IdealMHD2DConst::totalStep_MHD + 1; step++) {
        if (step % 10 == 0) {
            std::cout << std::to_string(step) << " step done : total time is "
                      << std::setprecision(4) << step * totalSubstep * PIC2DConst::dt_PIC * PIC2DConst::omegaPe_PIC
                      << " [omega_pe * t]"
                      << " : exist number of particles is " << PIC2DConst::existNumIon_PIC + PIC2DConst::existNumElectron_PIC
                      << std::endl;
        }

        if (step % recordStep == 0) {
            logfile << std::to_string(step) << " step done : total time is "
                    << std::setprecision(4) << step * totalSubstep * PIC2DConst::dt_PIC * PIC2DConst::omegaPe_PIC
                    << " [omega_pe * t]"
                    << std::endl;
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


        // STEP1 : MHD - predictor
        
        idealMHD2D_Lower.calculateDt();
        double dt_Lower_MHD = IdealMHD2DConst::dt_MHD;
        idealMHD2D_Upper.calculateDt();
        double dt_Upper_MHD = IdealMHD2DConst::dt_MHD;
        double dtCommon = min(min(dt_Lower_MHD / totalSubstep, dt_Upper_MHD / totalSubstep), min(0.7 * PIC2DConst::c_PIC, 0.1 * 1.0 / PIC2DConst::omegaPe_PIC));
        PIC2DConst::dt_PIC = dtCommon;
        IdealMHD2DConst::dt_MHD = totalSubstep * dtCommon;

        idealMHD2D_Lower.setPastU();
        idealMHD2D_Upper.setPastU();
        thrust::device_vector<ConservationParameter>& UPast_Lower = idealMHD2D_Lower.getUPastRef();
        thrust::device_vector<ConservationParameter>& UPast_Upper = idealMHD2D_Upper.getUPastRef();

        idealMHD2D_Lower.oneStepRK2PeriodicXSymmetricY_predictor();
        idealMHD2D_Upper.oneStepRK2PeriodicXSymmetricY_predictor();

        thrust::device_vector<ConservationParameter>& UNext_Lower = idealMHD2D_Lower.getURef();
        thrust::device_vector<ConservationParameter>& UNext_Upper = idealMHD2D_Upper.getURef();


        // STEP2 : PIC

        interface2D_Lower.resetTimeAveParameters();
        interface2D_Upper.resetTimeAveParameters();
        for (int substep = 1; substep <= totalSubstep; substep++) {
            pIC2D.oneStepPeriodicXFreeY(
                UPast_Lower, UPast_Upper, 
                UNext_Lower, UNext_Upper, 
                interface2D_Lower, interface2D_Upper, 
                interfaceNoiseRemover2D_Lower, interfaceNoiseRemover2D_Upper, 
                step, substep, totalSubstep
            );

            thrust::device_vector<MagneticField>& B = pIC2D.getBRef();
            thrust::device_vector<Particle>& particlesIon = pIC2D.getParticlesIonRef();
            thrust::device_vector<Particle>& particlesElectron = pIC2D.getParticlesElectronRef();

            interface2D_Lower.sumUpTimeAveParameters(B, particlesIon, particlesElectron);
            interface2D_Upper.sumUpTimeAveParameters(B, particlesIon, particlesElectron);
        }

        interface2D_Lower.calculateTimeAveParameters(totalSubstep);
        interface2D_Upper.calculateTimeAveParameters(totalSubstep);


        // STEP3 : MHD - corrector

        interface2D_Lower.sendPICtoMHD(UPast_Lower, UNext_Lower);
        interface2D_Upper.sendPICtoMHD(UPast_Upper, UNext_Upper);
        thrust::device_vector<ConservationParameter>& UHalf_Lower = interface2D_Lower.getUHalfRef();
        thrust::device_vector<ConservationParameter>& UHalf_Upper = interface2D_Upper.getUHalfRef();
        boundaryMHD.periodicBoundaryX2nd(UHalf_Lower);
        boundaryMHD.symmetricBoundaryY2nd(UHalf_Lower);
        boundaryMHD.periodicBoundaryX2nd(UHalf_Upper);
        boundaryMHD.symmetricBoundaryY2nd(UHalf_Upper);

        idealMHD2D_Lower.oneStepRK2PeriodicXSymmetricY_corrector(UHalf_Lower);
        idealMHD2D_Upper.oneStepRK2PeriodicXSymmetricY_corrector(UHalf_Upper);

        U_Lower = idealMHD2D_Lower.getURef();
        U_Upper = idealMHD2D_Upper.getURef();
        interfaceNoiseRemover2D_Lower.convolveU_lower(U_Lower);
        interfaceNoiseRemover2D_Upper.convolveU_upper(U_Upper);


        if (idealMHD2D_Lower.checkCalculationIsCrashed() || idealMHD2D_Upper.checkCalculationIsCrashed()) {
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


