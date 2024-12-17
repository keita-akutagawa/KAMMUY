#include "main_current_sheet_restart_const.hpp"


void IdealMHD2D::initializeU()
{
}

void PIC2D::initialize()
{
}


int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0, procs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    //int d2[2] = {};
    //MPI_Dims_create(procs, 2, d2);
    int gridX = procs;
    int gridY = 1;

    PIC2DMPI::MPIInfo mPIInfoPIC;
    PIC2DMPI::setupInfo(mPIInfoPIC, buffer, gridX, gridY);
    IdealMHD2DMPI::MPIInfo mPIInfoMHD;
    IdealMHD2DMPI::setupInfo(mPIInfoMHD, buffer, gridX, gridY);
    Interface2DMPI::MPIInfo mPIInfoInterface; 
    Interface2DMPI::setupInfo(mPIInfoInterface, buffer, gridX, gridY); 

    if (mPIInfoPIC.rank == 0) {
        std::cout   << mPIInfoPIC.gridX << "," << mPIInfoPIC.gridY << std::endl;
        mpifile_PIC << mPIInfoPIC.gridX << "," << mPIInfoPIC.gridY << std::endl;
    }
    if (mPIInfoMHD.rank == 0) {
        std::cout   << mPIInfoMHD.gridX << "," << mPIInfoMHD.gridY << std::endl;
        mpifile_MHD << mPIInfoMHD.gridX << "," << mPIInfoMHD.gridY << std::endl;
    }
    if (mPIInfoInterface.rank == 0) {
        std::cout   << mPIInfoInterface.gridX << "," << mPIInfoInterface.gridY << std::endl;
        mpifile_Interface << mPIInfoInterface.gridX << "," << mPIInfoInterface.gridY << std::endl;
    }

    cudaSetDevice(mPIInfoPIC.rank);

    PIC2DConst::initializeDeviceConstants();
    IdealMHD2DConst::initializeDeviceConstants();
    Interface2DConst::initializeDeviceConstants();
    cudaMemcpyToSymbol(device_sheatThickness, &sheatThickness, sizeof(float));
    cudaMemcpyToSymbol(device_betaUpstream, &betaUpstream, sizeof(float));
    cudaMemcpyToSymbol(device_triggerRatio, &triggerRatio, sizeof(float));

    mPIInfoPIC.existNumIonPerProcs      = static_cast<unsigned long long>(PIC2DConst::totalNumIon / mPIInfoPIC.procs);
    mPIInfoPIC.existNumElectronPerProcs = static_cast<unsigned long long>(PIC2DConst::totalNumElectron / mPIInfoPIC.procs);
    mPIInfoPIC.totalNumIonPerProcs = mPIInfoPIC.existNumIonPerProcs
                                   + PIC2DConst::numberDensityIon * (mPIInfoPIC.localSizeX + mPIInfoPIC.localSizeY) * (2 * mPIInfoPIC.buffer)
                                   + Interface2DConst::reloadParticlesTotalNum;
    mPIInfoPIC.totalNumElectronPerProcs = mPIInfoPIC.existNumElectronPerProcs
                                        + PIC2DConst::numberDensityElectron * (mPIInfoPIC.localSizeX + mPIInfoPIC.localSizeY) * (2 * mPIInfoPIC.buffer)
                                        + Interface2DConst::reloadParticlesTotalNum;

    mPIInfoPIC.xminForProcs = PIC2DConst::xmin + (PIC2DConst::xmax - PIC2DConst::xmin) / mPIInfoPIC.gridX * mPIInfoPIC.localGridX;
    mPIInfoPIC.xmaxForProcs = PIC2DConst::xmin + (PIC2DConst::xmax - PIC2DConst::xmin) / mPIInfoPIC.gridX * (mPIInfoPIC.localGridX + 1);
    mPIInfoPIC.yminForProcs = PIC2DConst::ymin + (PIC2DConst::ymax - PIC2DConst::ymin) / mPIInfoPIC.gridY * mPIInfoPIC.localGridY;
    mPIInfoPIC.ymaxForProcs = PIC2DConst::ymin + (PIC2DConst::ymax - PIC2DConst::ymin) / mPIInfoPIC.gridY * (mPIInfoPIC.localGridY + 1);

    for (int j = 0; j < Interface2DConst::ny; j++) {
        host_interlockingFunctionY_lower[j] = max(
            0.5 * (1.0 + cos(Interface2DConst::PI * (j - 0.0) / (Interface2DConst::ny - 0.0))), 
            Interface2DConst::EPS
        );
        host_interlockingFunctionY_upper[j] = max(
            0.5 * (1.0 - cos(Interface2DConst::PI * (j - 0.0) / (Interface2DConst::ny - 0.0))), 
            Interface2DConst::EPS
        );
    }
    for (int j = 0; j < Interface2DConst::ny; j++) {
        host_interlockingFunctionYHalf_lower[j] = max(
            0.5 * (1.0 + cos(Interface2DConst::PI * (j + 0.5 - 0.0) / (Interface2DConst::ny - 0.0))), 
            Interface2DConst::EPS
        );
        host_interlockingFunctionYHalf_upper[j] = max(
            0.5 * (1.0 - cos(Interface2DConst::PI * (j + 0.5 - 0.0) / (Interface2DConst::ny - 0.0))), 
            Interface2DConst::EPS
        );
    }

    IdealMHD2D idealMHD2D_lower(mPIInfoMHD);
    IdealMHD2D idealMHD2D_upper(mPIInfoMHD);
    PIC2D pIC2D(mPIInfoPIC); 
    InterfaceNoiseRemover2D interfaceNoiseRemover2D_lower( 
        mPIInfoMHD, mPIInfoPIC, 
        indexOfConvolutionStartInMHD_lowerInterface, 
        indexOfConvolutionStartInPIC_lowerInterface, 
        convolutionSizeX, convolutionSizeY 
    );
    InterfaceNoiseRemover2D interfaceNoiseRemover2D_upper( 
        mPIInfoMHD, mPIInfoPIC, 
        indexOfConvolutionStartInMHD_upperInterface, 
        indexOfConvolutionStartInPIC_upperInterface, 
        convolutionSizeX, convolutionSizeY 
    );
    Interface2D interface2D_lower(
        mPIInfoMHD, mPIInfoPIC, mPIInfoInterface, 
        indexOfInterfaceStartInMHD_lower, 
        indexOfInterfaceStartInPIC_lower, 
        host_interlockingFunctionY_lower, 
        host_interlockingFunctionYHalf_lower, 
        interfaceNoiseRemover2D_lower
    );
    Interface2D interface2D_upper(
        mPIInfoMHD, mPIInfoPIC, mPIInfoInterface, 
        indexOfInterfaceStartInMHD_upper, 
        indexOfInterfaceStartInPIC_upper, 
        host_interlockingFunctionY_upper, 
        host_interlockingFunctionYHalf_upper,
        interfaceNoiseRemover2D_upper
    );
    BoundaryMHD boundaryMHD(mPIInfoMHD);


    RestartMHD restartMHD(mPIInfoMHD); 
    RestartPIC restartPIC(mPIInfoPIC); 

    thrust::host_vector  <ConservationParameter>& hU_lower = idealMHD2D_lower.getHostURef();
    thrust::device_vector<ConservationParameter>& U_lower  = idealMHD2D_lower.getURef();
    thrust::host_vector  <ConservationParameter>& hU_upper = idealMHD2D_upper.getHostURef();
    thrust::device_vector<ConservationParameter>& U_upper  = idealMHD2D_upper.getURef();
    restartMHD.loadU(
        hU_lower, 
        U_lower, 
        directoryName, filenameWithoutStep + "_U_lower", restartStep
    ); 
    restartMHD.loadU(
        hU_upper, 
        U_upper, 
        directoryName, filenameWithoutStep + "_U_upper", restartStep
    ); 
    boundaryMHD.periodicBoundaryX2nd_U(U_lower);
    boundaryMHD.symmetricBoundaryY2nd_U(U_lower);
    boundaryMHD.periodicBoundaryX2nd_U(U_upper);
    boundaryMHD.symmetricBoundaryY2nd_U(U_upper);

    thrust::host_vector  <MagneticField>& host_B = pIC2D.getHostBRef();
    thrust::device_vector<MagneticField>& B      = pIC2D.getBRef();
    thrust::host_vector  <ElectricField>& host_E = pIC2D.getHostERef(); 
    thrust::device_vector<ElectricField>& E      = pIC2D.getERef();
    thrust::host_vector<Particle>&   host_particlesIon      = pIC2D.getHostParticlesIonRef();
    thrust::device_vector<Particle>& particlesIon           = pIC2D.getParticlesIonRef();
    thrust::host_vector<Particle>&   host_particlesElectron = pIC2D.getHostParticlesElectronRef();
    thrust::device_vector<Particle>& particlesElectron      = pIC2D.getParticlesElectronRef();
    restartPIC.loadFields(
        host_B, host_E, 
        B, E, 
        directoryName, filenameWithoutStep, restartStep
    ); 
    restartPIC.loadParticles(
        host_particlesIon, host_particlesElectron, 
        particlesIon, particlesElectron, 
        directoryName, filenameWithoutStep, restartStep
    );
    boundaryPIC.periodicBoundaryB_x(B);
    boundaryPIC.freeBoundaryB_y(B);
    boundaryPIC.periodicBoundaryE_x(E);
    boundaryPIC.freeBoundaryE_y(E);

    const int totalSubstep = int(round(sqrt(PIC2DConst::mRatio)));
    for (int step = restartStep + 1; step < IdealMHD2DConst::totalStep + 1; step++) {
        MPI_Barrier(MPI_COMM_WORLD);

        if (mPIInfoPIC.rank == 0) {
            if (step % 10 == 0) {
                std::cout << std::to_string(step) << " step done : total time is "
                        << std::setprecision(4) << step * totalSubstep * PIC2DConst::dt * PIC2DConst::omegaPe
                        << " [omega_pe * t]"
                        << std::endl;
            }
        }

        if (step % recordStep == 0) {
            logfile << std::setprecision(6) << PIC2DConst::totalTime << std::endl;
            pIC2D.saveFields(
                directoryName, filenameWithoutStep, step
            );
            pIC2D.saveZerothMoments(
                directoryName, filenameWithoutStep, step
            );
            pIC2D.saveFirstMoments(
                directoryName, filenameWithoutStep, step
            );
            idealMHD2D_lower.save(
                directoryName, filenameWithoutStep + "_U_lower", step
            );
            idealMHD2D_upper.save(
                directoryName, filenameWithoutStep + "_U_upper", step
            );
        }
        if (isParticleRecord && step % particleRecordStep == 0) {
            pIC2D.saveParticle(
                directoryName, filenameWithoutStep, step
            );
        }


        // STEP1 : MHD - predictor
        
        idealMHD2D_lower.calculateDt();
        double dt_lower = IdealMHD2DConst::dt;
        idealMHD2D_upper.calculateDt();
        double dt_upper = IdealMHD2DConst::dt;
        double dtCommon = min(min(dt_lower / totalSubstep, dt_upper / totalSubstep), min(0.7 * PIC2DConst::c, 0.1 * 1.0 / PIC2DConst::omegaPe));
        PIC2DConst::dt = dtCommon;
        IdealMHD2DConst::dt = totalSubstep * dtCommon;

        idealMHD2D_lower.setPastU();
        idealMHD2D_upper.setPastU();
        thrust::device_vector<ConservationParameter>& UPast_lower = idealMHD2D_lower.getUPastRef();
        thrust::device_vector<ConservationParameter>& UPast_upper = idealMHD2D_upper.getUPastRef();

        idealMHD2D_lower.oneStepRK2_periodicXSymmetricY_predictor();
        idealMHD2D_upper.oneStepRK2_periodicXSymmetricY_predictor();

        thrust::device_vector<ConservationParameter>& UNext_lower = idealMHD2D_lower.getURef();
        thrust::device_vector<ConservationParameter>& UNext_upper = idealMHD2D_upper.getURef();


        // STEP2 : PIC

        interface2D_lower.resetTimeAveParameters();
        interface2D_upper.resetTimeAveParameters();

        for (int substep = 1; substep <= totalSubstep; substep++) {
            pIC2D.oneStep_periodicXFreeY(
                UPast_lower, UPast_upper, 
                UNext_lower, UNext_upper, 
                interface2D_lower, interface2D_upper, 
                interfaceNoiseRemover2D_lower, interfaceNoiseRemover2D_upper, 
                step, substep, totalSubstep
            );

            thrust::device_vector<MagneticField>& B = pIC2D.getBRef();
            thrust::device_vector<Particle>& particlesIon = pIC2D.getParticlesIonRef();
            thrust::device_vector<Particle>& particlesElectron = pIC2D.getParticlesElectronRef();

            interface2D_lower.sumUpTimeAveParameters(B, particlesIon, particlesElectron);
            interface2D_upper.sumUpTimeAveParameters(B, particlesIon, particlesElectron);
        }

        interface2D_lower.calculateTimeAveParameters(totalSubstep);
        interface2D_upper.calculateTimeAveParameters(totalSubstep);


        // STEP3 : MHD - corrector
        
        interface2D_lower.sendPICtoMHD(UPast_lower, UNext_lower);
        interface2D_upper.sendPICtoMHD(UPast_upper, UNext_upper);
        thrust::device_vector<ConservationParameter>& UHalf_lower = interface2D_lower.getUHalfRef();
        thrust::device_vector<ConservationParameter>& UHalf_upper = interface2D_upper.getUHalfRef();

        IdealMHD2DMPI::sendrecv_U(UHalf_lower, mPIInfoMHD);
        boundaryMHD.periodicBoundaryX2nd_U(UHalf_lower);
        boundaryMHD.symmetricBoundaryY2nd_U(UHalf_lower);
        IdealMHD2DMPI::sendrecv_U(UHalf_upper, mPIInfoMHD);
        boundaryMHD.periodicBoundaryX2nd_U(UHalf_upper);
        boundaryMHD.symmetricBoundaryY2nd_U(UHalf_upper);

        idealMHD2D_lower.oneStepRK2_periodicXSymmetricY_corrector(UHalf_lower);
        idealMHD2D_upper.oneStepRK2_periodicXSymmetricY_corrector(UHalf_upper);

        U_lower = idealMHD2D_lower.getURef();
        U_upper = idealMHD2D_upper.getURef();
        for (int count = 0; count < Interface2DConst::convolutionCount; count++) {
            interfaceNoiseRemover2D_lower.convolveU(U_lower);
            interfaceNoiseRemover2D_upper.convolveU(U_upper);

            IdealMHD2DMPI::sendrecv_U_x(U_lower, mPIInfoMHD);
            boundaryMHD.periodicBoundaryX2nd_U(U_lower);
            boundaryMHD.symmetricBoundaryY2nd_U(U_lower);
            IdealMHD2DMPI::sendrecv_U_x(U_upper, mPIInfoMHD);
            boundaryMHD.periodicBoundaryX2nd_U(U_upper);
            boundaryMHD.symmetricBoundaryY2nd_U(U_upper);
        }

        //when crashed 
        if (idealMHD2D_lower.checkCalculationIsCrashed() || idealMHD2D_upper.checkCalculationIsCrashed()) {
            logfile << std::setprecision(6) << PIC2DConst::totalTime << std::endl;
            pIC2D.saveFields(
                directoryName, filenameWithoutStep, step
            );
            pIC2D.saveZerothMoments(
                directoryName, filenameWithoutStep, step
            );
            pIC2D.saveFirstMoments(
                directoryName, filenameWithoutStep, step
            );
            idealMHD2D_lower.save(
                directoryName, filenameWithoutStep + "_U_lower", step
            );
            idealMHD2D_upper.save(
                directoryName, filenameWithoutStep + "_U_upper", step
            );
            std::cout << "Calculation stopped! : " << step << " steps" << std::endl;
            break;
        }

        if (mPIInfoMHD.rank == 0) {
            IdealMHD2DConst::totalTime += IdealMHD2DConst::dt;
        }   
    }

    MPI_Finalize();

    if (mPIInfoMHD.rank == 0) {
        std::cout << "program was completed!" << std::endl;
    }

    return 0;
}


