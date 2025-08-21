#include "main_mrx_reload_const.hpp"


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

    PIC2DMPI::MPIInfo mPIInfoPIC;
    int mpiBufNumParticles = 10000000; 
    PIC2DMPI::setupInfo(mPIInfoPIC, bufferPIC, mpiBufNumParticles);
    IdealMHD2DMPI::MPIInfo mPIInfoMHD;
    IdealMHD2DMPI::setupInfo(mPIInfoMHD, bufferMHD);
    Interface2DMPI::MPIInfo mPIInfoInterface; 
    Interface2DMPI::setupInfo(mPIInfoInterface); 

    if (mPIInfoPIC.rank == 0) {
        std::cout   << mPIInfoPIC.gridX << std::endl;
        mpifile_PIC << mPIInfoPIC.gridX << std::endl;
    }
    if (mPIInfoMHD.rank == 0) {
        std::cout   << mPIInfoMHD.gridX << std::endl;
        mpifile_MHD << mPIInfoMHD.gridX << std::endl;
    }
    if (mPIInfoInterface.rank == 0) {
        std::cout   << mPIInfoInterface.gridX << std::endl;
        mpifile_Interface << mPIInfoInterface.gridX << std::endl;
    }

    cudaSetDevice(mPIInfoPIC.rank);

    PIC2DConst::initializeDeviceConstants();
    IdealMHD2DConst::initializeDeviceConstants();
    Interface2DConst::initializeDeviceConstants();

    mPIInfoPIC.existNumIonPerProcs      = static_cast<unsigned long long>(PIC2DConst::totalNumIon / mPIInfoPIC.procs);
    mPIInfoPIC.existNumElectronPerProcs = static_cast<unsigned long long>(PIC2DConst::totalNumElectron / mPIInfoPIC.procs);
    mPIInfoPIC.totalNumIonPerProcs      = static_cast<unsigned long long>(mPIInfoPIC.existNumIonPerProcs * 2.0);
    mPIInfoPIC.totalNumElectronPerProcs = static_cast<unsigned long long>(mPIInfoPIC.existNumElectronPerProcs * 2.0);

    mPIInfoPIC.xminForProcs = PIC2DConst::xmin
                            + (PIC2DConst::xmax - PIC2DConst::xmin) / mPIInfoPIC.gridX
                            * mPIInfoPIC.localGridX;
    mPIInfoPIC.xmaxForProcs = PIC2DConst::xmin
                            + (PIC2DConst::xmax - PIC2DConst::xmin) / mPIInfoPIC.gridX
                            * (mPIInfoPIC.localGridX + 1);


    thrust::host_vector<double> host_interlockingFunctionY(mPIInfoPIC.localSizeX * PIC2DConst::ny, 0.0);
    int bufferForInterlocking = 0;  
    for (int i = 0; i < mPIInfoPIC.localSizeX; i++) {
        for (int j = 0; j < PIC2DConst::ny / 2; j++) {
            if (j < bufferForInterlocking) {
                host_interlockingFunctionY[j + i * PIC2DConst::ny] = 1.0;
            } else if (bufferForInterlocking <= j && j < Interface2DConst::deltaForInterlockingFunction + bufferForInterlocking) {
                host_interlockingFunctionY[j + i * PIC2DConst::ny] = 0.5 * (1.0 + cos(Interface2DConst::PI * (j - bufferForInterlocking) / Interface2DConst::deltaForInterlockingFunction));
            } else {
                host_interlockingFunctionY[j + i * PIC2DConst::ny] = 0.0;
            }
        }
    }
    for (int i = 0; i < mPIInfoPIC.localSizeX; i++) {
        for (int j = PIC2DConst::ny / 2; j < PIC2DConst::ny; j++) {
            host_interlockingFunctionY[j + i * PIC2DConst::ny] = host_interlockingFunctionY[PIC2DConst::ny - 1 - j + i * PIC2DConst::ny];
        }
    }

    IdealMHD2D idealMHD2D(mPIInfoMHD);
    PIC2D pIC2D(mPIInfoPIC); 
    InterfaceNoiseRemover2D interfaceNoiseRemover2D( 
        mPIInfoMHD, mPIInfoPIC
    );
    Interface2D interface2D(
        mPIInfoMHD, mPIInfoPIC, mPIInfoInterface, 
        Interface2DConst::indexOfInterfaceStartInMHD, 
        host_interlockingFunctionY, 
        interfaceNoiseRemover2D
    );
    BoundaryMHD& boundaryMHD = idealMHD2D.getBoundaryMHDRef(); 
    BoundaryPIC& boundaryPIC = pIC2D.getBoundaryPICRef(); 
    Projection& projection = idealMHD2D.getProjectionRef();
    

    if (mPIInfoPIC.rank == 0) {
        size_t free_mem = 0;
        size_t total_mem = 0;
        cudaError_t status = cudaMemGetInfo(&free_mem, &total_mem);

        std::cout << "Free memory: " << free_mem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "Total memory: " << total_mem / (1024 * 1024) << " MB" << std::endl;

        std::cout << "exist number of partices is " 
                  << mPIInfoPIC.procs * (mPIInfoPIC.existNumIonPerProcs + mPIInfoPIC.existNumElectronPerProcs) 
                  << std::endl;
        std::cout << "exist number of partices + buffer particles is " 
                  << mPIInfoPIC.procs * (mPIInfoPIC.totalNumIonPerProcs + mPIInfoPIC.totalNumElectronPerProcs) 
                  << std::endl;
        
        std::cout << "PIC grid size is " 
                  << mPIInfoPIC.localSizeX << " X " << PIC2DConst::ny 
                  << std::endl;
        std::cout << "MHD grid size is " 
                  << mPIInfoMHD.localSizeX << " X " << IdealMHD2DConst::ny
                  << std::endl;
    }

    idealMHD2D.initializeU(); 
    pIC2D.initialize();


    std::cout << "Reload started" << std::endl;
    //ここから大幅に異なる
    thrust::host_vector<Particle>& host_particlesIon = pIC2D.getHostParticlesIonRef();
    thrust::host_vector<Particle>& host_particlesElectron = pIC2D.getHostParticlesElectronRef();
    thrust::host_vector<MagneticField>& host_B = pIC2D.getHostBRef();
    thrust::host_vector<ElectricField>& host_E = pIC2D.getHostERef();
    thrust::host_vector<ConservationParameter>& host_U = idealMHD2D.getHostURef();

    Reloader reloader(mPIInfoPIC, mPIInfoMHD);
    reloader.reloadPICData(
        host_particlesIon, host_particlesElectron, host_B, host_E, 
        directoryName, filenameWithoutStep, 
        reloadStep
    );
    reloader.reloadMHDData(
        host_U, 
        directoryName, filenameWithoutStep, 
        reloadStep
    );

    thrust::device_vector<Particle>& particlesIon = pIC2D.getParticlesIonRef();
    thrust::device_vector<Particle>& particlesElectron = pIC2D.getParticlesElectronRef();
    thrust::device_vector<MagneticField>& B = pIC2D.getBRef();
    thrust::device_vector<ElectricField>& E = pIC2D.getERef();
    thrust::device_vector<ConservationParameter>& U = idealMHD2D.getURef();

    particlesIon = host_particlesIon; 
    particlesElectron = host_particlesElectron; 
    B = host_B; 
    E = host_E; 
    U = host_U; 

    const int totalSubstep = int(round(sqrt(PIC2DConst::mRatio)) * Interface2DConst::gridSizeRatio);
    for (int step = reloadStep + 1; step < IdealMHD2DConst::totalStep + 1; step++) {
        MPI_Barrier(MPI_COMM_WORLD);

        if (mPIInfoPIC.rank == 0) {
            if (step % recordStep == 0) {
                std::cout << std::to_string(step) << " step done : total time is "
                        << std::setprecision(4) << step * totalSubstep * PIC2DConst::dt * PIC2DConst::omegaPe
                        << " [omega_pe * t]"
                        << std::endl;
            }
        }

        if (step % recordStep == 0) {
            logfile << std::setprecision(6) << IdealMHD2DConst::totalTime << std::endl;
            pIC2D.saveParticle(
                directoryName, filenameWithoutStep, step
            );
            pIC2D.saveFields(
                directoryName, filenameWithoutStep, step
            );
            pIC2D.saveZerothMoments(
                directoryName, filenameWithoutStep, step
            );
            pIC2D.saveFirstMoments(
                directoryName, filenameWithoutStep, step
            );
            pIC2D.saveSecondMoments(
                directoryName, filenameWithoutStep, step
            );
            idealMHD2D.save(
                directoryName, filenameWithoutStep + "_U", step
            );
        }

        double dtCommon = min(0.7 / PIC2DConst::c, 0.1 * 1.0 / PIC2DConst::omegaPe);
        PIC2DConst::dt = dtCommon;
        IdealMHD2DConst::dt = totalSubstep * dtCommon;
        IdealMHD2DConst::eta = 0.0 * pow(IdealMHD2DConst::dx, 2) / IdealMHD2DConst::dt; 
        IdealMHD2DConst::viscosity = 0.0 * pow(IdealMHD2DConst::dx, 2) / IdealMHD2DConst::dt; 
        cudaMemcpyToSymbol(PIC2DConst::device_dt, &PIC2DConst::dt, sizeof(double));
        cudaMemcpyToSymbol(IdealMHD2DConst::device_dt, &IdealMHD2DConst::dt, sizeof(double));
        cudaMemcpyToSymbol(IdealMHD2DConst::device_eta, &IdealMHD2DConst::eta, sizeof(double));
        cudaMemcpyToSymbol(IdealMHD2DConst::device_viscosity, &IdealMHD2DConst::viscosity, sizeof(double));


        // STEP1 : MHD step

        idealMHD2D.setPastU();
        thrust::device_vector<ConservationParameter>& UPast = idealMHD2D.getUPastRef();

        idealMHD2D.oneStepRK2_periodicXSymmetricY_predictor();

        thrust::device_vector<ConservationParameter>& UNext = idealMHD2D.getURef();

        // STEP2 : PIC step & send MHD to PIC

        interface2D.resetTimeAveragedPICParameters();

        //int sumUpCount = 0;  
        //pIC2D.calculateFullMoments();
        //thrust::device_vector<MagneticField>& B = pIC2D.getBRef();
        //thrust::device_vector<ZerothMoment>& zerothMomentIon = pIC2D.getZerothMomentIonRef(); 
        //thrust::device_vector<ZerothMoment>& zerothMomentElectron = pIC2D.getZerothMomentElectronRef(); 
        //thrust::device_vector<FirstMoment>& firstMomentIon = pIC2D.getFirstMomentIonRef(); 
        //thrust::device_vector<FirstMoment>& firstMomentElectron = pIC2D.getFirstMomentElectronRef(); 
        //thrust::device_vector<SecondMoment>& secondMomentIon = pIC2D.getSecondMomentIonRef(); 
        //thrust::device_vector<SecondMoment>& secondMomentElectron = pIC2D.getSecondMomentElectronRef(); 
        //interface2D.sumUpTimeAveragedPICParameters(
        //    B, 
        //    zerothMomentIon, zerothMomentElectron, 
        //    firstMomentIon, firstMomentElectron, 
        //    secondMomentIon, secondMomentElectron
        //);
        //sumUpCount += 1; 
        for (int substep = 1; substep <= totalSubstep; substep++) {

            double mixingRatio = 1.0 - static_cast<double>(substep) / static_cast<double>(totalSubstep);
            thrust::device_vector<ConservationParameter>& USub = interface2D.calculateAndGetSubU(UPast, UNext, mixingRatio);
            
            unsigned long long seedForReload; 
            seedForReload = substep + step * totalSubstep;
            pIC2D.oneStep_periodicXFreeY(
                interface2D, 
                USub, 
                seedForReload
            );

            //interface2D.sumUpTimeAveragedPICParameters(
            //    B, 
            //    zerothMomentIon, zerothMomentElectron, 
            //    firstMomentIon, firstMomentElectron, 
            //    secondMomentIon, secondMomentElectron
            //);
            //sumUpCount += 1; 
        }

        //interface2D.calculateTimeAveragedPICParameters(sumUpCount); 

        thrust::device_vector<MagneticField>& B = pIC2D.getBRef();
        thrust::device_vector<ZerothMoment>& zerothMomentIon = pIC2D.getZerothMomentIonRef(); 
        thrust::device_vector<ZerothMoment>& zerothMomentElectron = pIC2D.getZerothMomentElectronRef(); 
        thrust::device_vector<FirstMoment>& firstMomentIon = pIC2D.getFirstMomentIonRef(); 
        thrust::device_vector<FirstMoment>& firstMomentElectron = pIC2D.getFirstMomentElectronRef(); 
        thrust::device_vector<SecondMoment>& secondMomentIon = pIC2D.getSecondMomentIonRef(); 
        thrust::device_vector<SecondMoment>& secondMomentElectron = pIC2D.getSecondMomentElectronRef(); 
        interface2D.sumUpTimeAveragedPICParameters(
            B, 
            zerothMomentIon, zerothMomentElectron, 
            firstMomentIon, firstMomentElectron, 
            secondMomentIon, secondMomentElectron
        );

        interface2D.setParametersForPICtoMHD();

        // STEP3 : send PIC to MHD

        //interface2D.calculateUHalf(UPast, UNext); 
        //thrust::device_vector<ConservationParameter>& UHalf = interface2D.getUHalfRef();

        thrust::device_vector<ConservationParameter>& U = idealMHD2D.getURef();

        interface2D.sendPICtoMHD(U);
        boundaryMHD.periodicBoundaryX2nd_U(U);
        boundaryMHD.symmetricBoundaryY2nd_U(U);

        //idealMHD2D.oneStepRK2_periodicXSymmetricY_corrector(UHalf);
        
        if (step % 10 == 0) {
            projection.correctB(U); 
            boundaryMHD.periodicBoundaryX2nd_U(U);
            boundaryMHD.symmetricBoundaryY2nd_U(U);
            
            interfaceNoiseRemover2D.convolveU(U);
            boundaryMHD.periodicBoundaryX2nd_U(U);
            boundaryMHD.symmetricBoundaryY2nd_U(U);
        }

        //when crashed 
        if (idealMHD2D.checkCalculationIsCrashed()) {
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
            idealMHD2D.save(
                directoryName, filenameWithoutStep + "_U", step
            );
            std::cout << "Calculation stopped! : " << step << " steps" << std::endl;
            break;
        }

        if (mPIInfoMHD.rank == 0) {
            IdealMHD2DConst::totalTime += IdealMHD2DConst::dt;
        }   
    }

    if (mPIInfoMHD.rank == 0) {
        std::cout << "program was completed!" << std::endl;
    }

    MPI_Finalize();

    return 0;
}



