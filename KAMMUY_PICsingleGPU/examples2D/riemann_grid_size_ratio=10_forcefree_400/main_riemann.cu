#include "main_riemann_const.hpp"


__device__
double getEta(double xPosition, double yPosition)
{
    return 0.0;
}


__global__ void initializeU_kernel(
    ConservationParameter* U, 
    const double sheatThickness, const double triggerRatio, 
    IdealMHD2DMPI::MPIInfo* device_mPIInfo
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < IdealMHD2DConst::device_nx && j < IdealMHD2DConst::device_ny) {

        if (device_mPIInfo->isInside(i)) {
            int index = device_mPIInfo->globalToLocal(i, j);

            double rho, u, v, w, bX, bY, bZ, e, p;
            double x = i * IdealMHD2DConst::device_dx, y = j * IdealMHD2DConst::device_dy; 
            double xCenter = 0.5 * (IdealMHD2DConst::device_xmax - IdealMHD2DConst::device_xmin);
            double yCenter = 0.5 * (IdealMHD2DConst::device_ymax - IdealMHD2DConst::device_ymin);
            
            rho = IdealMHD2DConst::device_rho0;
            u   = 0.0;
            v   = 0.0;
            w   = 0.0;
            bX  = IdealMHD2DConst::device_B0 * tanh((y - yCenter) / sheatThickness);
            bY  = 0.1 * IdealMHD2DConst::device_B0; 
            bZ  = IdealMHD2DConst::device_B0 / cosh((y - yCenter) / sheatThickness);
            p   = IdealMHD2DConst::device_p0;
            e   = p / (IdealMHD2DConst::device_gamma - 1.0)
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
}

void IdealMHD2D::initializeU()
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((IdealMHD2DConst::nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (IdealMHD2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    initializeU_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        sheatThickness, triggerRatio, 
        device_mPIInfo
    );
    cudaDeviceSynchronize();

    MPI_Barrier(MPI_COMM_WORLD);

    boundaryMHD.periodicBoundaryX2nd_U(U);

    MPI_Barrier(MPI_COMM_WORLD);
}


__global__ void initializePICField_kernel(
    ElectricField* E, MagneticField* B, 
    const double sheatThickness, const double triggerRatio
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < PIC2DConst::device_nx && j < PIC2DConst::device_ny) {
        unsigned long long index = j + i * PIC2DConst::device_ny;

        double bX, bY, bZ, eX, eY, eZ;
        double x = i * PIC2DConst::device_dx + PIC2DConst::device_xmin, y = j * PIC2DConst::device_dy + PIC2DConst::device_ymin;
        double xCenter = 0.5 * (PIC2DConst::device_xmax - PIC2DConst::device_xmin);
        double yCenter = 0.5 * (PIC2DConst::device_ymax - PIC2DConst::device_ymin);

        bX = PIC2DConst::device_B0 * tanh((y - yCenter) / sheatThickness);
        bY = 0.1 * PIC2DConst::device_B0; 
        bZ = PIC2DConst::device_B0 / cosh((y - yCenter) / sheatThickness);
        eX = 0.0;
        eY = 0.0;
        eZ = 0.0;

        E[index].eX = eX;
        E[index].eY = eY;
        E[index].eZ = eZ;
        B[index].bX = bX;
        B[index].bY = bY; 
        B[index].bZ = bZ;
    }
}

void PIC2D::initialize()
{
    unsigned long long countIon = 0, countElectron = 0;
    for (int i = 0; i < PIC2DConst::nx; i++) {
        for (int j = 0; j < PIC2DConst::ny; j++) {
            double xminLocal, xmaxLocal, yminLocal, ymaxLocal;
            double y = j * PIC2DConst::dy + PIC2DConst::ymin;
            double yCenter = 0.5 * (PIC2DConst::ymax - PIC2DConst::ymin) + PIC2DConst::ymin;
            
            xminLocal = i * PIC2DConst::dx + PIC2DConst::xmin + PIC2DConst::EPS;
            xmaxLocal = (i + 1) * PIC2DConst::dx + PIC2DConst::xmin - PIC2DConst::EPS;
            yminLocal = j * PIC2DConst::dy + PIC2DConst::ymin + PIC2DConst::EPS;
            ymaxLocal = (j + 1) * PIC2DConst::dy + PIC2DConst::ymin - PIC2DConst::EPS;

            int ni = PIC2DConst::numberDensityIon;
            int ne = PIC2DConst::numberDensityElectron;

            double jX = -PIC2DConst::B0 / sheatThickness * tanh((y - yCenter) / sheatThickness) / cosh((y - yCenter) / sheatThickness);
            double jY = 0.0; 
            double jZ = -PIC2DConst::B0 / sheatThickness / pow(cosh((y - yCenter) / sheatThickness), 2);
            
            double bulkVxIonLocal = 0.0, bulkVyIonLocal = 0.0, bulkVzIonLocal = 0.0; 
            double bulkVxElectronLocal = jX / ne / PIC2DConst::qElectron; 
            double bulkVyElectronLocal = jY / ne / PIC2DConst::qElectron;
            double bulkVzElectronLocal = jZ / ne / PIC2DConst::qElectron; 

            initializeParticle.uniformForPosition_xy_maxwellDistributionForVelocity_eachCell(
                xminLocal, xmaxLocal, yminLocal, ymaxLocal, 
                bulkVxIonLocal, bulkVyIonLocal, bulkVzIonLocal, 
                PIC2DConst::vThIon, PIC2DConst::vThIon, PIC2DConst::vThIon, 
                countIon, countIon + ni, 
                j + i * PIC2DConst::ny + PIC2DConst::nx * PIC2DConst::ny, 
                particlesIon
            ); 
            initializeParticle.uniformForPosition_xy_maxwellDistributionForVelocity_eachCell(
                xminLocal, xmaxLocal, yminLocal, ymaxLocal, 
                bulkVxElectronLocal, bulkVyElectronLocal, bulkVzElectronLocal, 
                PIC2DConst::vThElectron, PIC2DConst::vThElectron, PIC2DConst::vThElectron, 
                countElectron, countElectron + ne, 
                j + i * PIC2DConst::ny + PIC2DConst::nx * PIC2DConst::ny, 
                particlesElectron
            ); 

            countIon += ni; 
            countElectron += ne; 
        }
    }
    PIC2DConst::existNumIon = countIon; 
    PIC2DConst::existNumElectron = countElectron;


    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((PIC2DConst::nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (PIC2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    initializePICField_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(E.data()), thrust::raw_pointer_cast(B.data()), 
        sheatThickness, triggerRatio
    );
    cudaDeviceSynchronize();
}


int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0, procs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    IdealMHD2DMPI::MPIInfo mPIInfoMHD;
    IdealMHD2DMPI::setupInfo(mPIInfoMHD, bufferMHD);

    if (mPIInfoMHD.rank == 0) {
        std::cout   << mPIInfoMHD.gridX << std::endl;
        mpifile_MHD << mPIInfoMHD.gridX << std::endl;
    }

    cudaSetDevice(mPIInfoMHD.rank);

    PIC2DConst::initializeDeviceConstants();
    IdealMHD2DConst::initializeDeviceConstants();
    Interface2DConst::initializeDeviceConstants();

    thrust::host_vector<double> host_interlockingFunctionY(PIC2DConst::nx * PIC2DConst::ny, 0.0);
    int bufferForInterlocking = 0;  
    for (int i = 0; i < PIC2DConst::nx; i++) {
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
    for (int i = 0; i < PIC2DConst::nx; i++) {
        for (int j = PIC2DConst::ny / 2; j < PIC2DConst::ny; j++) {
            host_interlockingFunctionY[j + i * PIC2DConst::ny] = host_interlockingFunctionY[PIC2DConst::ny - 1 - j + i * PIC2DConst::ny];
        }
    }

    IdealMHD2D idealMHD2D(mPIInfoMHD);
    PIC2D pIC2D; 
    InterfaceNoiseRemover2D interfaceNoiseRemover2D(mPIInfoMHD);
    Interface2D interface2D(
        mPIInfoMHD, 
        Interface2DConst::indexOfInterfaceStartInMHD, 
        host_interlockingFunctionY, 
        interfaceNoiseRemover2D
    );
    BoundaryMHD& boundaryMHD = idealMHD2D.getBoundaryMHDRef(); 
    BoundaryPIC& boundaryPIC = pIC2D.getBoundaryPICRef(); 
    Projection& projection = idealMHD2D.getProjectionRef();
    

    if (mPIInfoMHD.rank == 0) {
        size_t free_mem = 0;
        size_t total_mem = 0;
        cudaError_t status = cudaMemGetInfo(&free_mem, &total_mem);

        std::cout << "Free memory: " << free_mem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "Total memory: " << total_mem / (1024 * 1024) << " MB" << std::endl;

        std::cout << "exist number of partices is " << PIC2DConst::existNumIon + PIC2DConst::existNumElectron << std::endl;
        std::cout << "exist number of partices + buffer particles is " << PIC2DConst::totalNumParticles << std::endl;
        
        std::cout << "PIC grid size is " 
                  << PIC2DConst::nx << " X " << PIC2DConst::ny 
                  << std::endl;
        std::cout << "MHD grid size is " 
                  << mPIInfoMHD.localSizeX << " X " << IdealMHD2DConst::ny
                  << std::endl;
    }

    idealMHD2D.initializeU(); 
    pIC2D.initialize();

    const int totalSubstep = int(round(sqrt(PIC2DConst::mRatio)) * Interface2DConst::gridSizeRatio);
    for (int step = 0; step < IdealMHD2DConst::totalStep + 1; step++) {
        MPI_Barrier(MPI_COMM_WORLD);

        if (mPIInfoMHD.rank == 0) {
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



