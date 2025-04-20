#include "main_shock_tube_const.hpp"

__global__ void initializeU_kernel(
    ConservationParameter* U, 
    double rhoLeft, double uLeft, double vLeft, double wLeft, 
    double bXLeft, double bYLeft, double bZLeft, double pLeft, 
    double rhoRight, double uRight, double vRight, double wRight, 
    double bXRight, double bYRight, double bZRight, double pRight, 
    IdealMHD2DMPI::MPIInfo* device_mPIInfo
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < IdealMHD2DConst::device_nx && j < IdealMHD2DConst::device_ny) {
        IdealMHD2DMPI::MPIInfo mPIInfo = *device_mPIInfo;

        if (mPIInfo.isInside(i)) {
            int index = mPIInfo.globalToLocal(i, j);

            double rho, u, v, w, bX, bY, bZ, e, p;

            if (j < IdealMHD2DConst::device_ny / 2) {
                rho = rhoLeft;
                u   = uLeft;
                v   = vLeft;
                w   = wLeft;
                bX  = bXLeft;
                bY  = bYLeft;
                bZ  = bZLeft;
                p   = pLeft; 
                e   = p / (IdealMHD2DConst::device_gamma - 1.0)
                    + 0.5 * rho * (u * u + v * v + w * w)
                    + 0.5 * (bX * bX + bY * bY + bZ * bZ);
            } else {
                rho = rhoRight;
                u   = uRight;
                v   = vRight;
                w   = wRight;
                bX  = bXRight;
                bY  = bYRight;
                bZ  = bZRight;
                p   = pRight; 
                e   = p / (IdealMHD2DConst::device_gamma - 1.0)
                    + 0.5 * rho * (u * u + v * v + w * w)
                    + 0.5 * (bX * bX + bY * bY + bZ * bZ);
            }
            
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
        rho0Left, bulkVLeft[0], bulkVLeft[1], bulkVLeft[2], 
        bLeft[0], bLeft[1], bLeft[2], p0Left, 
        rho0Right, bulkVRight[0], bulkVRight[1], bulkVRight[2], 
        bRight[0], bRight[1], bRight[2], p0Right, 
        device_mPIInfo
    );
    cudaDeviceSynchronize();

    MPI_Barrier(MPI_COMM_WORLD);

    boundaryMHD.periodicBoundaryX2nd_U(U);

    MPI_Barrier(MPI_COMM_WORLD);
}


__global__ void initializePICField_kernel(
    ElectricField* E, MagneticField* B, 
    float uLeft, float vLeft, float wLeft, 
    float bXLeft, float bYLeft, float bZLeft, 
    float uRight, float vRight, float wRight, 
    float bXRight, float bYRight, float bZRight, 
    PIC2DMPI::MPIInfo* device_mPIInfo
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < PIC2DConst::device_nx && j < PIC2DConst::device_ny) {
        PIC2DMPI::MPIInfo mPIInfo = *device_mPIInfo;

        if (mPIInfo.isInside(i)) {
            int index = mPIInfo.globalToLocal(i, j);
            float u, v, w, bX, bY, bZ, eX, eY, eZ;

            if (j < PIC2DConst::device_ny / 2) {
                bX = bXLeft; 
                bY = bYLeft; 
                bZ = bZLeft; 
                u = uLeft; 
                v = vLeft; 
                w = wLeft; 
                eX = -(v * bZ - w * bY);
                eY = -(w * bX - u * bZ);
                eZ = -(u * bY - v * bX);
            } else {
                bX = bXRight; 
                bY = bYRight; 
                bZ = bZRight; 
                u = uRight; 
                v = vRight; 
                w = wRight; 
                eX = -(v * bZ - w * bY);
                eY = -(w * bX - u * bZ);
                eZ = -(u * bY - v * bX);
            }

            E[index].eX = eX;
            E[index].eY = eY;
            E[index].eZ = eZ;
            B[index].bX = bX;
            B[index].bY = bY; 
            B[index].bZ = bZ;
        }
    }
}

void PIC2D::initialize()
{
    unsigned long long countIon = 0, countElectron = 0;
    for (int i = 0; i < mPIInfo.localNx; i++) {
        for (int j = 0; j < PIC2DConst::ny; j++) {
            float xminLocal, xmaxLocal, yminLocal, ymaxLocal;
            int ni, ne; 
            float bulkVxLocal, bulkVyLocal, bulkVzLocal, vThIonLocal, vThElectronLocal;

            xminLocal = i * PIC2DConst::dx + mPIInfo.xminForProcs;
            xmaxLocal = (i + 1) * PIC2DConst::dx + mPIInfo.xminForProcs;
            yminLocal = j * PIC2DConst::dy + PIC2DConst::ymin;
            ymaxLocal = (j + 1) * PIC2DConst::dy + PIC2DConst::ymin;

            if (j < PIC2DConst::ny / 2) {
                ni = numberDensityIonLeft; 
                ne = numberDensityElectronLeft; 
                bulkVxLocal = bulkVLeft[0]; 
                bulkVyLocal = bulkVLeft[1]; 
                bulkVzLocal = bulkVLeft[2]; 
                vThIonLocal = vThIonLeft; 
                vThElectronLocal = vThElectronLeft; 
            } else {
                ni = numberDensityIonRight; 
                ne = numberDensityElectronRight; 
                bulkVxLocal = bulkVRight[0]; 
                bulkVyLocal = bulkVRight[1]; 
                bulkVzLocal = bulkVRight[2]; 
                vThIonLocal = vThIonRight; 
                vThElectronLocal = vThElectronRight; 
            }
            
            initializeParticle.uniformForPosition_xy_maxwellDistributionForVelocity_eachCell(
                xminLocal, xmaxLocal, yminLocal, ymaxLocal, 
                bulkVxLocal, bulkVyLocal, bulkVzLocal,  
                vThIonLocal, vThIonLocal, vThIonLocal, 
                countIon, countIon + ni, 
                j + i * PIC2DConst::ny + mPIInfo.rank * mPIInfo.localNx * PIC2DConst::ny, 
                particlesIon
            );
            initializeParticle.uniformForPosition_xy_maxwellDistributionForVelocity_eachCell(
                xminLocal, xmaxLocal, yminLocal, ymaxLocal, 
                bulkVxLocal, bulkVyLocal, bulkVzLocal,  
                vThElectronLocal, vThElectronLocal, vThElectronLocal, 
                countElectron, countElectron + ni, 
                j + i * PIC2DConst::ny + mPIInfo.localNx * PIC2DConst::ny + mPIInfo.rank * mPIInfo.localNx * PIC2DConst::ny, 
                particlesElectron
            );

            countIon      += ni;
            countElectron += ne;
        }
    }

    mPIInfo.existNumIonPerProcs = countIon; 
    mPIInfo.existNumElectronPerProcs = countElectron; 


    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((PIC2DConst::nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (PIC2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    initializePICField_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(E.data()), thrust::raw_pointer_cast(B.data()), 
        bulkVLeft[0], bulkVLeft[1], bulkVLeft[2], 
        bLeft[0], bLeft[1], bLeft[2], 
        bulkVRight[0], bulkVRight[1], bulkVRight[2], 
        bRight[0], bRight[1], bRight[2],  
        device_mPIInfo
    );
    cudaDeviceSynchronize();

    MPI_Barrier(MPI_COMM_WORLD);

    boundaryPIC.periodicBoundaryB_x(B);
    boundaryPIC.freeBoundaryB_y(B);
    boundaryPIC.periodicBoundaryE_x(E);
    boundaryPIC.freeBoundaryE_y(E);
    boundaryPIC.periodicBoundaryParticle_x(particlesIon, particlesElectron);
    boundaryPIC.freeBoundaryParticle_y(particlesIon, particlesElectron);
    
    MPI_Barrier(MPI_COMM_WORLD);
}


int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0, procs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    PIC2DMPI::MPIInfo mPIInfoPIC;
    int mpiBufNumParticles = 1000000; 
    PIC2DMPI::setupInfo(mPIInfoPIC, buffer, mpiBufNumParticles);
    IdealMHD2DMPI::MPIInfo mPIInfoMHD;
    IdealMHD2DMPI::setupInfo(mPIInfoMHD, buffer);
    Interface2DMPI::MPIInfo mPIInfoInterface; 
    Interface2DMPI::setupInfo(mPIInfoInterface, buffer); 

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
    mPIInfoPIC.totalNumIonPerProcs = static_cast<unsigned long long>(mPIInfoPIC.existNumIonPerProcs * 2.0);
    mPIInfoPIC.totalNumElectronPerProcs = static_cast<unsigned long long>(mPIInfoPIC.existNumElectronPerProcs * 2.0);

    mPIInfoPIC.xminForProcs = PIC2DConst::xmin
                            + (PIC2DConst::xmax - PIC2DConst::xmin) / mPIInfoPIC.gridX
                            * mPIInfoPIC.localGridX;
    mPIInfoPIC.xmaxForProcs = PIC2DConst::xmin
                            + (PIC2DConst::xmax - PIC2DConst::xmin) / mPIInfoPIC.gridX
                            * (mPIInfoPIC.localGridX + 1);

    int bufferForInterlocking = 5;  
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

    const int totalSubstep = int(round(sqrt(PIC2DConst::mRatio)) * Interface2DConst::gridSizeRatio);
    for (int step = 0; step < IdealMHD2DConst::totalStep + 1; step++) {
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
        cudaMemcpyToSymbol(PIC2DConst::device_dt, &PIC2DConst::dt, sizeof(float));
        cudaMemcpyToSymbol(IdealMHD2DConst::device_dt, &IdealMHD2DConst::dt, sizeof(double));

        // STEP1 : MHD step

        idealMHD2D.setPastU();
        thrust::device_vector<ConservationParameter>& UPast = idealMHD2D.getUPastRef();

        idealMHD2D.oneStepRK2_periodicXY_predictor();

        thrust::device_vector<ConservationParameter>& UNext = idealMHD2D.getURef();

        // STEP2 : PIC step & send MHD to PIC

        interface2D.resetTimeAveragedPICParameters();

        int sumUpCount; 
        sumUpCount = 0; 
        pIC2D.calculateFullMoments();
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
        sumUpCount += 1; 
        for (int substep = 1; substep <= totalSubstep; substep++) {

            float mixingRatio = 1.0 - substep / totalSubstep;
            thrust::device_vector<ConservationParameter>& USub = interface2D.calculateAndGetSubU(UPast, UNext, mixingRatio);
            
            unsigned long long seedForReload; 
            seedForReload = substep + step * totalSubstep;
            pIC2D.oneStep_periodicXFreeY(
                interface2D, 
                USub, 
                seedForReload
            );

            interface2D.sumUpTimeAveragedPICParameters(
                B, 
                zerothMomentIon, zerothMomentElectron, 
                firstMomentIon, firstMomentElectron, 
                secondMomentIon, secondMomentElectron
            );
            sumUpCount += 1; 
        }

        interface2D.calculateTimeAveragedPICParameters(sumUpCount); 

        interface2D.setParametersForPICtoMHD();

        // STEP3 : send PIC to MHD

        interface2D.calculateUHalf(UPast, UNext); 
        thrust::device_vector<ConservationParameter>& UHalf = interface2D.getUHalfRef();

        interface2D.sendPICtoMHD(UHalf);
        boundaryMHD.periodicBoundaryX2nd_U(UHalf);
        boundaryMHD.symmetricBoundaryY2nd_U(UHalf);

        for (int count = 0; count < Interface2DConst::convolutionCount; count++) {
            interfaceNoiseRemover2D.convolveU(UHalf);

            boundaryMHD.periodicBoundaryX2nd_U(UHalf);
            boundaryMHD.symmetricBoundaryY2nd_U(UHalf);
        }

        projection.correctB(UHalf); 
        boundaryMHD.periodicBoundaryX2nd_U(UHalf);
        boundaryMHD.symmetricBoundaryY2nd_U(UHalf);
        MPI_Barrier(MPI_COMM_WORLD);

        idealMHD2D.oneStepRK2_periodicXSymmetricY_corrector(UHalf);

        thrust::device_vector<ConservationParameter>& U = idealMHD2D.getURef();
        for (int count = 0; count < Interface2DConst::convolutionCount; count++) {
            interfaceNoiseRemover2D.convolveU(U);

            boundaryMHD.periodicBoundaryX2nd_U(U);
            boundaryMHD.symmetricBoundaryY2nd_U(U);
        }

        projection.correctB(U); 
        boundaryMHD.periodicBoundaryX2nd_U(U);
        boundaryMHD.symmetricBoundaryY2nd_U(U);
        MPI_Barrier(MPI_COMM_WORLD);


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



