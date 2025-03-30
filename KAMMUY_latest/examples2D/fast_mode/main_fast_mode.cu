#include "main_fast_mode_const.hpp"


__global__ void initializeU_kernel(
    ConservationParameter* U, 
    double Cf, double waveAmp, double waveNumber, 
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
            double y = j * IdealMHD2DConst::device_dy;
            
            rho = IdealMHD2DConst::device_rho0 * (1.0 + waveAmp * sin(waveNumber * y));;
            u   = 0.0;
            v   = waveAmp * Cf * sin(waveNumber * y);
            w   = 0.0;
            bX  = 0.0;
            bY  = 0.0;
            bZ  = IdealMHD2DConst::device_B0 * (1.0 + waveAmp * sin(waveNumber * y));
            p   = IdealMHD2DConst::device_p0 * (1.0 + IdealMHD2DConst::device_gamma * waveAmp * sin(waveNumber * y));
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
    double Cf = sqrt(IdealMHD2DConst::B0 * IdealMHD2DConst::B0 / IdealMHD2DConst::rho0 + IdealMHD2DConst::gamma * IdealMHD2DConst::p0 / IdealMHD2DConst::rho0);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((IdealMHD2DConst::nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (IdealMHD2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    initializeU_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        Cf, waveAmp, waveNumber, 
        device_mPIInfo
    );
    cudaDeviceSynchronize();

    MPI_Barrier(MPI_COMM_WORLD);

    boundaryMHD.periodicBoundaryX2nd_U(U);

    MPI_Barrier(MPI_COMM_WORLD);
}


__global__ void initializePICField_kernel(
    ElectricField* E, MagneticField* B, 
    double Cf, double waveAmp, double waveNumber, 
    PIC2DMPI::MPIInfo* device_mPIInfo
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < PIC2DConst::device_nx && j < PIC2DConst::device_ny) {
        PIC2DMPI::MPIInfo mPIInfo = *device_mPIInfo;

        if (mPIInfo.isInside(i)) {
            int index = mPIInfo.globalToLocal(i, j);
            double u, v, w, bX, bY, bZ, eX, eY, eZ;
            double y = j * PIC2DConst::device_dy + 9500 * IdealMHD2DConst::device_dy;

            u   = 0.0;
            v   = waveAmp * Cf * sin(waveNumber * y);
            w   = 0.0;
            bX  = 0.0;
            bY  = 0.0;
            bZ  = PIC2DConst::device_B0 * (1.0 + waveAmp * sin(waveNumber * y));
            eX = -(v * bZ - w * bY);
            eY = -(w * bX - u * bZ);
            eZ = -(u * bY - v * bX);

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
    float Cf = sqrt(IdealMHD2DConst::B0 * IdealMHD2DConst::B0 / IdealMHD2DConst::rho0 + IdealMHD2DConst::gamma * IdealMHD2DConst::p0 / IdealMHD2DConst::rho0);

    unsigned long long countIon = 0, countElectron = 0;
    for (int i = 0; i < mPIInfo.localNx; i++) {
        for (int j = 0; j < PIC2DConst::ny; j++) {
            float xminLocal, xmaxLocal, yminLocal, ymaxLocal;
            float rho, bulkVx, bulkVy, bulkVz;
            int ni, ne; 
            float y = j * PIC2DConst::dy + 2450 * IdealMHD2DConst::dy;

            xminLocal = i * PIC2DConst::dx + mPIInfo.xminForProcs;
            xmaxLocal = (i + 1) * PIC2DConst::dx + mPIInfo.xminForProcs;
            yminLocal = j * PIC2DConst::dy + PIC2DConst::ymin;
            ymaxLocal = (j + 1) * PIC2DConst::dy + PIC2DConst::ymin;
            
            float rho0 = PIC2DConst::mIon * PIC2DConst::numberDensityIon + PIC2DConst::mElectron * PIC2DConst::numberDensityElectron;
            rho = rho0 * (1.0f + waveAmp * sin(waveNumber * y));
            ni  = round(rho / (PIC2DConst::mIon + PIC2DConst::mElectron));
            ne  = ni; 
            bulkVx = 0.0f;
            bulkVy = waveAmp * Cf * sin(waveNumber * y);
            bulkVz = 0.0f;

            initializeParticle.uniformForPosition_xy_maxwellDistributionForVelocity_eachCell(
                xminLocal, xmaxLocal, yminLocal, ymaxLocal, 
                bulkVx, bulkVy, bulkVz,  
                PIC2DConst::vThIon, PIC2DConst::vThIon, PIC2DConst::vThIon, 
                countIon, countIon + ni, 
                j + i * PIC2DConst::ny + mPIInfo.rank * mPIInfo.localNx * PIC2DConst::ny, 
                particlesIon
            );
            initializeParticle.uniformForPosition_xy_maxwellDistributionForVelocity_eachCell(
                xminLocal, xmaxLocal, yminLocal, ymaxLocal, 
                bulkVx, bulkVy, bulkVz,  
                PIC2DConst::vThElectron, PIC2DConst::vThElectron, PIC2DConst::vThElectron, 
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
        Cf, waveAmp, waveNumber, 
        device_mPIInfo
    );
    cudaDeviceSynchronize();

    MPI_Barrier(MPI_COMM_WORLD);

    boundaryPIC.periodicBoundaryB_x(B);
    boundaryPIC.periodicBoundaryE_x(E);
    boundaryPIC.periodicBoundaryCurrent_x(current);
    boundaryPIC.periodicBoundaryForInitializeParticle_x(particlesIon, particlesElectron);
    
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

    mPIInfoPIC.existNumIonPerProcs      = PIC2DConst::totalNumIon / mPIInfoPIC.procs;
    mPIInfoPIC.existNumElectronPerProcs = PIC2DConst::totalNumElectron / mPIInfoPIC.procs;
    mPIInfoPIC.totalNumIonPerProcs = mPIInfoPIC.existNumIonPerProcs * 2;
    mPIInfoPIC.totalNumElectronPerProcs = mPIInfoPIC.existNumElectronPerProcs * 2;

    mPIInfoPIC.xminForProcs = PIC2DConst::xmin
                            + (PIC2DConst::xmax - PIC2DConst::xmin) / mPIInfoPIC.gridX
                            * mPIInfoPIC.localGridX;
    mPIInfoPIC.xmaxForProcs = PIC2DConst::xmin
                            + (PIC2DConst::xmax - PIC2DConst::xmin) / mPIInfoPIC.gridX
                            * (mPIInfoPIC.localGridX + 1);
    
    for (int i = 0; i < mPIInfoPIC.localSizeX; i++) {
        for (int j = 0; j < PIC2DConst::ny; j++) {
            double delta = 3.0; 

            host_interlockingFunctionY[j + i * PIC2DConst::ny]
                = 1.0
                - (1.0 - exp(-pow((j - 0) / delta, 2)))
                * (1.0 - exp(-pow((j - (PIC2DConst::ny - 1)) / delta, 2))); 
        }
    }
    
    IdealMHD2D idealMHD2D(mPIInfoMHD);
    PIC2D pIC2D(mPIInfoPIC); 
    InterfaceNoiseRemover2D interfaceNoiseRemover2D( 
        mPIInfoMHD, mPIInfoPIC
    );
    Interface2D interface2D(
        mPIInfoMHD, mPIInfoPIC, mPIInfoInterface, 
        indexOfInterfaceStartInMHD, 
        host_interlockingFunctionY, 
        interfaceNoiseRemover2D
    );
    BoundaryMHD& boundaryMHD = idealMHD2D.getBoundaryMHDRef(); 
    BoundaryPIC& boundaryPIC = pIC2D.getBoundaryPICRef(); 
    

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
    }

    idealMHD2D.initializeU(); 
    pIC2D.initialize();

    const int totalSubstep = int(round(sqrt(PIC2DConst::mRatio)));
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
            idealMHD2D.save(
                directoryName, filenameWithoutStep + "_U", step
            );
        }

        double dtCommon = min(0.7 * PIC2DConst::c, 0.1 * 1.0 / PIC2DConst::omegaPe);
        PIC2DConst::dt = dtCommon;
        IdealMHD2DConst::dt = totalSubstep * dtCommon;
        cudaMemcpyToSymbol(PIC2DConst::device_dt, &PIC2DConst::dt, sizeof(float));
        cudaMemcpyToSymbol(IdealMHD2DConst::device_dt, &IdealMHD2DConst::dt, sizeof(double));

        // STEP1 : MHD step

        idealMHD2D.setPastU();
        thrust::device_vector<ConservationParameter>& UPast = idealMHD2D.getUPastRef();

        idealMHD2D.oneStepRK2_periodicXSymmetricY_predictor();

        thrust::device_vector<ConservationParameter>& UNext = idealMHD2D.getURef();


        // STEP2 : send MHD to PIC

        float mixingRatio = 0.5f;
        thrust::device_vector<ConservationParameter>& USub = interface2D.calculateAndGetSubU(UPast, UNext, mixingRatio);
        
        thrust::device_vector<MagneticField>& B = pIC2D.getBRef();
        interface2D.sendMHDtoPIC_magneticField_y(USub, B);
        boundaryPIC.periodicBoundaryB_x(B);
        boundaryPIC.freeBoundaryB_y(B);
        
        thrust::device_vector<ElectricField>& E = pIC2D.getERef();
        interface2D.sendMHDtoPIC_electricField_y(USub, E);
        boundaryPIC.periodicBoundaryE_x(E);
        boundaryPIC.freeBoundaryE_y(E);    

        thrust::device_vector<Particle>& particlesIon = pIC2D.getParticlesIonRef();
        thrust::device_vector<Particle>& particlesElectron = pIC2D.getParticlesElectronRef();
        interface2D.sendMHDtoPIC_particle(
            USub, particlesIon, particlesElectron, step * totalSubstep
        );
        boundaryPIC.periodicBoundaryParticle_x(
            particlesIon, particlesElectron
        );
        boundaryPIC.freeBoundaryParticle_y(
            particlesIon, particlesElectron
        );

        // STEP3 : PIC step

        interface2D.resetTimeAveParameters();

        int getDataSubstep = totalSubstep / 2 + 1; 
        for (int substep = 1; substep <= totalSubstep; substep++) {
            pIC2D.oneStep_periodicXFreeY();

            if (substep == getDataSubstep) {
                thrust::device_vector<MagneticField>& B = pIC2D.getBRef();
                thrust::device_vector<MagneticField>& tmpB = pIC2D.getTmpBRef();
                thrust::copy(B.begin(), B.end(), tmpB.begin());
                thrust::device_vector<Particle>& particlesIon = pIC2D.getParticlesIonRef();
                thrust::device_vector<Particle>& particlesElectron = pIC2D.getParticlesElectronRef();
                
                interface2D.sumUpTimeAveParameters(tmpB, particlesIon, particlesElectron);
            }
        }

        // STEP4 : send PIC to MHD
        
        thrust::device_vector<ConservationParameter>& U = idealMHD2D.getURef();

        interface2D.sendPICtoMHD(U);

        boundaryMHD.periodicBoundaryX2nd_U(U);
        boundaryMHD.symmetricBoundaryY2nd_U(U);
        
        for (int count = 0; count < Interface2DConst::convolutionCount; count++) {
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

    MPI_Finalize();

    if (mPIInfoMHD.rank == 0) {
        std::cout << "program was completed!" << std::endl;
    }

    return 0;
}



