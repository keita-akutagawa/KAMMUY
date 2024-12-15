#include "main_test_const.hpp"


// 別にinitializeUを作ることにする。
void IdealMHD2D::initializeU()
{
}


__global__ void initializeU_lower_kernel(
    ConservationParameter* U, 
    IdealMHD2DMPI::MPIInfo* device_mPIInfo
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < IdealMHD2DConst::device_nx && j < IdealMHD2DConst::device_ny) {
        IdealMHD2DMPI::MPIInfo mPIInfo = *device_mPIInfo;

        if (mPIInfo.isInside(i, j)) {
            int index = mPIInfo.globalToLocal(i, j);

            double rho, u, v, w, bX, bY, bZ, e, p;
            
            rho = IdealMHD2DConst::device_rho0;
            u   = 0.0;
            v   = 0.0;
            w   = 0.0;
            bX  = 0.0;
            bY  = 0.0;
            bZ  = 0.0;
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


__global__ void initializeU_upper_kernel(
    ConservationParameter* U, 
    IdealMHD2DMPI::MPIInfo* device_mPIInfo
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < IdealMHD2DConst::device_nx && j < IdealMHD2DConst::device_ny) {
        IdealMHD2DMPI::MPIInfo mPIInfo = *device_mPIInfo;

        if (mPIInfo.isInside(i, j)) {
            int index = mPIInfo.globalToLocal(i, j);
            double rho, u, v, w, bX, bY, bZ, e, p;
            
            rho = IdealMHD2DConst::device_rho0;
            u   = 0.0;
            v   = 0.0;
            w   = 0.0;
            bX  = 0.0;
            bY  = 0.0;
            bZ  = 0.0;
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


void initializeU(
    thrust::device_vector<ConservationParameter>& U_lower, 
    thrust::device_vector<ConservationParameter>& U_upper, 
    BoundaryMHD& boundaryMHD, 
    IdealMHD2DMPI::MPIInfo& mPIInfoMHD
)
{
    IdealMHD2DMPI::MPIInfo* device_mPIInfoMHD; 
    cudaMalloc(&device_mPIInfoMHD, sizeof(IdealMHD2DMPI::MPIInfo));
    cudaMemcpy(device_mPIInfoMHD, &mPIInfoMHD, sizeof(IdealMHD2DMPI::MPIInfo), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((IdealMHD2DConst::nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (IdealMHD2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    initializeU_lower_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U_lower.data()), 
        device_mPIInfoMHD
    );
    cudaDeviceSynchronize();

    initializeU_upper_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U_upper.data()), 
        device_mPIInfoMHD
    );
    cudaDeviceSynchronize();

    MPI_Barrier(MPI_COMM_WORLD);

    IdealMHD2DMPI::sendrecv_U(U_lower, mPIInfoMHD);
    boundaryMHD.periodicBoundaryX2nd_U(U_lower);
    boundaryMHD.symmetricBoundaryY2nd_U(U_lower);
    IdealMHD2DMPI::sendrecv_U(U_upper, mPIInfoMHD);
    boundaryMHD.periodicBoundaryX2nd_U(U_upper);
    boundaryMHD.symmetricBoundaryY2nd_U(U_upper);

    MPI_Barrier(MPI_COMM_WORLD);
}


__global__ void initializePICField_kernel(
    ElectricField* E, MagneticField* B, 
    PIC2DMPI::MPIInfo* device_mPIInfo
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < PIC2DConst::device_nx && j < PIC2DConst::device_ny) {
        PIC2DMPI::MPIInfo mPIInfo = *device_mPIInfo;

        if (mPIInfo.isInside(i, j)) {
            int index = mPIInfo.globalToLocal(i, j);
            double u, v, w, bX, bY, bZ, eX, eY, eZ;

            bX = 0.0;
            bY = 0.0; 
            bZ = 0.0;
            u  = 0.0;
            v  = 0.0;
            w  = 0.0;
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

    for (int i = 0; i < mPIInfo.localNx; i++) {
        for (int j = 0; j < mPIInfo.localSizeY; j++) {
            float xminLocal, xmaxLocal, yminLocal, ymaxLocal;
            float bulkVx, bulkVy, bulkVz;

            xminLocal = i * PIC2DConst::dx + mPIInfo.xminForProcs;
            xmaxLocal = (i + 1) * PIC2DConst::dx + mPIInfo.xminForProcs;
            yminLocal = j * PIC2DConst::dy + mPIInfo.yminForProcs;
            ymaxLocal = (j + 1) * PIC2DConst::dy + mPIInfo.yminForProcs;
            bulkVx = 0.0;
            bulkVy = 0.0;
            bulkVz = 0.0;

            initializeParticle.uniformForPosition_xy_maxwellDistributionForVelocity_eachCell(
                xminLocal, xmaxLocal, yminLocal, ymaxLocal, 
                bulkVx, bulkVy, bulkVz,  
                PIC2DConst::vThIon, PIC2DConst::vThIon, PIC2DConst::vThIon, 
                (j + i * mPIInfo.localNy) * PIC2DConst::numberDensityIon, (j + i * mPIInfo.localNy + 1) * PIC2DConst::numberDensityIon, 
                j + i * mPIInfo.localNy + mPIInfo.rank * mPIInfo.localNx * mPIInfo.localNy, 
                particlesIon
            );
            initializeParticle.uniformForPosition_xy_maxwellDistributionForVelocity_eachCell(
                xminLocal, xmaxLocal, yminLocal, ymaxLocal, 
                bulkVx, bulkVy, bulkVz,  
                PIC2DConst::vThElectron, PIC2DConst::vThElectron, PIC2DConst::vThElectron, 
                (j + i * mPIInfo.localNy) * PIC2DConst::numberDensityElectron, (j + i * mPIInfo.localNy + 1) * PIC2DConst::numberDensityElectron, 
                j + i * mPIInfo.localNy + mPIInfo.localNx * mPIInfo.localNy + mPIInfo.rank * mPIInfo.localNx * mPIInfo.localNy, 
                particlesElectron
            );
        }
    }


    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((PIC2DConst::nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (PIC2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    initializePICField_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(E.data()), thrust::raw_pointer_cast(B.data()), 
        device_mPIInfo
    );
    cudaDeviceSynchronize();

    MPI_Barrier(MPI_COMM_WORLD);

    PIC2DMPI::sendrecv_field(B, mPIInfo, mPIInfo.mpi_fieldType);
    PIC2DMPI::sendrecv_field(E, mPIInfo, mPIInfo.mpi_fieldType);
    PIC2DMPI::sendrecv_field(current, mPIInfo, mPIInfo.mpi_fieldType);

    boundaryPIC.periodicBoundaryB_x(B);
    boundaryPIC.freeBoundaryB_y(B);
    boundaryPIC.periodicBoundaryE_x(E);
    boundaryPIC.freeBoundaryE_y(E);
    boundaryPIC.periodicBoundaryCurrent_x(current);
    boundaryPIC.freeBoundaryCurrent_y(current);
    boundaryPIC.periodicBoundaryForInitializeParticle_x(particlesIon, particlesElectron);
    //boundaryPIC.freeBoundaryForInitializeParticle_y(particlesIon, particlesElectron);
    
    MPI_Barrier(MPI_COMM_WORLD);
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

    mPIInfoPIC.existNumIonPerProcs      = static_cast<unsigned long long>(PIC2DConst::totalNumIon / mPIInfoPIC.procs);
    mPIInfoPIC.existNumElectronPerProcs = static_cast<unsigned long long>(PIC2DConst::totalNumElectron / mPIInfoPIC.procs);
    mPIInfoPIC.totalNumIonPerProcs = mPIInfoPIC.existNumIonPerProcs
                                   + PIC2DConst::numberDensityIon * (mPIInfoPIC.localSizeX + mPIInfoPIC.localSizeY) * (2 * mPIInfoPIC.buffer + 10)
                                   + Interface2DConst::reloadParticlesTotalNum;
    mPIInfoPIC.totalNumElectronPerProcs = mPIInfoPIC.existNumElectronPerProcs
                                        + PIC2DConst::numberDensityElectron * (mPIInfoPIC.localSizeX + mPIInfoPIC.localSizeY) * (2 * mPIInfoPIC.buffer + 10)
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


    thrust::device_vector<ConservationParameter>& U_lower = idealMHD2D_lower.getURef();
    thrust::device_vector<ConservationParameter>& U_upper = idealMHD2D_upper.getURef();

    initializeU(U_lower, U_upper, boundaryMHD, mPIInfoMHD);
    pIC2D.initialize();

    const int totalSubstep = int(round(sqrt(PIC2DConst::mRatio)));
    for (int step = 0; step < IdealMHD2DConst::totalStep + 1; step++) {
        MPI_Barrier(MPI_COMM_WORLD);

        if (mPIInfoPIC.rank == 0) {
            if (step % 1 == 0) {
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
            IdealMHD2DMPI::sendrecv_U_x(U_upper, mPIInfoMHD);
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


