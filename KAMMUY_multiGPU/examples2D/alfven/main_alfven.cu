#include "main_alfven_const.hpp"


// 別にinitializeUを作ることにする。
void IdealMHD2D::initializeU()
{
}


__global__ void initializeU_lower_kernel(
    ConservationParameter* U, 
    double VA, double waveAmp, double waveNumber, 
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
            double y = j * IdealMHD2DConst::device_dy;
            
            rho = IdealMHD2DConst::device_rho0;
            u   = waveAmp * VA * cos(waveNumber * y);
            v   = 0.0;
            w   = -waveAmp * VA * sin(waveNumber * y);
            bX  = -waveAmp * IdealMHD2DConst::device_B0 * cos(waveNumber * y);
            bY  = IdealMHD2DConst::device_B0;
            bZ  = waveAmp * IdealMHD2DConst::device_B0 * sin(waveNumber * y);
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
    double VA, double waveAmp, double waveNumber, 
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
            double y = j * PIC2DConst::device_dy + 9500 * IdealMHD2DConst::device_dy + 950 * PIC2DConst::device_dy;
            
            rho = IdealMHD2DConst::device_rho0;
            u   = waveAmp * VA * cos(waveNumber * y);
            v   = 0.0;
            w   = -waveAmp * VA * sin(waveNumber * y);
            bX  = -waveAmp * IdealMHD2DConst::device_B0 * cos(waveNumber * y);
            bY  = IdealMHD2DConst::device_B0;
            bZ  = waveAmp * IdealMHD2DConst::device_B0 * sin(waveNumber * y);
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

    double VA = IdealMHD2DConst::B0 / sqrt(IdealMHD2DConst::rho0); 

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((IdealMHD2DConst::nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (IdealMHD2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    initializeU_lower_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U_lower.data()), 
        VA, waveAmp, waveNumber, 
        device_mPIInfoMHD
    );
    cudaDeviceSynchronize();

    initializeU_upper_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U_upper.data()), 
        VA, waveAmp, waveNumber, 
        device_mPIInfoMHD
    );
    cudaDeviceSynchronize();

    MPI_Barrier(MPI_COMM_WORLD);

    IdealMHD2DMPI::sendrecv_U(U_lower, mPIInfoMHD);
    boundaryMHD.periodicBoundaryX2nd_U(U_lower);
    boundaryMHD.periodicBoundaryY2nd_U(U_lower);
    IdealMHD2DMPI::sendrecv_U(U_upper, mPIInfoMHD);
    boundaryMHD.periodicBoundaryX2nd_U(U_upper);
    boundaryMHD.periodicBoundaryY2nd_U(U_upper);

    MPI_Barrier(MPI_COMM_WORLD);
}


__global__ void initializePICField_kernel(
    ElectricField* E, MagneticField* B, 
    double VA, double waveAmp, double waveNumber, 
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
            double y = j * PIC2DConst::device_dy + 9500 * IdealMHD2DConst::device_dy;

            bX = -waveAmp * PIC2DConst::device_B0 * cos(waveNumber * y);
            bY = PIC2DConst::device_B0; 
            bZ = waveAmp * PIC2DConst::device_B0 * sin(waveNumber * y);
            u  = waveAmp * VA * cos(waveNumber * y);
            v  = 0.0;
            w  = -waveAmp * VA * sin(waveNumber * y);
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
    double VA = IdealMHD2DConst::B0 / sqrt(IdealMHD2DConst::rho0); 

    /*
    for (int i = 0; i < PIC2DConst::nx; i++) {
        for (int j = 0; j < PIC2DConst::ny; j++) {
            double xmin, ymin, u, v, w;
            double y = j * PIC2DConst::dy + 9500 * IdealMHD2DConst::dy;

            xmin = i * PIC2DConst::dx + PIC2DConst::xmin;
            ymin = j * PIC2DConst::dy + PIC2DConst::ymin;
            u = waveAmp * VA * cos(waveNumber * y);
            v = 0.0;
            w = -waveAmp * VA * sin(waveNumber * y);

            initializeParticle.uniformForPositionXY_maxwellDistributionForVelocity_detail(
                xmin, ymin, u, v, w, vThIon, vThIon, vThIon, 
                (j + i * ny) * numberDensityIon, (j + i * ny + 1) * numberDensityIon, j + i * ny, particlesIon
            );
            initializeParticle.uniformForPositionXY_maxwellDistributionForVelocity_detail(
                xmin, ymin, u, v, w, vThElectron, vThElectron, vThElectron, 
                (j + i * ny) * numberDensityElectron, (j + i * ny + 1) * numberDensityElectron, j + i * ny + nx * ny, particlesElectron
            );
        }
    }
    */


    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((PIC2DConst::nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (PIC2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    initializePICField_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(E.data()), thrust::raw_pointer_cast(B.data()), 
        VA, waveAmp, waveNumber, 
        device_mPIInfo
    );
    cudaDeviceSynchronize();

    MPI_Barrier(MPI_COMM_WORLD);

    PIC2DMPI::sendrecv_field(B, mPIInfo);
    PIC2DMPI::sendrecv_field(E, mPIInfo);
    PIC2DMPI::sendrecv_field(current, mPIInfo);

    boundaryPIC.periodicBoundaryB_x(B);
    boundaryPIC.periodicBoundaryB_y(B);
    boundaryPIC.periodicBoundaryE_x(E);
    boundaryPIC.periodicBoundaryE_y(E);
    boundaryPIC.periodicBoundaryCurrent_x(current);
    boundaryPIC.periodicBoundaryCurrent_y(current);
    boundaryPIC.boundaryForInitializeParticle_xy(particlesIon, particlesElectron);
    
    MPI_Barrier(MPI_COMM_WORLD);
}



int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    PIC2DMPI::MPIInfo mPIInfoPIC;
    PIC2DMPI::setupInfo(mPIInfoPIC, buffer);
    IdealMHD2DMPI::MPIInfo mPIInfoMHD;
    IdealMHD2DMPI::setupInfo(mPIInfoMHD, buffer);

    if (mPIInfoPIC.rank == 0) {
        std::cout     << mPIInfoPIC.gridX << "," << mPIInfoPIC.gridY << std::endl;
        mpifile   << mPIInfoPIC.gridX << "," << mPIInfoPIC.gridY << std::endl;
    }
    if (mPIInfoMHD.rank == 0) {
        std::cout     << mPIInfoMHD.gridX << "," << mPIInfoMHD.gridY << std::endl;
        mpifile   << mPIInfoMHD.gridX << "," << mPIInfoMHD.gridY << std::endl;
    }

    cudaSetDevice(mPIInfoPIC.rank);

    PIC2DConst::initializeDeviceConstants();
    IdealMHD2DConst::initializeDeviceConstants();
    Interface2DConst::initializeDeviceConstants();

    for (int i = 0; i < interfaceLength; i++) {
        host_interlockingFunctionY_lower[i] = max(
            0.5 * (1.0 + cos(Interface2DConst::PI * (i - 0.0) / (interfaceLength - 0.0))), 
            1e-20
        );
        host_interlockingFunctionY_upper[i] = max(
            0.5 * (1.0 - cos(Interface2DConst::PI * (i - 0.0) / (interfaceLength - 0.0))), 
            Interface2DConst::EPS
        );
    }
    for (int i = 0; i < interfaceLength; i++) {
        host_interlockingFunctionYHalf_lower[i] = max(
            0.5 * (1.0 + cos(Interface2DConst::PI * (i + 0.5 - 0.0) / (interfaceLength - 0.0))), 
            1e-20
        );
        host_interlockingFunctionYHalf_upper[i] = max(
            0.5 * (1.0 - cos(Interface2DConst::PI * (i + 0.5 - 0.0) / (interfaceLength - 0.0))), 
            Interface2DConst::EPS
        );
    }


    IdealMHD2D idealMHD2D_lower(mPIInfoMHD);
    IdealMHD2D idealMHD2D_upper(mPIInfoMHD);
    PIC2D pIC2D(mPIInfoPIC); 
    InterfaceNoiseRemover2D interfaceNoiseRemover2D( 
        mPIInfoMHD, mPIInfoPIC, 
        indexOfInterfaceStartInMHD_lower, 
        indexOfInterfaceStartInPIC_lower, 
        indexOfInterfaceStartInMHD_upper, 
        indexOfInterfaceStartInPIC_upper, 
        Interface2DConst::interfaceLength, 
        Interface2DConst::windowSizeForConvolution, 
        nxInterface, nyInterface
    );
    Interface2D interface2D_lower(
        mPIInfoMHD, mPIInfoPIC, 
        indexOfInterfaceStartInMHD_lower, 
        indexOfInterfaceStartInPIC_lower, 
        Interface2DConst::interfaceLength, 
        host_interlockingFunctionY_lower, 
        host_interlockingFunctionYHalf_lower, 
        interfaceNoiseRemover2D
    );
    Interface2D interface2D_upper(
        mPIInfoMHD, mPIInfoPIC, 
        indexOfInterfaceStartInMHD_upper, 
        indexOfInterfaceStartInPIC_upper, 
        Interface2DConst::interfaceLength, 
        host_interlockingFunctionY_upper, 
        host_interlockingFunctionYHalf_upper,
        interfaceNoiseRemover2D
    );
    //BoundaryPIC boundaryPIC;
    BoundaryMHD boundaryMHD(mPIInfoMHD);
    

    if (mPIInfoPIC.rank == 0) {
        size_t free_mem = 0;
        size_t total_mem = 0;
        cudaError_t status = cudaMemGetInfo(&free_mem, &total_mem);

        std::cout << "Free memory: " << free_mem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "Total memory: " << total_mem / (1024 * 1024) << " MB" << std::endl;

        std::cout << "exist number of partices is " 
                  << mPIInfoPIC.procs * (mPIInfo.existNumIonPerProcs + mPIInfoPIC.existNumElectronPerProcs) 
                  << std::endl;
        std::cout << "exist number of partices + buffer particles is " 
                  << mPIInfoPIC.procs * (mPIInfo.totalNumIonPerProcs + mPIInfoPIC.totalNumElectronPerProcs) 
                  << std::endl;
        std::cout << std::setprecision(4) 
                << "omega_pe * t = " << PIC2DConst::totalStep * PIC2DConst::dt * PIC2DConst::omegaPe << std::endl;
    }


    thrust::device_vector<ConservationParameter>& U_lower = idealMHD2D_lower.getURef();
    thrust::device_vector<ConservationParameter>& U_upper = idealMHD2D_upper.getURef();

    initializeU(U_lower, U_upper, boundaryMHD, mPIInfoMHD);
    pIC2D.initialize();

    const int totalSubstep = int(round(sqrt(PIC2DConst::mRatio)));
    for (int step = 0; step < IdealMHD2DConst::totalStep + 1; step++) {
        if (step % 10 == 0) {
            std::cout << std::to_string(step) << " step done : total time is "
                      << std::setprecision(4) << step * totalSubstep * PIC2DConst::dt * PIC2DConst::omegaPe
                      << " [omega_pe * t]"
                      << std::endl;
        }

        if (step % recordStep == 0) {
            logfile << std::setprecision(6) << PIC2DConst::totalTime << std::endl;
            pIC2D.saveFields(
                directoryname, filenameWithoutStep, step
            );
            pIC2D.saveZerothMoments(
                directoryname, filenameWithoutStep, step
            );
            pIC2D.saveFirstMoments(
                directoryname, filenameWithoutStep, step
            );
            idealMHD2D_lower.save(
                directoryname, filenameWithoutStep + "_lower", step
            );
            idealMHD2D_upper.save(
                directoryname, filenameWithoutStep + "_upper", step
            );
        }
        
        if (isParticleRecord && step % particleRecordStep == 0) {
            pIC2D.saveParticle(
                directoryname, filenameWithoutStep, step
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

        idealMHD2D_lower.oneStepRK2_predictor();
        idealMHD2D_upper.oneStepRK2_predictor();

        thrust::device_vector<ConservationParameter>& UNext_lower = idealMHD2D_lower.getURef();
        thrust::device_vector<ConservationParameter>& UNext_upper = idealMHD2D_upper.getURef();


        // STEP2 : PIC

        interface2D_lower.resetTimeAveParameters();
        interface2D_upper.resetTimeAveParameters();
        for (int substep = 1; substep <= totalSubstep; substep++) {
            pIC2D.oneStepPeriodicXFreeY(
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
        boundaryMHD.periodicBoundaryX2nd(UHalf_lower);
        boundaryMHD.symmetricBoundaryY2nd(UHalf_lower);
        boundaryMHD.periodicBoundaryX2nd(UHalf_upper);
        boundaryMHD.symmetricBoundaryY2nd(UHalf_upper);

        idealMHD2D_lower.oneStepRK2_corrector(UHalf_lower);
        idealMHD2D_upper.oneStepRK2_corrector(UHalf_upper);

        U_lower = idealMHD2D_lower.getURef();
        U_upper = idealMHD2D_upper.getURef();
        interfaceNoiseRemover2D_lower.convolveU_lower(U_lower);
        interfaceNoiseRemover2D_upper.convolveU_upper(U_upper);


        if (idealMHD2D_lower.checkCalculationIsCrashed() || idealMHD2D_upper.checkCalculationIsCrashed()) {
            logfile << std::setprecision(6) << PIC2DConst::totalTime << std::endl;
            pIC2D.saveFields(
                directoryname, filenameWithoutStep, step
            );
            pIC2D.saveZerothMoments(
                directoryname, filenameWithoutStep, step
            );
            pIC2D.saveFirstMoments(
                directoryname, filenameWithoutStep, step
            );
            idealMHD2D_lower.save(
                directoryname, filenameWithoutStep + "_lower", step
            );
            idealMHD2D_upper.save(
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

        IdealMHD2DConst::totalTime += IdealMHD2DConst::dt;
    }

    return 0;
}


