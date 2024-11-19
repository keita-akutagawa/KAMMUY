#include "pic2D.hpp"


PIC2D::PIC2D(PIC2DMPI::MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo), 

      particlesIon     (mPIInfo.totalNumIonPerProcs), 
      particlesElectron(mPIInfo.totalNumElectronPerProcs),
      E                   (mPIInfo.localSizeX * mPIInfo.localSizeY), 
      tmpE                (mPIInfo.localSizeX * mPIInfo.localSizeY), 
      B                   (mPIInfo.localSizeX * mPIInfo.localSizeY), 
      tmpB                (mPIInfo.localSizeX * mPIInfo.localSizeY), 
      current             (mPIInfo.localSizeX * mPIInfo.localSizeY), 
      tmpCurrent          (mPIInfo.localSizeX * mPIInfo.localSizeY), 
      zerothMomentIon     (mPIInfo.localSizeX * mPIInfo.localSizeY), 
      zerothMomentElectron(mPIInfo.localSizeX * mPIInfo.localSizeY), 
      firstMomentIon      (mPIInfo.localSizeX * mPIInfo.localSizeY), 
      firstMomentElectron (mPIInfo.localSizeX * mPIInfo.localSizeY), 
      secondMomentIon     (mPIInfo.localSizeX * mPIInfo.localSizeY), 
      secondMomentElectron(mPIInfo.localSizeX * mPIInfo.localSizeY), 

      host_particlesIon     (mPIInfo.totalNumIonPerProcs), 
      host_particlesElectron(mPIInfo.totalNumElectronPerProcs), 
      host_E                   (mPIInfo.localSizeX * mPIInfo.localSizeY),  
      host_B                   (mPIInfo.localSizeX * mPIInfo.localSizeY),  
      host_current             (mPIInfo.localSizeX * mPIInfo.localSizeY),  
      host_zerothMomentIon     (mPIInfo.localSizeX * mPIInfo.localSizeY),  
      host_zerothMomentElectron(mPIInfo.localSizeX * mPIInfo.localSizeY),  
      host_firstMomentIon      (mPIInfo.localSizeX * mPIInfo.localSizeY),  
      host_firstMomentElectron (mPIInfo.localSizeX * mPIInfo.localSizeY),  
      host_secondMomentIon     (mPIInfo.localSizeX * mPIInfo.localSizeY),  
      host_secondMomentElectron(mPIInfo.localSizeX * mPIInfo.localSizeY), 

      initializeParticle(mPIInfo), 
      particlePush      (mPIInfo), 
      fieldSolver       (mPIInfo), 
      currentCalculator (mPIInfo), 
      boundaryPIC       (mPIInfo), 
      momentCalculater  (mPIInfo), 
      filter            (mPIInfo)
{

    cudaMalloc(&device_mPIInfo, sizeof(PIC2DMPI::MPIInfo));
    cudaMemcpy(device_mPIInfo, &mPIInfo, sizeof(PIC2DMPI::MPIInfo), cudaMemcpyHostToDevice);
    
}


__global__ void getCenterBE_kernel(
    MagneticField* tmpB, ElectricField* tmpE, 
    const MagneticField* B, const ElectricField* E, 
    int localSizeX, int localSizeY
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < localSizeX) && (0 < j) && (j < localSizeY)) {
        int index = j + localSizeY * i; 

        tmpB[index].bX = 0.5f * (B[index].bX + B[index - 1].bX);
        tmpB[index].bY = 0.5f * (B[index].bY + B[index - localSizeY].bY);
        tmpB[index].bZ = 0.25f * (B[index].bZ + B[index - localSizeY].bZ
                                + B[index - 1].bZ + B[index - 1 - localSizeY].bZ);
        tmpE[index].eX = 0.5f * (E[index].eX + E[index - localSizeY].eX);
        tmpE[index].eY = 0.5f * (E[index].eY + E[index - 1].eY);
        tmpE[index].eZ = E[index].eZ;
    }
}

__global__ void getHalfCurrent_kernel(
    CurrentField* current, const CurrentField* tmpCurrent, 
    int localSizeX, int localSizeY
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localSizeX - 1 && j < localSizeY - 1) {
        int index = j + localSizeY * i; 

        current[index].jX = 0.5f * (tmpCurrent[index].jX + tmpCurrent[index + localSizeY].jX);
        current[index].jY = 0.5f * (tmpCurrent[index].jY + tmpCurrent[index + 1].jY);
        current[index].jZ = tmpCurrent[index].jZ;
    }
}


void PIC2D::oneStep_periodicXFreeY(
    thrust::device_vector<ConservationParameter>& UPast_lower, 
    thrust::device_vector<ConservationParameter>& UPast_upper, 
    thrust::device_vector<ConservationParameter>& UNext_lower, 
    thrust::device_vector<ConservationParameter>& UNext_upper, 
    Interface2D& interface2D_lower, 
    Interface2D& interface2D_upper, 
    InterfaceNoiseRemover2D& interfaceNoiseRemover2D, 
    int step, int substep, int totalSubstep
)
{
    MPI_Barrier(MPI_COMM_WORLD);

    bool isLower, isUpper; 
    
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (mPIInfo.localSizeY + threadsPerBlock.y - 1) / threadsPerBlock.y);
                       

    float mixingRatio = (totalSubstep - substep) / totalSubstep;
    thrust::device_vector<ConservationParameter>& USub_lower = interface2D_lower.calculateAndGetSubU(UPast_lower, UNext_lower, mixingRatio);
    thrust::device_vector<ConservationParameter>& USub_upper = interface2D_upper.calculateAndGetSubU(UPast_upper, UNext_upper, mixingRatio);
                       
    fieldSolver.timeEvolutionB(B, E, PIC2DConst::dt / 2.0f);
    PIC2DMPI::sendrecv_field(B, mPIInfo, mPIInfo.mpi_fieldType);
    boundaryPIC.periodicBoundaryB_x(B);
    boundaryPIC.freeBoundaryB_y(B);
    
    getCenterBE_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpB.data()), 
        thrust::raw_pointer_cast(tmpE.data()), 
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(E.data()), 
        mPIInfo.localSizeX, mPIInfo.localSizeY
    );
    cudaDeviceSynchronize();
    PIC2DMPI::sendrecv_field(tmpB, mPIInfo, mPIInfo.mpi_fieldType);
    PIC2DMPI::sendrecv_field(tmpE, mPIInfo, mPIInfo.mpi_fieldType);
    boundaryPIC.periodicBoundaryB_x(tmpB);
    boundaryPIC.freeBoundaryB_y(tmpB);
    boundaryPIC.periodicBoundaryE_x(tmpE);
    boundaryPIC.freeBoundaryE_y(tmpE);

    particlePush.pushVelocity(
        particlesIon, particlesElectron, tmpB, tmpE, PIC2DConst::dt
    );

    particlePush.pushPosition(
        particlesIon, particlesElectron, PIC2DConst::dt / 2.0f
    );
    boundaryPIC.periodicBoundaryParticle_x(
        particlesIon, particlesElectron
    );
    boundaryPIC.modifySendNumParticles();
    boundaryPIC.freeBoundaryParticle_y(
        particlesIon, particlesElectron
    );

    currentCalculator.resetCurrent(tmpCurrent);
    currentCalculator.calculateCurrent(
        tmpCurrent, particlesIon, particlesElectron
    );
    PIC2DMPI::sendrecv_field(tmpCurrent, mPIInfo, mPIInfo.mpi_fieldType);
    boundaryPIC.periodicBoundaryCurrent_x(tmpCurrent);
    boundaryPIC.freeBoundaryCurrent_y(tmpCurrent);
    getHalfCurrent_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(current.data()), 
        thrust::raw_pointer_cast(tmpCurrent.data()), 
        mPIInfo.localSizeX, mPIInfo.localSizeY
    );
    PIC2DMPI::sendrecv_field(current, mPIInfo, mPIInfo.mpi_fieldType);
    boundaryPIC.periodicBoundaryCurrent_x(current);
    boundaryPIC.freeBoundaryCurrent_y(current);
    interface2D_lower.sendMHDtoPIC_currentField_yDirection(USub_lower, current);
    interface2D_upper.sendMHDtoPIC_currentField_yDirection(USub_upper, current);
    PIC2DMPI::sendrecv_field(current, mPIInfo, mPIInfo.mpi_fieldType);
    boundaryPIC.periodicBoundaryCurrent_x(current);
    boundaryPIC.freeBoundaryCurrent_y(current);
    //isLower = true, isUpper = false;
    //interfaceNoiseRemover2D.convolve_currentField(current, isLower, isUpper);
    //isLower = false, isUpper = true;
    //interfaceNoiseRemover2D.convolve_currentField(current, isLower, isUpper);
    //boundaryPIC.periodicBoundaryCurrent_x(current);
    //boundaryPIC.freeBoundaryCurrent_y(current);

    fieldSolver.timeEvolutionB(B, E, PIC2DConst::dt / 2.0f);
    PIC2DMPI::sendrecv_field(B, mPIInfo, mPIInfo.mpi_fieldType);
    boundaryPIC.periodicBoundaryB_x(B);
    boundaryPIC.freeBoundaryB_y(B);

    fieldSolver.timeEvolutionE(E, B, current, PIC2DConst::dt);
    PIC2DMPI::sendrecv_field(E, mPIInfo, mPIInfo.mpi_fieldType);
    boundaryPIC.periodicBoundaryE_x(E);
    boundaryPIC.freeBoundaryE_y(E);
    filter.langdonMarderTypeCorrection(E, particlesIon, particlesElectron, PIC2DConst::dt);
    PIC2DMPI::sendrecv_field(E, mPIInfo, mPIInfo.mpi_fieldType);
    boundaryPIC.periodicBoundaryE_x(E);
    boundaryPIC.freeBoundaryE_y(E);

    particlePush.pushPosition(
        particlesIon, particlesElectron, PIC2DConst::dt / 2.0f
    );
    boundaryPIC.periodicBoundaryParticle_x(
        particlesIon, particlesElectron
    );
    boundaryPIC.modifySendNumParticles();
    boundaryPIC.freeBoundaryParticle_y(
        particlesIon, particlesElectron
    );

    //isLower = true, isUpper = false;
    //interfaceNoiseRemover2D.convolve_magneticField(B, isLower, isUpper);
    //interfaceNoiseRemover2D.convolve_electricField(E, isLower, isUpper);
    //isLower = false, isUpper = true;
    //interfaceNoiseRemover2D.convolve_magneticField(B, isLower, isUpper);
    //interfaceNoiseRemover2D.convolve_electricField(E, isLower, isUpper);

    interface2D_lower.sendMHDtoPIC_magneticField_yDirection(USub_lower, B);
    interface2D_upper.sendMHDtoPIC_magneticField_yDirection(USub_upper, B);
    PIC2DMPI::sendrecv_field(B, mPIInfo, mPIInfo.mpi_fieldType);
    boundaryPIC.periodicBoundaryB_x(B);
    boundaryPIC.freeBoundaryB_y(B);
    
    interface2D_lower.sendMHDtoPIC_electricField_yDirection(USub_lower, E);
    interface2D_upper.sendMHDtoPIC_electricField_yDirection(USub_upper, E);
    PIC2DMPI::sendrecv_field(E, mPIInfo, mPIInfo.mpi_fieldType);
    boundaryPIC.periodicBoundaryE_x(E);
    boundaryPIC.freeBoundaryE_y(E);    

    interface2D_lower.sendMHDtoPIC_particle(USub_lower, particlesIon, particlesElectron, step * totalSubstep + substep);
    interface2D_upper.sendMHDtoPIC_particle(USub_upper, particlesIon, particlesElectron, step * totalSubstep + substep);
}   


void PIC2D::saveFields(
    std::string directoryname, 
    std::string filenameWithoutStep, 
    int step
)
{
    host_E = E;
    host_B = B;
    host_current = current;
    std::string filenameB, filenameE, filenameCurrent;
    std::string filenameBEnergy, filenameEEnergy;
    float BEnergy = 0.0f, EEnergy = 0.0f;

    filenameB = directoryname + "/"
             + filenameWithoutStep + "_B_" + std::to_string(step)
             + "_" + std::to_string(mPIInfo.rank)
             + ".bin";
    filenameE = directoryname + "/"
             + filenameWithoutStep + "_E_" + std::to_string(step)
             + "_" + std::to_string(mPIInfo.rank)
             + ".bin";
    filenameCurrent = directoryname + "/"
             + filenameWithoutStep + "_current_" + std::to_string(step)
             + "_" + std::to_string(mPIInfo.rank)
             + ".bin";
    filenameBEnergy = directoryname + "/"
             + filenameWithoutStep + "_BEnergy_" + std::to_string(step)
             + "_" + std::to_string(mPIInfo.rank)
             + ".bin";
    filenameEEnergy = directoryname + "/"
             + filenameWithoutStep + "_EEnergy_" + std::to_string(step)
             + "_" + std::to_string(mPIInfo.rank)
             + ".bin";


    std::ofstream ofsB(filenameB, std::ios::binary);
    ofsB << std::fixed << std::setprecision(6);
    for (int i = mPIInfo.buffer; i < mPIInfo.localNx + mPIInfo.buffer; i++) {
        for (int j = mPIInfo.buffer; j < mPIInfo.localNy + mPIInfo.buffer; j++) {
            int index = j + mPIInfo.localSizeY * i;

            ofsB.write(reinterpret_cast<const char*>(&host_B[index].bX), sizeof(float));
            ofsB.write(reinterpret_cast<const char*>(&host_B[index].bY), sizeof(float));
            ofsB.write(reinterpret_cast<const char*>(&host_B[index].bZ), sizeof(float));
            BEnergy += host_B[index].bX * host_B[index].bX 
                     + host_B[index].bY * host_B[index].bY
                     + host_B[index].bZ * host_B[index].bZ;
        }
    }
    BEnergy *= 0.5f / PIC2DConst::mu0;

    std::ofstream ofsE(filenameE, std::ios::binary);
    ofsE << std::fixed << std::setprecision(6);
    for (int i = mPIInfo.buffer; i < mPIInfo.localNx + mPIInfo.buffer; i++) {
        for (int j = mPIInfo.buffer; j < mPIInfo.localNy + mPIInfo.buffer; j++) {
            int index = j + mPIInfo.localSizeY * i;

            ofsE.write(reinterpret_cast<const char*>(&host_E[index].eX), sizeof(float));
            ofsE.write(reinterpret_cast<const char*>(&host_E[index].eY), sizeof(float));
            ofsE.write(reinterpret_cast<const char*>(&host_E[index].eZ), sizeof(float));
            EEnergy += host_E[index].eX * host_E[index].eX
                     + host_E[index].eY * host_E[index].eY
                     + host_E[index].eZ * host_E[index].eZ;
        }
    }
    EEnergy *= 0.5f * PIC2DConst::epsilon0;

    std::ofstream ofsCurrent(filenameCurrent, std::ios::binary);
    ofsCurrent << std::fixed << std::setprecision(6);
    for (int i = mPIInfo.buffer; i < mPIInfo.localNx + mPIInfo.buffer; i++) {
        for (int j = mPIInfo.buffer; j < mPIInfo.localNy + mPIInfo.buffer; j++) {
            int index = j + mPIInfo.localSizeY * i;

            ofsCurrent.write(reinterpret_cast<const char*>(&host_current[index].jX), sizeof(float));
            ofsCurrent.write(reinterpret_cast<const char*>(&host_current[index].jY), sizeof(float));
            ofsCurrent.write(reinterpret_cast<const char*>(&host_current[index].jZ), sizeof(float));
        }
    }

    std::ofstream ofsBEnergy(filenameBEnergy, std::ios::binary);
    ofsBEnergy << std::fixed << std::setprecision(6);
    ofsBEnergy.write(reinterpret_cast<const char*>(&BEnergy), sizeof(float));

    std::ofstream ofsEEnergy(filenameEEnergy, std::ios::binary);
    ofsEEnergy << std::fixed << std::setprecision(6);
    ofsEEnergy.write(reinterpret_cast<const char*>(&EEnergy), sizeof(float));
}


void PIC2D::calculateFullMoments()
{
    calculateZerothMoments();
    calculateFirstMoments();
    calculateSecondMoments();
}


void PIC2D::calculateZerothMoments()
{
    momentCalculater.calculateZerothMomentOfOneSpecies(
        zerothMomentIon, particlesIon, mPIInfo.existNumIonPerProcs
    );
    momentCalculater.calculateZerothMomentOfOneSpecies(
        zerothMomentElectron, particlesElectron, mPIInfo.existNumElectronPerProcs
    );
}


void PIC2D::calculateFirstMoments()
{
    momentCalculater.calculateFirstMomentOfOneSpecies(
        firstMomentIon, particlesIon, mPIInfo.existNumIonPerProcs
    );
    momentCalculater.calculateFirstMomentOfOneSpecies(
        firstMomentElectron, particlesElectron, mPIInfo.existNumElectronPerProcs
    );
}


void PIC2D::calculateSecondMoments()
{
    momentCalculater.calculateSecondMomentOfOneSpecies(
        secondMomentIon, particlesIon, mPIInfo.existNumIonPerProcs
    );
    momentCalculater.calculateSecondMomentOfOneSpecies(
        secondMomentElectron, particlesElectron, mPIInfo.existNumElectronPerProcs
    );
}


void PIC2D::saveFullMoments(
    std::string directoryname, 
    std::string filenameWithoutStep, 
    int step
)
{
    saveZerothMoments(directoryname, filenameWithoutStep, step);
    saveFirstMoments(directoryname, filenameWithoutStep, step);
    saveSecondMoments(directoryname, filenameWithoutStep, step);
}


void PIC2D::saveZerothMoments(
    std::string directoryname, 
    std::string filenameWithoutStep, 
    int step
)
{
    calculateZerothMoments();

    host_zerothMomentIon = zerothMomentIon;
    host_zerothMomentElectron = zerothMomentElectron;

    std::string filenameZerothMomentIon, filenameZerothMomentElectron;
    
    filenameZerothMomentIon = directoryname + "/"
                            + filenameWithoutStep + "_zeroth_moment_ion_" + std::to_string(step)
                            + "_" + std::to_string(mPIInfo.rank)
                            + ".bin";
    filenameZerothMomentElectron = directoryname + "/"
                                 + filenameWithoutStep + "_zeroth_moment_electron_" + std::to_string(step)
                                 + "_" + std::to_string(mPIInfo.rank)
                                 + ".bin";
    

    std::ofstream ofsZerothMomentIon(filenameZerothMomentIon, std::ios::binary);
    ofsZerothMomentIon << std::fixed << std::setprecision(6);
    for (int i = mPIInfo.buffer; i < mPIInfo.localNx + mPIInfo.buffer; i++) {
        for (int j = mPIInfo.buffer; j < mPIInfo.localNy + mPIInfo.buffer; j++) {
            int index = j + mPIInfo.localSizeY * i;

            ofsZerothMomentIon.write(reinterpret_cast<const char*>(
                &host_zerothMomentIon[index].n), sizeof(float)
            );
        }
    }

    std::ofstream ofsZerothMomentElectron(filenameZerothMomentElectron, std::ios::binary);
    ofsZerothMomentElectron << std::fixed << std::setprecision(6);
    for (int i = mPIInfo.buffer; i < mPIInfo.localNx + mPIInfo.buffer; i++) {
        for (int j = mPIInfo.buffer; j < mPIInfo.localNy + mPIInfo.buffer; j++) {
            int index = j + mPIInfo.localSizeY * i;

            ofsZerothMomentElectron.write(reinterpret_cast<const char*>(
                &host_zerothMomentElectron[index].n), sizeof(float)
            );
        }
    }
}


void PIC2D::saveFirstMoments(
    std::string directoryname, 
    std::string filenameWithoutStep, 
    int step
)
{
    calculateFirstMoments();

    host_firstMomentIon = firstMomentIon;
    host_firstMomentElectron = firstMomentElectron;

    std::string filenameFirstMomentIon, filenameFirstMomentElectron;
    
    filenameFirstMomentIon = directoryname + "/"
                           + filenameWithoutStep + "_first_moment_ion_" + std::to_string(step)
                           + "_" + std::to_string(mPIInfo.rank)
                           + ".bin";
    filenameFirstMomentElectron = directoryname + "/"
                                + filenameWithoutStep + "_first_moment_electron_" + std::to_string(step)
                                + "_" + std::to_string(mPIInfo.rank)
                                + ".bin";
    

    std::ofstream ofsFirstMomentIon(filenameFirstMomentIon, std::ios::binary);
    ofsFirstMomentIon << std::fixed << std::setprecision(6);
    for (int i = mPIInfo.buffer; i < mPIInfo.localNx + mPIInfo.buffer; i++) {
        for (int j = mPIInfo.buffer; j < mPIInfo.localNy + mPIInfo.buffer; j++) {
            int index = j + mPIInfo.localSizeY * i;

            ofsFirstMomentIon.write(reinterpret_cast<const char*>(
                &host_firstMomentIon[index].x), sizeof(float)
            );
            ofsFirstMomentIon.write(reinterpret_cast<const char*>(
                &host_firstMomentIon[index].y), sizeof(float)
            );
            ofsFirstMomentIon.write(reinterpret_cast<const char*>(
                &host_firstMomentIon[index].z), sizeof(float)
            );
        }
    }

    std::ofstream ofsFirstMomentElectron(filenameFirstMomentElectron, std::ios::binary);
    ofsFirstMomentElectron << std::fixed << std::setprecision(6);
    for (int i = mPIInfo.buffer; i < mPIInfo.localNx + mPIInfo.buffer; i++) {
        for (int j = mPIInfo.buffer; j < mPIInfo.localNy + mPIInfo.buffer; j++) {
            int index = j + mPIInfo.localSizeY * i;

            ofsFirstMomentElectron.write(reinterpret_cast<const char*>(
                &host_firstMomentElectron[index].x), sizeof(float)
            );
            ofsFirstMomentElectron.write(reinterpret_cast<const char*>(
                &host_firstMomentElectron[index].y), sizeof(float)
            );
            ofsFirstMomentElectron.write(reinterpret_cast<const char*>(
                &host_firstMomentElectron[index].z), sizeof(float)
            );
        }
    }
}


void PIC2D::saveSecondMoments(
    std::string directoryname, 
    std::string filenameWithoutStep, 
    int step
)
{
    calculateSecondMoments();

    host_secondMomentIon = secondMomentIon;
    host_secondMomentElectron = secondMomentElectron;

    std::string filenameSecondMomentIon, filenameSecondMomentElectron;

    filenameSecondMomentIon = directoryname + "/"
                            + filenameWithoutStep + "_second_moment_ion_" + std::to_string(step)
                            + "_" + std::to_string(mPIInfo.rank)
                            + ".bin";
    filenameSecondMomentElectron = directoryname + "/"
                                 + filenameWithoutStep + "_second_moment_electron_" + std::to_string(step)
                                 + "_" + std::to_string(mPIInfo.rank)
                                 + ".bin";
    

    std::ofstream ofsSecondMomentIon(filenameSecondMomentIon, std::ios::binary);
    ofsSecondMomentIon << std::fixed << std::setprecision(6);
    for (int i = mPIInfo.buffer; i < mPIInfo.localNx + mPIInfo.buffer; i++) {
        for (int j = mPIInfo.buffer; j < mPIInfo.localNy + mPIInfo.buffer; j++) {
            int index = j + mPIInfo.localSizeY * i;

            ofsSecondMomentIon.write(reinterpret_cast<const char*>(
                &host_secondMomentIon[index].xx), sizeof(float)
            );
            ofsSecondMomentIon.write(reinterpret_cast<const char*>(
                &host_secondMomentIon[index].yy), sizeof(float)
            );
            ofsSecondMomentIon.write(reinterpret_cast<const char*>(
                &host_secondMomentIon[index].zz), sizeof(float)
            );
            ofsSecondMomentIon.write(reinterpret_cast<const char*>(
                &host_secondMomentIon[index].xy), sizeof(float)
            );
            ofsSecondMomentIon.write(reinterpret_cast<const char*>(
                &host_secondMomentIon[index].xz), sizeof(float)
            );
            ofsSecondMomentIon.write(reinterpret_cast<const char*>(
                &host_secondMomentIon[index].yz), sizeof(float)
            );
        }
    }

    std::ofstream ofsSecondMomentElectron(filenameSecondMomentElectron, std::ios::binary);
    ofsSecondMomentElectron << std::fixed << std::setprecision(6);
    for (int i = mPIInfo.buffer; i < mPIInfo.localNx + mPIInfo.buffer; i++) {
        for (int j = mPIInfo.buffer; j < mPIInfo.localNy + mPIInfo.buffer; j++) {
            int index = j + mPIInfo.localSizeY * i;

            ofsSecondMomentElectron.write(reinterpret_cast<const char*>(
                &host_secondMomentElectron[index].xx), sizeof(float)
            );
            ofsSecondMomentElectron.write(reinterpret_cast<const char*>(
                &host_secondMomentElectron[index].yy), sizeof(float)
            );
            ofsSecondMomentElectron.write(reinterpret_cast<const char*>(
                &host_secondMomentElectron[index].zz), sizeof(float)
            );
            ofsSecondMomentElectron.write(reinterpret_cast<const char*>(
                &host_secondMomentElectron[index].xy), sizeof(float)
            );
            ofsSecondMomentElectron.write(reinterpret_cast<const char*>(
                &host_secondMomentElectron[index].xz), sizeof(float)
            );
            ofsSecondMomentElectron.write(reinterpret_cast<const char*>(
                &host_secondMomentElectron[index].yz), sizeof(float)
            );
        }
    }
}


void PIC2D::saveParticle(
    std::string directoryname, 
    std::string filenameWithoutStep, 
    int step
)
{
    host_particlesIon = particlesIon;
    host_particlesElectron = particlesElectron;

    std::string filenameXIon, filenameXElectron;
    std::string filenameVIon, filenameVElectron;
    std::string filenameNumIon, filenameNumElectron;
    std::string filenameKineticEnergy;

    filenameXIon = directoryname + "/"
             + filenameWithoutStep + "_x_ion_" + std::to_string(step)
             + "_" + std::to_string(mPIInfo.rank)
             + ".bin";
    filenameXElectron = directoryname + "/"
             + filenameWithoutStep + "_x_electron_" + std::to_string(step)
             + "_" + std::to_string(mPIInfo.rank)
             + ".bin";
    filenameVIon = directoryname + "/"
             + filenameWithoutStep + "_v_ion_" + std::to_string(step)
             + "_" + std::to_string(mPIInfo.rank)
             + ".bin";
    filenameVElectron = directoryname + "/"
             + filenameWithoutStep + "_v_electron_" + std::to_string(step)
             + "_" + std::to_string(mPIInfo.rank)
             + ".bin";
    filenameNumIon = directoryname + "/"
             + filenameWithoutStep + "_num_ion_" + std::to_string(step)
             + "_" + std::to_string(mPIInfo.rank)
             + ".bin";
    filenameNumElectron = directoryname + "/"
             + filenameWithoutStep + "_num_electron_" + std::to_string(step)
             + "_" + std::to_string(mPIInfo.rank)
             + ".bin";
    filenameKineticEnergy = directoryname + "/"
             + filenameWithoutStep + "_KEnergy_" + std::to_string(step)
             + "_" + std::to_string(mPIInfo.rank)
             + ".bin";

    float x, y, z;
    float vx, vy, vz, KineticEnergy = 0.0f;

    std::ofstream ofsXIon(filenameXIon, std::ios::binary);
    ofsXIon << std::fixed << std::setprecision(6);
    std::ofstream ofsVIon(filenameVIon, std::ios::binary);
    ofsVIon << std::fixed << std::setprecision(6);
    for (unsigned long long i = 0; i < mPIInfo.existNumIonPerProcs; i++) {
        x = host_particlesIon[i].x;
        y = host_particlesIon[i].y;
        z = host_particlesIon[i].z;
        vx = host_particlesIon[i].vx / host_particlesIon[i].gamma;
        vy = host_particlesIon[i].vy / host_particlesIon[i].gamma;
        vz = host_particlesIon[i].vz / host_particlesIon[i].gamma;

        //if (mPIInfo.xminForProcs < x && x < mPIInfo.xmaxForProcs && mPIInfo.yminForProcs < y && y < mPIInfo.ymaxForProcs) {
            ofsXIon.write(reinterpret_cast<const char*>(&x), sizeof(float));
            ofsXIon.write(reinterpret_cast<const char*>(&y), sizeof(float));
            ofsXIon.write(reinterpret_cast<const char*>(&z), sizeof(float));

            ofsVIon.write(reinterpret_cast<const char*>(&vx), sizeof(float));
            ofsVIon.write(reinterpret_cast<const char*>(&vy), sizeof(float));
            ofsVIon.write(reinterpret_cast<const char*>(&vz), sizeof(float));

            KineticEnergy += (host_particlesIon[i].gamma - 1.0f) * PIC2DConst::mIon * pow(PIC2DConst::c, 2);
        //}
    }

    std::ofstream ofsXElectron(filenameXElectron, std::ios::binary);
    ofsXElectron << std::fixed << std::setprecision(6);
    std::ofstream ofsVElectron(filenameVElectron, std::ios::binary);
    ofsVElectron << std::fixed << std::setprecision(6);
    for (unsigned long long i = 0; i < mPIInfo.existNumElectronPerProcs; i++) {
        x = host_particlesElectron[i].x;
        y = host_particlesElectron[i].y;
        z = host_particlesElectron[i].z;
        vx = host_particlesElectron[i].vx / host_particlesElectron[i].gamma;
        vy = host_particlesElectron[i].vy / host_particlesElectron[i].gamma;
        vz = host_particlesElectron[i].vz / host_particlesElectron[i].gamma;

        //if (mPIInfo.xminForProcs < x && x < mPIInfo.xmaxForProcs && mPIInfo.yminForProcs < y && y < mPIInfo.ymaxForProcs) {
            ofsXElectron.write(reinterpret_cast<const char*>(&x), sizeof(float));
            ofsXElectron.write(reinterpret_cast<const char*>(&y), sizeof(float));
            ofsXElectron.write(reinterpret_cast<const char*>(&z), sizeof(float));

            ofsVElectron.write(reinterpret_cast<const char*>(&vx), sizeof(float));
            ofsVElectron.write(reinterpret_cast<const char*>(&vy), sizeof(float));
            ofsVElectron.write(reinterpret_cast<const char*>(&vz), sizeof(float));
            
            KineticEnergy += (host_particlesElectron[i].gamma - 1.0f) * PIC2DConst::mElectron * pow(PIC2DConst::c, 2);
        //}
    }

    std::ofstream ofsKineticEnergy(filenameKineticEnergy, std::ios::binary);
    ofsKineticEnergy << std::fixed << std::setprecision(6);
    ofsKineticEnergy.write(reinterpret_cast<const char*>(&KineticEnergy), sizeof(float));

    std::ofstream ofsNumIon(filenameNumIon, std::ios::binary);
    std::ofstream ofsNumElectron(filenameNumElectron, std::ios::binary);

    ofsNumIon.write(reinterpret_cast<const char*>(&mPIInfo.existNumIonPerProcs), sizeof(unsigned long long));
    ofsNumElectron.write(reinterpret_cast<const char*>(&mPIInfo.existNumElectronPerProcs), sizeof(unsigned long long));
}


//////////////////////////////////////////////////


thrust::device_vector<MagneticField>& PIC2D::getBRef()
{
    return B;
}


thrust::device_vector<Particle>& PIC2D::getParticlesIonRef()
{
    return particlesIon;
}


thrust::device_vector<Particle>& PIC2D::getParticlesElectronRef()
{
    return particlesElectron;
}

