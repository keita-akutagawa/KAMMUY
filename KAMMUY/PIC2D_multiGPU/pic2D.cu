#include "pic2D.hpp"


PIC2D::PIC2D(PIC2DMPI::MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo), 

      particlesIon     (mPIInfo.totalNumIonPerProcs), 
      particlesElectron(mPIInfo.totalNumElectronPerProcs),
      E                   (mPIInfo.localSizeX * PIC2DConst::ny), 
      tmpE                (mPIInfo.localSizeX * PIC2DConst::ny), 
      B                   (mPIInfo.localSizeX * PIC2DConst::ny), 
      tmpB                (mPIInfo.localSizeX * PIC2DConst::ny), 
      current             (mPIInfo.localSizeX * PIC2DConst::ny), 
      tmpCurrent          (mPIInfo.localSizeX * PIC2DConst::ny), 
      zerothMomentIon     (mPIInfo.localSizeX * PIC2DConst::ny), 
      zerothMomentElectron(mPIInfo.localSizeX * PIC2DConst::ny), 
      firstMomentIon      (mPIInfo.localSizeX * PIC2DConst::ny), 
      firstMomentElectron (mPIInfo.localSizeX * PIC2DConst::ny), 
      secondMomentIon     (mPIInfo.localSizeX * PIC2DConst::ny), 
      secondMomentElectron(mPIInfo.localSizeX * PIC2DConst::ny), 

      host_particlesIon     (mPIInfo.totalNumIonPerProcs), 
      host_particlesElectron(mPIInfo.totalNumElectronPerProcs), 
      host_E                   (mPIInfo.localSizeX * PIC2DConst::ny),  
      host_B                   (mPIInfo.localSizeX * PIC2DConst::ny),  
      host_current             (mPIInfo.localSizeX * PIC2DConst::ny),  
      host_zerothMomentIon     (mPIInfo.localSizeX * PIC2DConst::ny),  
      host_zerothMomentElectron(mPIInfo.localSizeX * PIC2DConst::ny),  
      host_firstMomentIon      (mPIInfo.localSizeX * PIC2DConst::ny),  
      host_firstMomentElectron (mPIInfo.localSizeX * PIC2DConst::ny),  
      host_secondMomentIon     (mPIInfo.localSizeX * PIC2DConst::ny),  
      host_secondMomentElectron(mPIInfo.localSizeX * PIC2DConst::ny), 

      initializeParticle(mPIInfo), 
      particlePush      (mPIInfo), 
      fieldSolver       (mPIInfo), 
      currentCalculator (mPIInfo), 
      boundaryPIC       (mPIInfo), 
      momentCalculator  (mPIInfo), 
      filter            (mPIInfo)
{

    cudaMalloc(&device_mPIInfo, sizeof(PIC2DMPI::MPIInfo));
    cudaMemcpy(device_mPIInfo, &mPIInfo, sizeof(PIC2DMPI::MPIInfo), cudaMemcpyHostToDevice);
    
}


__global__ void getCenterBE_kernel(
    MagneticField* tmpB, ElectricField* tmpE, 
    const MagneticField* B, const ElectricField* E, 
    int localSizeX
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < localSizeX) && (0 < j) && (j < PIC2DConst::device_ny)) {
        int index = j + PIC2DConst::device_ny * i; 

        tmpB[index].bX = 0.5f * (B[index].bX + B[index - 1].bX);
        tmpB[index].bY = 0.5f * (B[index].bY + B[index - PIC2DConst::device_ny].bY);
        tmpB[index].bZ = 0.25f * (B[index].bZ + B[index - PIC2DConst::device_ny].bZ
                                + B[index - 1].bZ + B[index - 1 - PIC2DConst::device_ny].bZ);
        tmpE[index].eX = 0.5f * (E[index].eX + E[index - PIC2DConst::device_ny].eX);
        tmpE[index].eY = 0.5f * (E[index].eY + E[index - 1].eY);
        tmpE[index].eZ = E[index].eZ;
    }
}

__global__ void getHalfCurrent_kernel(
    CurrentField* current, const CurrentField* tmpCurrent, 
    int localSizeX
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localSizeX - 1 && j < PIC2DConst::device_ny - 1) {
        int index = j + PIC2DConst::device_ny * i; 

        current[index].jX = 0.5f * (tmpCurrent[index].jX + tmpCurrent[index + PIC2DConst::device_ny].jX);
        current[index].jY = 0.5f * (tmpCurrent[index].jY + tmpCurrent[index + 1].jY);
        current[index].jZ = tmpCurrent[index].jZ;
    }
}


void PIC2D::oneStep_periodicXFreeY(
    Interface2D& interface2D, 
    thrust::device_vector<ConservationParameter>& U, 
    unsigned long long seedForReload
)
{
    MPI_Barrier(MPI_COMM_WORLD);
    
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (PIC2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);
                      
    fieldSolver.timeEvolutionB(B, E, PIC2DConst::dt / 2.0f);
    boundaryPIC.periodicBoundaryB_x(B);
    boundaryPIC.freeBoundaryB_y(B);
    
    getCenterBE_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpB.data()), 
        thrust::raw_pointer_cast(tmpE.data()), 
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(E.data()), 
        mPIInfo.localSizeX
    );
    cudaDeviceSynchronize();
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
    boundaryPIC.freeBoundaryParticle_y(
        particlesIon, particlesElectron
    );

    currentCalculator.calculateCurrent(
        tmpCurrent, firstMomentIon, firstMomentElectron,
        particlesIon, particlesElectron
    );
    boundaryPIC.periodicBoundaryCurrent_x(tmpCurrent);
    boundaryPIC.freeBoundaryCurrent_y(tmpCurrent);

    interface2D.sendMHDtoPIC_currentField_y(U, tmpCurrent);
    boundaryPIC.periodicBoundaryCurrent_x(tmpCurrent);
    boundaryPIC.freeBoundaryCurrent_y(tmpCurrent);
    
    getHalfCurrent_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(current.data()), 
        thrust::raw_pointer_cast(tmpCurrent.data()), 
        mPIInfo.localSizeX
    );
    boundaryPIC.periodicBoundaryCurrent_x(current);
    boundaryPIC.freeBoundaryCurrent_y(current);

    fieldSolver.timeEvolutionB(B, E, PIC2DConst::dt / 2.0f);
    boundaryPIC.periodicBoundaryB_x(B);
    boundaryPIC.freeBoundaryB_y(B);

    fieldSolver.timeEvolutionE(E, B, current, PIC2DConst::dt);
    boundaryPIC.periodicBoundaryE_x(E);
    boundaryPIC.freeBoundaryE_y(E);

    filter.calculateRho(
        zerothMomentIon, zerothMomentElectron, 
        particlesIon, particlesElectron
    ); 
    filter.langdonMarderTypeCorrection(E, PIC2DConst::dt);
    boundaryPIC.periodicBoundaryE_x(E);
    boundaryPIC.freeBoundaryE_y(E);

    particlePush.pushPosition(
        particlesIon, particlesElectron, PIC2DConst::dt / 2.0f
    );
    boundaryPIC.periodicBoundaryParticle_x(
        particlesIon, particlesElectron
    );
    boundaryPIC.freeBoundaryParticle_y(
        particlesIon, particlesElectron
    );

    //interface2D.sendMHDtoPIC_magneticField_y(U, B);
    //boundaryPIC.periodicBoundaryB_x(B);
    //boundaryPIC.freeBoundaryB_y(B);
    
    interface2D.sendMHDtoPIC_electricField_y(U, E);
    boundaryPIC.periodicBoundaryE_x(E);
    boundaryPIC.freeBoundaryE_y(E);
    
    boundaryPIC.periodicBoundaryZerothMoment_x(zerothMomentIon);
    boundaryPIC.freeBoundaryZerothMoment_y(zerothMomentIon);
    boundaryPIC.periodicBoundaryZerothMoment_x(zerothMomentElectron);
    boundaryPIC.freeBoundaryZerothMoment_y(zerothMomentElectron);
    boundaryPIC.periodicBoundaryFirstMoment_x(firstMomentIon);
    boundaryPIC.freeBoundaryFirstMoment_y(firstMomentIon);
    boundaryPIC.periodicBoundaryFirstMoment_x(firstMomentElectron);
    boundaryPIC.freeBoundaryFirstMoment_y(firstMomentElectron);
    interface2D.sendMHDtoPIC_particle(
        U, 
        zerothMomentIon, zerothMomentElectron, 
        firstMomentIon, firstMomentElectron, 
        particlesIon, particlesElectron, 
        seedForReload
    );
    boundaryPIC.periodicBoundaryParticle_x(
        particlesIon, particlesElectron
    );
    boundaryPIC.freeBoundaryParticle_y(
        particlesIon, particlesElectron
    );
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
    for (int i = 0; i < mPIInfo.localSizeX; i++) {
        for (int j = 0; j < PIC2DConst::ny; j++) {
            int index = j + PIC2DConst::ny * i;

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
    for (int i = 0; i < mPIInfo.localSizeX; i++) {
        for (int j = 0; j < PIC2DConst::ny; j++) {
            int index = j + PIC2DConst::ny * i;

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
    for (int i = 0; i < mPIInfo.localSizeX; i++) {
        for (int j = 0; j < PIC2DConst::ny; j++) {
            int index = j + PIC2DConst::ny * i;

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
    momentCalculator.calculateZerothMomentOfOneSpecies(
        zerothMomentIon, particlesIon, mPIInfo.existNumIonPerProcs
    );
    momentCalculator.calculateZerothMomentOfOneSpecies(
        zerothMomentElectron, particlesElectron, mPIInfo.existNumElectronPerProcs
    );
}


void PIC2D::calculateFirstMoments()
{
    momentCalculator.calculateFirstMomentOfOneSpecies(
        firstMomentIon, particlesIon, mPIInfo.existNumIonPerProcs
    );
    momentCalculator.calculateFirstMomentOfOneSpecies(
        firstMomentElectron, particlesElectron, mPIInfo.existNumElectronPerProcs
    );
}


void PIC2D::calculateSecondMoments()
{
    momentCalculator.calculateSecondMomentOfOneSpecies(
        secondMomentIon, particlesIon, mPIInfo.existNumIonPerProcs
    );
    momentCalculator.calculateSecondMomentOfOneSpecies(
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
    for (int i = 0; i < mPIInfo.localSizeX; i++) {
        for (int j = 0; j < PIC2DConst::ny; j++) {
            int index = j + PIC2DConst::ny * i;

            ofsZerothMomentIon.write(reinterpret_cast<const char*>(
                &host_zerothMomentIon[index].n), sizeof(float)
            );
        }
    }

    std::ofstream ofsZerothMomentElectron(filenameZerothMomentElectron, std::ios::binary);
    ofsZerothMomentElectron << std::fixed << std::setprecision(6);
    for (int i = 0; i < mPIInfo.localSizeX; i++) {
        for (int j = 0; j < PIC2DConst::ny; j++) {
            int index = j + PIC2DConst::ny * i;

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
    for (int i = 0; i < mPIInfo.localSizeX; i++) {
        for (int j = 0; j < PIC2DConst::ny; j++) {
            int index = j + PIC2DConst::ny * i;

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
    for (int i = 0; i < mPIInfo.localSizeX; i++) {
        for (int j = 0; j < PIC2DConst::ny; j++) {
            int index = j + PIC2DConst::ny * i;

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
    for (int i = 0; i < mPIInfo.localSizeX; i++) {
        for (int j = 0; j < PIC2DConst::ny; j++) {
            int index = j + PIC2DConst::ny * i;

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
    for (int i = 0; i < mPIInfo.localSizeX; i++) {
        for (int j = 0; j < PIC2DConst::ny; j++) {
            int index = j + PIC2DConst::ny * i;

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

        ofsXIon.write(reinterpret_cast<const char*>(&x), sizeof(float));
        ofsXIon.write(reinterpret_cast<const char*>(&y), sizeof(float));
        ofsXIon.write(reinterpret_cast<const char*>(&z), sizeof(float));

        ofsVIon.write(reinterpret_cast<const char*>(&vx), sizeof(float));
        ofsVIon.write(reinterpret_cast<const char*>(&vy), sizeof(float));
        ofsVIon.write(reinterpret_cast<const char*>(&vz), sizeof(float));

        KineticEnergy += (host_particlesIon[i].gamma - 1.0f) * PIC2DConst::mIon * pow(PIC2DConst::c, 2);
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

        ofsXElectron.write(reinterpret_cast<const char*>(&x), sizeof(float));
        ofsXElectron.write(reinterpret_cast<const char*>(&y), sizeof(float));
        ofsXElectron.write(reinterpret_cast<const char*>(&z), sizeof(float));

        ofsVElectron.write(reinterpret_cast<const char*>(&vx), sizeof(float));
        ofsVElectron.write(reinterpret_cast<const char*>(&vy), sizeof(float));
        ofsVElectron.write(reinterpret_cast<const char*>(&vz), sizeof(float));
        
        KineticEnergy += (host_particlesElectron[i].gamma - 1.0f) * PIC2DConst::mElectron * pow(PIC2DConst::c, 2);
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


thrust::host_vector<MagneticField>& PIC2D::getHostBRef()
{
    return host_B;
}


thrust::device_vector<MagneticField>& PIC2D::getBRef()
{
    return B;
}

thrust::device_vector<MagneticField>& PIC2D::getTmpBRef()
{
    return tmpB;
}


thrust::host_vector<ElectricField>& PIC2D::getHostERef()
{
    return host_E;
}


thrust::device_vector<ElectricField>& PIC2D::getERef()
{
    return E;
}


thrust::host_vector<Particle>& PIC2D::getHostParticlesIonRef()
{
    return host_particlesIon;
}


thrust::device_vector<Particle>& PIC2D::getParticlesIonRef()
{
    return particlesIon;
}


thrust::host_vector<Particle>& PIC2D::getHostParticlesElectronRef()
{
    return host_particlesElectron;
}


thrust::device_vector<Particle>& PIC2D::getParticlesElectronRef()
{
    return particlesElectron;
}


thrust::device_vector<ZerothMoment>& PIC2D::getZerothMomentIonRef()
{
    return zerothMomentIon; 
}


thrust::device_vector<ZerothMoment>& PIC2D::getZerothMomentElectronRef()
{
    return zerothMomentElectron;
}


thrust::device_vector<FirstMoment>& PIC2D::getFirstMomentIonRef()
{
    return firstMomentIon; 
}


thrust::device_vector<FirstMoment>& PIC2D::getFirstMomentElectronRef()
{
    return firstMomentElectron; 
}


BoundaryPIC& PIC2D::getBoundaryPICRef()
{
    return boundaryPIC; 
}

