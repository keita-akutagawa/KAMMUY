#include "PIC2D.hpp"


PIC2D::PIC2D(
    nlohmann::json& constJSON, 
    nlohmann::json& gridJSON
)
    : pICConstParameter(constJSON), 
      pICGridParameter(gridJSON), 
      
      NX(pICGridParameter.NX), 
      NY(pICGridParameter.NY), 
      DX(pICGridParameter.DX), 
      DY(pICGridParameter.DY), 
      
      particlesIon     (pICConstParameter.TOTAL_NUM_ION), 
      particlesElectron(pICConstParameter.TOTAL_NUM_ELECTRON),
      E                   (NX * NY), 
      tmpE                (NX * NY), 
      B                   (NX * NY), 
      tmpB                (NX * NY), 
      current             (NX * NY), 
      tmpCurrent          (NX * NY), 
      zerothMomentIon     (NX * NY), 
      zerothMomentElectron(NX * NY), 
      firstMomentIon      (NX * NY), 
      firstMomentElectron (NX * NY), 
      secondMomentIon     (NX * NY), 
      secondMomentElectron(NX * NY), 

      host_particlesIon     (pICConstParameter.TOTAL_NUM_ION), 
      host_particlesElectron(pICConstParameter.TOTAL_NUM_ELECTRON), 
      host_E                   (NX * NY),  
      host_B                   (NX * NY),  
      host_current             (NX * NY),  
      host_zerothMomentIon     (NX * NY),  
      host_zerothMomentElectron(NX * NY),  
      host_firstMomentIon      (NX * NY),  
      host_firstMomentElectron (NX * NY),  
      host_secondMomentIon     (NX * NY),  
      host_secondMomentElectron(NX * NY), 

      initializeParticle(pICConstParameter, pICGridParameter), 
      particlePush(pICConstParameter, pICGridParameter), 
      fieldSolver(pICConstParameter, pICGridParameter), 
      currentCalculator(pICConstParameter, pICGridParameter), 
      momentCalculator(pICConstParameter, pICGridParameter), 
      filter(pICConstParameter, pICGridParameter), 
      pICBoundary(pICConstParameter, pICGridParameter)
{
}


__global__ void getCenterBE_kernel(
    const PICUnsignedInt NX, PICUnsignedInt NY, 
    const MagneticField* B, const ElectricField* E, 
    MagneticField* tmpB, ElectricField* tmpE
)
{
    PICUnsignedInt i = blockIdx.x * blockDim.x + threadIdx.x;
    PICUnsignedInt j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < NX && 0 < j && j < NY) {
        PICUnsignedLongLong index = getIndex<PICUnsignedLongLong>(i, j, NX, NY); 

        tmpB[index].bX = 0.5 * (B[index].bX + B[index - 1].bX);
        tmpB[index].bY = 0.5 * (B[index].bY + B[index - NY].bY);
        tmpB[index].bZ = 0.25 * (B[index].bZ + B[index - NY].bZ + B[index - 1].bZ + B[index - 1 - NY].bZ);
        tmpE[index].eX = 0.5 * (E[index].eX + E[index - NY].eX);
        tmpE[index].eY = 0.5 * (E[index].eY + E[index - 1].eY);
        tmpE[index].eZ = E[index].eZ;
    }
}

__global__ void getHalfCurrent_kernel(
    const PICUnsignedInt NX, PICUnsignedInt NY, 
    const CurrentField* tmpCurrent, 
    CurrentField* current
)
{
    PICUnsignedInt i = blockIdx.x * blockDim.x + threadIdx.x;
    PICUnsignedInt j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < NX - 1 && j < NY - 1) {
        PICUnsignedLongLong index = getIndex<PICUnsignedLongLong>(i, j, NX, NY); 

        current[index].jX = 0.5 * (tmpCurrent[index].jX + tmpCurrent[index + NY].jX);
        current[index].jY = 0.5 * (tmpCurrent[index].jY + tmpCurrent[index + 1].jY);
        current[index].jZ = tmpCurrent[index].jZ;
    }
}


void PIC2D::oneStep(
    Interface2D& interface2D, 
    const thrust::device_vector<MHDValue>& timeInterpolatedU, 
    const PICUnsignedLongLong& seedForReload
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (NY + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    fieldSolver.timeEvolutionB(E, pICConstParameter.DT / 2.0, B);
    //pICBoundary.freeBoundaryFieldX(B); 
    //pICBoundary.freeBoundaryFieldY(B); 
    interface2D.sendMHDtoPIC_B(timeInterpolatedU, B);
    filter.langdonMarderTypeCorrectionB(
        pICConstParameter.DT / 2.0, B
    );
    
    getCenterBE_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        NX, NY, 
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(E.data()), 
        thrust::raw_pointer_cast(tmpB.data()), 
        thrust::raw_pointer_cast(tmpE.data())
    );
    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess) {
        printf("Kernel launch failed at getCenterBE_kernel: %s\n", cudaGetErrorString(err1));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err1 = cudaDeviceSynchronize();
    if (err1 != cudaSuccess) {
        printf("Kernel execution failed at getCenterBE_kernel: %s\n", cudaGetErrorString(err1));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    //pICBoundary.freeBoundaryFieldX(tmpB); 
    //pICBoundary.freeBoundaryFieldY(tmpB); 
    interface2D.sendMHDtoPIC_E(timeInterpolatedU, tmpE);
    interface2D.sendMHDtoPIC_B(timeInterpolatedU, tmpB);

    particlePush.pushVelocity(
        tmpB, 
        tmpE, 
        pICConstParameter.DT, 
        particlesIon, 
        particlesElectron
    );

    particlePush.pushPosition(
        pICConstParameter.DT / 2.0, 
        particlesIon, 
        particlesElectron
    );
    interface2D.sendMHDtoPIC_particle(
        timeInterpolatedU, 
        zerothMomentIon, zerothMomentElectron, 
        firstMomentIon, firstMomentElectron, 
        secondMomentIon, secondMomentElectron, 
        seedForReload, 
        particlesIon, particlesElectron
    ); 

    //このタイミングx, vの時間が揃う
    calculateFullMoments(); 

    currentCalculator.calculateCurrent(
        particlesIon, particlesElectron, 
        firstMomentIon, firstMomentElectron, tmpCurrent
    );
    interface2D.sendMHDtoPIC_current(timeInterpolatedU, tmpCurrent);
    
    getHalfCurrent_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        NX, NY, 
        thrust::raw_pointer_cast(tmpCurrent.data()), 
        thrust::raw_pointer_cast(current.data())
    );
    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess) {
        printf("Kernel launch failed at calculateHalfCurrent_kernel: %s\n", cudaGetErrorString(err2));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err2 = cudaDeviceSynchronize();
    if (err2 != cudaSuccess) {
        printf("Kernel execution failed at calculateHalfCurrent_kernel: %s\n", cudaGetErrorString(err2));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    interface2D.sendMHDtoPIC_current(timeInterpolatedU, current);

    fieldSolver.timeEvolutionB(
        E, 
        pICConstParameter.DT / 2.0, 
        B
    );
    //pICBoundary.freeBoundaryFieldX(B); 
    //pICBoundary.freeBoundaryFieldY(B); 
    interface2D.sendMHDtoPIC_B(timeInterpolatedU, B);
    filter.langdonMarderTypeCorrectionB(
        pICConstParameter.DT / 2.0, B
    );

    fieldSolver.timeEvolutionE(
        B, 
        current, 
        pICConstParameter.DT, 
        E
    );
    interface2D.sendMHDtoPIC_E(timeInterpolatedU, E);

    particlePush.pushPosition(
        pICConstParameter.DT / 2.0, 
        particlesIon, 
        particlesElectron
    );
    interface2D.sendMHDtoPIC_particle(
        timeInterpolatedU, 
        zerothMomentIon, zerothMomentElectron, 
        firstMomentIon, firstMomentElectron, 
        secondMomentIon, secondMomentElectron, 
        seedForReload + 100000000, 
        particlesIon, particlesElectron
    ); 

    filter.calculateRho(
        particlesIon, particlesElectron, 
        zerothMomentIon, zerothMomentElectron
    ); 
    filter.langdonMarderTypeCorrectionE(
        pICConstParameter.DT, E
    );
}   


void PIC2D::saveFields()
{
    host_E = E;
    host_B = B;
    host_current = current;

    std::string filenameB = pICConstParameter.SAVE_DIRNAME + "/"
             + pICConstParameter.SAVE_FILENAME_WITHOUT_STEP + "_B_" + std::to_string(pICConstParameter.CURRENT_STEP) 
             + ".bin";
    std::string filenameE = pICConstParameter.SAVE_DIRNAME + "/"
             + pICConstParameter.SAVE_FILENAME_WITHOUT_STEP + "_E_" + std::to_string(pICConstParameter.CURRENT_STEP) 
             + ".bin";
    std::string filenameCurrent = pICConstParameter.SAVE_DIRNAME + "/"
             + pICConstParameter.SAVE_FILENAME_WITHOUT_STEP + "_current_" + std::to_string(pICConstParameter.CURRENT_STEP) 
             + ".bin";
    std::string filenameBEnergy = pICConstParameter.SAVE_DIRNAME + "/"
             + pICConstParameter.SAVE_FILENAME_WITHOUT_STEP + "_BEnergy_" + std::to_string(pICConstParameter.CURRENT_STEP) 
             + ".bin";
    std::string filenameEEnergy = pICConstParameter.SAVE_DIRNAME + "/"
             + pICConstParameter.SAVE_FILENAME_WITHOUT_STEP + "_EEnergy_" + std::to_string(pICConstParameter.CURRENT_STEP) 
             + ".bin";

    PICFloat BEnergy = 0.0, EEnergy = 0.0;

    std::ofstream ofsB(filenameB, std::ios::binary);
    ofsB << std::fixed << std::setprecision(6);
    for (PICUnsignedInt i = 0; i < NX; i++) {
        for (PICUnsignedInt j = 0; j < NY; j++) {
            PICUnsignedLongLong index = getIndex<PICUnsignedLongLong>(i, j, NX, NY);

            ofsB.write(reinterpret_cast<const char*>(&host_B[index].bX), sizeof(PICFloat));
            ofsB.write(reinterpret_cast<const char*>(&host_B[index].bY), sizeof(PICFloat));
            ofsB.write(reinterpret_cast<const char*>(&host_B[index].bZ), sizeof(PICFloat));
            BEnergy += host_B[index].bX * host_B[index].bX 
                     + host_B[index].bY * host_B[index].bY
                     + host_B[index].bZ * host_B[index].bZ;
        }
    }
    BEnergy *= 0.5 / pICConstParameter.MU0;

    std::ofstream ofsE(filenameE, std::ios::binary);
    ofsE << std::fixed << std::setprecision(6);
    for (PICUnsignedInt i = 0; i < NX; i++) {
        for (PICUnsignedInt j = 0; j < NY; j++) {
            PICUnsignedLongLong index = getIndex<PICUnsignedLongLong>(i, j, NX, NY);

            ofsE.write(reinterpret_cast<const char*>(&host_E[index].eX), sizeof(PICFloat));
            ofsE.write(reinterpret_cast<const char*>(&host_E[index].eY), sizeof(PICFloat));
            ofsE.write(reinterpret_cast<const char*>(&host_E[index].eZ), sizeof(PICFloat));
            EEnergy += host_E[index].eX * host_E[index].eX
                     + host_E[index].eY * host_E[index].eY
                     + host_E[index].eZ * host_E[index].eZ;
        }
    }
    EEnergy *= 0.5 * pICConstParameter.EPSILON0;

    std::ofstream ofsCurrent(filenameCurrent, std::ios::binary);
    ofsCurrent << std::fixed << std::setprecision(6);
    for (PICUnsignedInt i = 0; i < NX; i++) {
        for (PICUnsignedInt j = 0; j < NY; j++) {
            PICUnsignedLongLong index = getIndex<PICUnsignedLongLong>(i, j, NX, NY);

            ofsCurrent.write(reinterpret_cast<const char*>(&host_current[index].jX), sizeof(PICFloat));
            ofsCurrent.write(reinterpret_cast<const char*>(&host_current[index].jY), sizeof(PICFloat));
            ofsCurrent.write(reinterpret_cast<const char*>(&host_current[index].jZ), sizeof(PICFloat));
        }
    }

    std::ofstream ofsBEnergy(filenameBEnergy, std::ios::binary);
    ofsBEnergy << std::fixed << std::setprecision(6);
    ofsBEnergy.write(reinterpret_cast<const char*>(&BEnergy), sizeof(PICFloat));

    std::ofstream ofsEEnergy(filenameEEnergy, std::ios::binary);
    ofsEEnergy << std::fixed << std::setprecision(6);
    ofsEEnergy.write(reinterpret_cast<const char*>(&EEnergy), sizeof(PICFloat));
}


void PIC2D::calculateFullMoments()
{
    calculateZerothMoments();
    calculateFirstMoments();
    calculateSecondMoments();
}


void PIC2D::calculateZerothMoments()
{
    momentCalculator.calculateZerothMoment(
        particlesIon, pICConstParameter.EXIST_NUM_ION, zerothMomentIon
    );
    momentCalculator.calculateZerothMoment(
        particlesElectron, pICConstParameter.EXIST_NUM_ELECTRON, zerothMomentElectron
    );
}


void PIC2D::calculateFirstMoments()
{
    momentCalculator.calculateFirstMoment(
        particlesIon, pICConstParameter.EXIST_NUM_ION, firstMomentIon
    );
    momentCalculator.calculateFirstMoment(
        particlesElectron, pICConstParameter.EXIST_NUM_ELECTRON, firstMomentElectron
    );
}


void PIC2D::calculateSecondMoments()
{
    momentCalculator.calculateSecondMoment(
        particlesIon, pICConstParameter.EXIST_NUM_ION, secondMomentIon
    );
    momentCalculator.calculateSecondMoment(
        particlesElectron, pICConstParameter.EXIST_NUM_ELECTRON, secondMomentElectron
    );
}


void PIC2D::saveFullMoments()
{
    saveZerothMoments();
    saveFirstMoments();
    saveSecondMoments();
}


void PIC2D::saveZerothMoments()
{
    calculateZerothMoments();

    host_zerothMomentIon = zerothMomentIon;
    host_zerothMomentElectron = zerothMomentElectron;
    
    std::string filenameZerothMomentIon = pICConstParameter.SAVE_DIRNAME + "/"
                            + pICConstParameter.SAVE_FILENAME_WITHOUT_STEP + "_zeroth_moment_ion_" + std::to_string(pICConstParameter.CURRENT_STEP) 
                            + ".bin";
    std::string filenameZerothMomentElectron = pICConstParameter.SAVE_DIRNAME + "/"
                                 + pICConstParameter.SAVE_FILENAME_WITHOUT_STEP + "_zeroth_moment_electron_" + std::to_string(pICConstParameter.CURRENT_STEP)
                                 + ".bin";
    

    std::ofstream ofsZerothMomentIon(filenameZerothMomentIon, std::ios::binary);
    ofsZerothMomentIon << std::fixed << std::setprecision(6);
    for (PICUnsignedInt i = 0; i < NX; i++) {
        for (PICUnsignedInt j = 0; j < NY; j++) {
            PICUnsignedLongLong index = getIndex<PICUnsignedLongLong>(i, j, NX, NY);

            ofsZerothMomentIon.write(reinterpret_cast<const char*>(
                &host_zerothMomentIon[index].n), sizeof(PICFloat)
            );
        }
    }

    std::ofstream ofsZerothMomentElectron(filenameZerothMomentElectron, std::ios::binary);
    ofsZerothMomentElectron << std::fixed << std::setprecision(6);
    for (PICUnsignedInt i = 0; i < NX; i++) {
        for (PICUnsignedInt j = 0; j < NY; j++) {
            PICUnsignedLongLong index = getIndex<PICUnsignedLongLong>(i, j, NX, NY);

            ofsZerothMomentElectron.write(reinterpret_cast<const char*>(
                &host_zerothMomentElectron[index].n), sizeof(PICFloat)
            );
        }
    }
}


void PIC2D::saveFirstMoments()
{
    calculateFirstMoments();

    host_firstMomentIon = firstMomentIon;
    host_firstMomentElectron = firstMomentElectron;

    std::string filenameFirstMomentIon, filenameFirstMomentElectron;
    
    filenameFirstMomentIon = pICConstParameter.SAVE_DIRNAME + "/"
                           + pICConstParameter.SAVE_FILENAME_WITHOUT_STEP + "_first_moment_ion_" + std::to_string(pICConstParameter.CURRENT_STEP) 
                           + ".bin";
    filenameFirstMomentElectron = pICConstParameter.SAVE_DIRNAME + "/"
                                + pICConstParameter.SAVE_FILENAME_WITHOUT_STEP + "_first_moment_electron_" + std::to_string(pICConstParameter.CURRENT_STEP) 
                                + ".bin";
    

    std::ofstream ofsFirstMomentIon(filenameFirstMomentIon, std::ios::binary);
    ofsFirstMomentIon << std::fixed << std::setprecision(6);
    for (PICUnsignedInt i = 0; i < NX; i++) {
        for (PICUnsignedInt j = 0; j < NY; j++) {
            PICUnsignedLongLong index = getIndex<PICUnsignedLongLong>(i, j, NX, NY);

            ofsFirstMomentIon.write(reinterpret_cast<const char*>(
                &host_firstMomentIon[index].x), sizeof(PICFloat)
            );
            ofsFirstMomentIon.write(reinterpret_cast<const char*>(
                &host_firstMomentIon[index].y), sizeof(PICFloat)
            );
            ofsFirstMomentIon.write(reinterpret_cast<const char*>(
                &host_firstMomentIon[index].z), sizeof(PICFloat)
            );
        }
    }

    std::ofstream ofsFirstMomentElectron(filenameFirstMomentElectron, std::ios::binary);
    ofsFirstMomentElectron << std::fixed << std::setprecision(6);
    for (PICUnsignedInt i = 0; i < NX; i++) {
        for (PICUnsignedInt j = 0; j < NY; j++) {
            PICUnsignedLongLong index = getIndex<PICUnsignedLongLong>(i, j, NX, NY);

            ofsFirstMomentElectron.write(reinterpret_cast<const char*>(
                &host_firstMomentElectron[index].x), sizeof(PICFloat)
            );
            ofsFirstMomentElectron.write(reinterpret_cast<const char*>(
                &host_firstMomentElectron[index].y), sizeof(PICFloat)
            );
            ofsFirstMomentElectron.write(reinterpret_cast<const char*>(
                &host_firstMomentElectron[index].z), sizeof(PICFloat)
            );
        }
    }
}


void PIC2D::saveSecondMoments()
{
    calculateSecondMoments();

    host_secondMomentIon = secondMomentIon;
    host_secondMomentElectron = secondMomentElectron;

    std::string filenameSecondMomentIon, filenameSecondMomentElectron;

    filenameSecondMomentIon = pICConstParameter.SAVE_DIRNAME + "/"
                            + pICConstParameter.SAVE_FILENAME_WITHOUT_STEP + "_second_moment_ion_" + std::to_string(pICConstParameter.CURRENT_STEP) 
                            + ".bin";
    filenameSecondMomentElectron = pICConstParameter.SAVE_DIRNAME + "/"
                                 + pICConstParameter.SAVE_FILENAME_WITHOUT_STEP + "_second_moment_electron_" + std::to_string(pICConstParameter.CURRENT_STEP) 
                                 + ".bin";
    

    std::ofstream ofsSecondMomentIon(filenameSecondMomentIon, std::ios::binary);
    ofsSecondMomentIon << std::fixed << std::setprecision(6);
    for (PICUnsignedInt i = 0; i < NX; i++) {
        for (PICUnsignedInt j = 0; j < NY; j++) {
            PICUnsignedLongLong index = getIndex<PICUnsignedLongLong>(i, j, NX, NY);

            ofsSecondMomentIon.write(reinterpret_cast<const char*>(
                &host_secondMomentIon[index].xx), sizeof(PICFloat)
            );
            ofsSecondMomentIon.write(reinterpret_cast<const char*>(
                &host_secondMomentIon[index].yy), sizeof(PICFloat)
            );
            ofsSecondMomentIon.write(reinterpret_cast<const char*>(
                &host_secondMomentIon[index].zz), sizeof(PICFloat)
            );
            ofsSecondMomentIon.write(reinterpret_cast<const char*>(
                &host_secondMomentIon[index].xy), sizeof(PICFloat)
            );
            ofsSecondMomentIon.write(reinterpret_cast<const char*>(
                &host_secondMomentIon[index].xz), sizeof(PICFloat)
            );
            ofsSecondMomentIon.write(reinterpret_cast<const char*>(
                &host_secondMomentIon[index].yz), sizeof(PICFloat)
            );
        }
    }

    std::ofstream ofsSecondMomentElectron(filenameSecondMomentElectron, std::ios::binary);
    ofsSecondMomentElectron << std::fixed << std::setprecision(6);
    for (PICUnsignedInt i = 0; i < NX; i++) {
        for (PICUnsignedInt j = 0; j < NY; j++) {
            PICUnsignedLongLong index = getIndex<PICUnsignedLongLong>(i, j, NX, NY);

            ofsSecondMomentElectron.write(reinterpret_cast<const char*>(
                &host_secondMomentElectron[index].xx), sizeof(PICFloat)
            );
            ofsSecondMomentElectron.write(reinterpret_cast<const char*>(
                &host_secondMomentElectron[index].yy), sizeof(PICFloat)
            );
            ofsSecondMomentElectron.write(reinterpret_cast<const char*>(
                &host_secondMomentElectron[index].zz), sizeof(PICFloat)
            );
            ofsSecondMomentElectron.write(reinterpret_cast<const char*>(
                &host_secondMomentElectron[index].xy), sizeof(PICFloat)
            );
            ofsSecondMomentElectron.write(reinterpret_cast<const char*>(
                &host_secondMomentElectron[index].xz), sizeof(PICFloat)
            );
            ofsSecondMomentElectron.write(reinterpret_cast<const char*>(
                &host_secondMomentElectron[index].yz), sizeof(PICFloat)
            );
        }
    }
}


void PIC2D::saveParticle()
{
    host_particlesIon = particlesIon;
    host_particlesElectron = particlesElectron;

    std::string filenameXIon = pICConstParameter.SAVE_DIRNAME + "/"
             + pICConstParameter.SAVE_FILENAME_WITHOUT_STEP + "_x_ion_" + std::to_string(pICConstParameter.CURRENT_STEP) 
             + ".bin";
    std::string filenameXElectron = pICConstParameter.SAVE_DIRNAME + "/"
             + pICConstParameter.SAVE_FILENAME_WITHOUT_STEP + "_x_electron_" + std::to_string(pICConstParameter.CURRENT_STEP) 
             + ".bin";
    std::string filenameVIon = pICConstParameter.SAVE_DIRNAME + "/"
             + pICConstParameter.SAVE_FILENAME_WITHOUT_STEP + "_v_ion_" + std::to_string(pICConstParameter.CURRENT_STEP) 
             + ".bin";
    std::string filenameVElectron = pICConstParameter.SAVE_DIRNAME + "/"
             + pICConstParameter.SAVE_FILENAME_WITHOUT_STEP + "_v_electron_" + std::to_string(pICConstParameter.CURRENT_STEP) 
             + ".bin";
    std::string filenameNumIon = pICConstParameter.SAVE_DIRNAME + "/"
             + pICConstParameter.SAVE_FILENAME_WITHOUT_STEP + "_num_ion_" + std::to_string(pICConstParameter.CURRENT_STEP) 
             + ".bin";
    std::string filenameNumElectron = pICConstParameter.SAVE_DIRNAME + "/"
             + pICConstParameter.SAVE_FILENAME_WITHOUT_STEP + "_num_electron_" + std::to_string(pICConstParameter.CURRENT_STEP) 
             + ".bin";
    std::string filenameKineticEnergy = pICConstParameter.SAVE_DIRNAME + "/"
             + pICConstParameter.SAVE_FILENAME_WITHOUT_STEP + "_KEnergy_" + std::to_string(pICConstParameter.CURRENT_STEP) 
             + ".bin";


    PICFloat KineticEnergy = 0.0;

    std::ofstream ofsXIon(filenameXIon, std::ios::binary);
    ofsXIon << std::fixed << std::setprecision(6);
    std::ofstream ofsVIon(filenameVIon, std::ios::binary);
    ofsVIon << std::fixed << std::setprecision(6);
    for (PICUnsignedLongLong i = 0; i < pICConstParameter.EXIST_NUM_ION; i++) {
        PICFloat x = host_particlesIon[i].x;
        PICFloat y = host_particlesIon[i].y;
        PICFloat z = host_particlesIon[i].z;
        PICFloat vx = host_particlesIon[i].ux / host_particlesIon[i].gamma;
        PICFloat vy = host_particlesIon[i].uy / host_particlesIon[i].gamma;
        PICFloat vz = host_particlesIon[i].uz / host_particlesIon[i].gamma;

        ofsXIon.write(reinterpret_cast<const char*>(&x), sizeof(PICFloat));
        ofsXIon.write(reinterpret_cast<const char*>(&y), sizeof(PICFloat));
        ofsXIon.write(reinterpret_cast<const char*>(&z), sizeof(PICFloat));

        ofsVIon.write(reinterpret_cast<const char*>(&vx), sizeof(PICFloat));
        ofsVIon.write(reinterpret_cast<const char*>(&vy), sizeof(PICFloat));
        ofsVIon.write(reinterpret_cast<const char*>(&vz), sizeof(PICFloat));

        KineticEnergy += (host_particlesIon[i].gamma - 1.0) * pICConstParameter.M_ION * pow(pICConstParameter.C, 2);
    }

    std::ofstream ofsXElectron(filenameXElectron, std::ios::binary);
    ofsXElectron << std::fixed << std::setprecision(6);
    std::ofstream ofsVElectron(filenameVElectron, std::ios::binary);
    ofsVElectron << std::fixed << std::setprecision(6);
    for (PICUnsignedLongLong i = 0; i < pICConstParameter.EXIST_NUM_ELECTRON; i++) {
        PICFloat x = host_particlesElectron[i].x;
        PICFloat y = host_particlesElectron[i].y;
        PICFloat z = host_particlesElectron[i].z;
        PICFloat vx = host_particlesElectron[i].ux / host_particlesElectron[i].gamma;
        PICFloat vy = host_particlesElectron[i].uy / host_particlesElectron[i].gamma;
        PICFloat vz = host_particlesElectron[i].uz / host_particlesElectron[i].gamma;

        ofsXElectron.write(reinterpret_cast<const char*>(&x), sizeof(PICFloat));
        ofsXElectron.write(reinterpret_cast<const char*>(&y), sizeof(PICFloat));
        ofsXElectron.write(reinterpret_cast<const char*>(&z), sizeof(PICFloat));

        ofsVElectron.write(reinterpret_cast<const char*>(&vx), sizeof(PICFloat));
        ofsVElectron.write(reinterpret_cast<const char*>(&vy), sizeof(PICFloat));
        ofsVElectron.write(reinterpret_cast<const char*>(&vz), sizeof(PICFloat));
        
        KineticEnergy += (host_particlesElectron[i].gamma - 1.0) * pICConstParameter.M_ELECTRON * pow(pICConstParameter.C, 2);
    }

    std::ofstream ofsKineticEnergy(filenameKineticEnergy, std::ios::binary);
    ofsKineticEnergy << std::fixed << std::setprecision(6);
    ofsKineticEnergy.write(reinterpret_cast<const char*>(&KineticEnergy), sizeof(PICFloat));

    std::ofstream ofsNumIon(filenameNumIon, std::ios::binary);
    std::ofstream ofsNumElectron(filenameNumElectron, std::ios::binary);

    ofsNumIon.write(reinterpret_cast<const char*>(&pICConstParameter.EXIST_NUM_ION), sizeof(PICUnsignedLongLong));
    ofsNumElectron.write(reinterpret_cast<const char*>(&pICConstParameter.EXIST_NUM_ELECTRON), sizeof(PICUnsignedLongLong));
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


thrust::device_vector<SecondMoment>& PIC2D::getSecondMomentIonRef()
{
    return secondMomentIon; 
}


thrust::device_vector<SecondMoment>& PIC2D::getSecondMomentElectronRef()
{
    return secondMomentElectron; 
}


PICGridParameter& PIC2D::getPICGridParameterRef()
{
    return pICGridParameter;
}


PICConstParameter& PIC2D::getPICConstParameterRef()
{
    return pICConstParameter;
}

