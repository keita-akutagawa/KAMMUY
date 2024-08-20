#include <fstream>
#include <iomanip>
#include <iostream>
#include <cstdio>
#include "PIC2D.hpp"


using namespace PIC2DConst;

PIC2D::PIC2D()
    : particlesIon(totalNumIon_PIC), 
      particlesElectron(totalNumElectron_PIC), 
      E(nx_PIC * ny_PIC), 
      tmpE(nx_PIC * ny_PIC), 
      B(nx_PIC * ny_PIC), 
      tmpB(nx_PIC * ny_PIC), 
      current(nx_PIC * ny_PIC), 
      tmpCurrent(nx_PIC * ny_PIC), 
      zerothMomentIon(nx_PIC * ny_PIC), 
      zerothMomentElectron(nx_PIC * ny_PIC), 
      firstMomentIon(nx_PIC * ny_PIC), 
      firstMomentElectron(nx_PIC * ny_PIC), 
      secondMomentIon(nx_PIC * ny_PIC), 
      secondMomentElectron(nx_PIC * ny_PIC),

      host_particlesIon(totalNumIon_PIC), 
      host_particlesElectron(totalNumElectron_PIC), 
      host_E(nx_PIC * ny_PIC), 
      host_B(nx_PIC * ny_PIC), 
      host_current(nx_PIC * ny_PIC), 
      host_zerothMomentIon(nx_PIC * ny_PIC), 
      host_zerothMomentElectron(nx_PIC * ny_PIC), 
      host_firstMomentIon(nx_PIC * ny_PIC), 
      host_firstMomentElectron(nx_PIC * ny_PIC), 
      host_secondMomentIon(nx_PIC * ny_PIC), 
      host_secondMomentElectron(nx_PIC * ny_PIC)
{
}


__global__ void getCenterBE_kernel(
    MagneticField* tmpB, ElectricField* tmpE, 
    const MagneticField* B, const ElectricField* E
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < device_nx_PIC) && (0 < j) && (j < device_ny_PIC)) {
        tmpB[j + device_ny_PIC * i].bX = 0.5 * (B[j + device_ny_PIC * i].bX + B[j - 1 + device_ny_PIC * i].bX);
        tmpB[j + device_ny_PIC * i].bY = 0.5 * (B[j + device_ny_PIC * i].bY + B[j + device_ny_PIC * (i - 1)].bY);
        tmpB[j + device_ny_PIC * i].bZ = 0.25 * (B[j + device_ny_PIC * i].bZ + B[j + device_ny_PIC * (i - 1)].bZ
                                   + B[j - 1 + device_ny_PIC * i].bZ + B[j - 1 + device_ny_PIC * (i - 1)].bZ);
        tmpE[j + device_ny_PIC * i].eX = 0.5 * (E[j + device_ny_PIC * i].eX + E[j + device_ny_PIC * (i - 1)].eX);
        tmpE[j + device_ny_PIC * i].eY = 0.5 * (E[j + device_ny_PIC * i].eY + E[j - 1 + device_ny_PIC * i].eY);
        tmpE[j + device_ny_PIC * i].eZ = E[j + device_ny_PIC * i].eZ;
    }

    if ((i == 0) && (0 < j) && (j < device_ny_PIC)) {
        tmpB[j + device_ny_PIC * i].bX = 0.5 * (B[j + device_ny_PIC * i].bX + B[j - 1 + device_ny_PIC * i].bX);
        tmpB[j + device_ny_PIC * i].bY = 0.5 * (B[j + device_ny_PIC * i].bY + B[j + device_ny_PIC * (device_nx_PIC - 1)].bY);
        tmpB[j + device_ny_PIC * i].bZ = 0.25 * (B[j + device_ny_PIC * i].bZ + B[j + device_ny_PIC * (device_nx_PIC - 1)].bZ
                                   + B[j - 1 + device_ny_PIC * i].bZ + B[j - 1 + device_ny_PIC * (device_nx_PIC - 1)].bZ);
        tmpE[j + device_ny_PIC * i].eX = 0.5 * (E[j + device_ny_PIC * i].eX + E[j + device_ny_PIC * (device_nx_PIC - 1)].eX);
        tmpE[j + device_ny_PIC * i].eY = 0.5 * (E[j + device_ny_PIC * i].eY + E[j - 1 + device_ny_PIC * i].eY);
        tmpE[j + device_ny_PIC * i].eZ = E[j + device_ny_PIC * i].eZ;
    }

    if ((0 < i) && (i < device_nx_PIC) && (j == 0)) {
        tmpB[j + device_ny_PIC * i].bX = 0.5 * (B[j + device_ny_PIC * i].bX + B[device_ny_PIC - 1 + device_ny_PIC * i].bX);
        tmpB[j + device_ny_PIC * i].bY = 0.5 * (B[j + device_ny_PIC * i].bY + B[j + device_ny_PIC * (i - 1)].bY);
        tmpB[j + device_ny_PIC * i].bZ = 0.25 * (B[j + device_ny_PIC * i].bZ + B[j + device_ny_PIC * (i - 1)].bZ
                                   + B[device_ny_PIC - 1 + device_ny_PIC * i].bZ + B[device_ny_PIC - 1 + device_ny_PIC * (i - 1)].bZ);
        tmpE[j + device_ny_PIC * i].eX = 0.5 * (E[j + device_ny_PIC * i].eX + E[j + device_ny_PIC * (i - 1)].eX);
        tmpE[j + device_ny_PIC * i].eY = 0.5 * (E[j + device_ny_PIC * i].eY + E[device_ny_PIC - 1 + device_ny_PIC * i].eY);
        tmpE[j + device_ny_PIC * i].eZ = E[j + device_ny_PIC * i].eZ;
    }

    if (i == 0 && j == 0) {
        tmpB[j + device_ny_PIC * i].bX = 0.5 * (B[j + device_ny_PIC * i].bX + B[device_ny_PIC - 1 + device_ny_PIC * i].bX);
        tmpB[j + device_ny_PIC * i].bY = 0.5 * (B[j + device_ny_PIC * i].bY + B[j + device_ny_PIC * (device_nx_PIC - 1)].bY);
        tmpB[j + device_ny_PIC * i].bZ = 0.25 * (B[j + device_ny_PIC * i].bZ + B[j + device_ny_PIC * (device_nx_PIC - 1)].bZ
                                   + B[device_ny_PIC - 1 + device_ny_PIC * i].bZ + B[device_ny_PIC - 1 + device_ny_PIC * (device_nx_PIC - 1)].bZ);
        tmpE[j + device_ny_PIC * i].eX = 0.5 * (E[j + device_ny_PIC * i].eX + E[j + device_ny_PIC * (device_nx_PIC - 1)].eX);
        tmpE[j + device_ny_PIC * i].eY = 0.5 * (E[j + device_ny_PIC * i].eY + E[device_ny_PIC - 1 + device_ny_PIC * i].eY);
        tmpE[j + device_ny_PIC * i].eZ = E[j + device_ny_PIC * i].eZ;
    }
}

__global__ void getHalfCurrent_kernel(
    CurrentField* current, const CurrentField* tmpCurrent
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < device_nx_PIC - 1 && j < device_ny_PIC - 1) {
        current[j + device_ny_PIC * i].jX = 0.5 * (tmpCurrent[j + device_ny_PIC * i].jX + tmpCurrent[j + device_ny_PIC * (i + 1)].jX);
        current[j + device_ny_PIC * i].jY = 0.5 * (tmpCurrent[j + device_ny_PIC * i].jY + tmpCurrent[j + 1 + device_ny_PIC * i].jY);
        current[j + device_ny_PIC * i].jZ = tmpCurrent[j + device_ny_PIC * i].jZ;
    }

    if (i == device_nx_PIC - 1 && j < device_ny_PIC - 1) {
        current[j + device_ny_PIC * i].jX = 0.5 * (tmpCurrent[j + device_ny_PIC * i].jX + tmpCurrent[j + device_ny_PIC * 0].jX);
        current[j + device_ny_PIC * i].jY = 0.5 * (tmpCurrent[j + device_ny_PIC * i].jY + tmpCurrent[j + 1 + device_ny_PIC * i].jY);
        current[j + device_ny_PIC * i].jZ = tmpCurrent[j + device_ny_PIC * i].jZ;
    }

    if (i < device_nx_PIC - 1 && j == device_ny_PIC - 1) {
        current[j + device_ny_PIC * i].jX = 0.5 * (tmpCurrent[j + device_ny_PIC * i].jX + tmpCurrent[j + device_ny_PIC * (i + 1)].jX);
        current[j + device_ny_PIC * i].jY = 0.5 * (tmpCurrent[j + device_ny_PIC * i].jY + tmpCurrent[0 + device_ny_PIC * i].jY);
        current[j + device_ny_PIC * i].jZ = tmpCurrent[j + device_ny_PIC * i].jZ;
    }

    if (i == device_nx_PIC - 1 && j == device_ny_PIC - 1) {
        current[j + device_ny_PIC * i].jX = 0.5 * (tmpCurrent[j + device_ny_PIC * i].jX + tmpCurrent[j + device_ny_PIC * 0].jX);
        current[j + device_ny_PIC * i].jY = 0.5 * (tmpCurrent[j + device_ny_PIC * i].jY + tmpCurrent[0 + device_ny_PIC * i].jY);
        current[j + device_ny_PIC * i].jZ = tmpCurrent[j + device_ny_PIC * i].jZ;
    }
}


void PIC2D::oneStepPeriodicXFreeY(
    thrust::device_vector<ConservationParameter>& UPast_Lower, 
    thrust::device_vector<ConservationParameter>& UPast_Upper, 
    thrust::device_vector<ConservationParameter>& UNext_Lower, 
    thrust::device_vector<ConservationParameter>& UNext_Upper, 
    Interface2D& interface2D_Lower, 
    Interface2D& interface2D_Upper, 
    InterfaceNoiseRemover2D& interfaceNoiseRemover2D_Lower, 
    InterfaceNoiseRemover2D& interfaceNoiseRemover2D_Upper, 
    int step, int substep, int totalSubstep
)
{
    double mixingRatio = (totalSubstep - substep) / totalSubstep;
    thrust::device_vector<ConservationParameter>& USub_Lower = interface2D_Lower.calculateAndGetSubU(UPast_Lower, UNext_Lower, mixingRatio);
    thrust::device_vector<ConservationParameter>& USub_Upper = interface2D_Upper.calculateAndGetSubU(UPast_Upper, UNext_Upper, mixingRatio);


    fieldSolver.timeEvolutionB(B, E, dt_PIC/2.0);
    boundaryPIC.freeBoundaryBY(B);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny_PIC + threadsPerBlock.y - 1) / threadsPerBlock.y);
    getCenterBE_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpB.data()), 
        thrust::raw_pointer_cast(tmpE.data()), 
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(E.data())
    );
    cudaDeviceSynchronize();
    boundaryPIC.freeBoundaryBY(tmpB);
    boundaryPIC.freeBoundaryEY(tmpE);
    
    particlePush.pushVelocity(
        particlesIon, particlesElectron, tmpB, tmpE, dt_PIC
    );
    
    particlePush.pushPosition(
        particlesIon, particlesElectron, dt_PIC/2.0
    );
    boundaryPIC.periodicBoundaryParticleX(
        particlesIon, particlesElectron
    );
    boundaryPIC.openBoundaryParticleY(
        particlesIon, particlesElectron
    );

    currentCalculator.resetCurrent(tmpCurrent);
    currentCalculator.calculateCurrent(
        tmpCurrent, particlesIon, particlesElectron
    );
    boundaryPIC.freeBoundaryCurrentY(tmpCurrent);
    getHalfCurrent_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(current.data()), 
        thrust::raw_pointer_cast(tmpCurrent.data())
    );
    boundaryPIC.freeBoundaryCurrentY(current);
    interface2D_Lower.sendMHDtoPIC_currentField_yDirection(USub_Lower, current);
    interface2D_Upper.sendMHDtoPIC_currentField_yDirection(USub_Upper, current);
    boundaryPIC.freeBoundaryCurrentY(current);
    //interfaceNoiseRemover2D_Lower.convolve_lower_currentField(current);
    //interfaceNoiseRemover2D_Upper.convolve_upper_currentField(current);

    fieldSolver.timeEvolutionB(B, E, dt_PIC/2.0);
    boundaryPIC.freeBoundaryBY(B);

    fieldSolver.timeEvolutionE(E, B, current, dt_PIC);
    boundaryPIC.freeBoundaryEY(E);
    filter.langdonMarderTypeCorrection(E, particlesIon, particlesElectron, dt_PIC);
    boundaryPIC.freeBoundaryEY(E);

    particlePush.pushPosition(
        particlesIon, particlesElectron, dt_PIC/2.0
    );
    boundaryPIC.periodicBoundaryParticleX(
        particlesIon, particlesElectron
    );
    boundaryPIC.openBoundaryParticleY(
        particlesIon, particlesElectron
    );


    interface2D_Lower.sendMHDtoPIC_magneticField_yDirection(USub_Lower, B);
    interface2D_Lower.sendMHDtoPIC_electricField_yDirection(USub_Lower, E);
    interface2D_Lower.sendMHDtoPIC_particle(USub_Lower, particlesIon, particlesElectron, step * totalSubstep + substep);
    interface2D_Upper.sendMHDtoPIC_magneticField_yDirection(USub_Upper, B);
    interface2D_Upper.sendMHDtoPIC_electricField_yDirection(USub_Upper, E);
    interface2D_Upper.sendMHDtoPIC_particle(USub_Upper, particlesIon, particlesElectron, step * totalSubstep + substep);

    boundaryPIC.freeBoundaryBY(B);
    boundaryPIC.freeBoundaryEY(E);
    boundaryPIC.periodicBoundaryParticleX(particlesIon, particlesElectron);
    boundaryPIC.openBoundaryParticleY(particlesIon, particlesElectron);

    //interfaceNoiseRemover2D_Lower.convolve_lower_magneticField(B);
    //interfaceNoiseRemover2D_Upper.convolve_upper_magneticField(B);
    //interfaceNoiseRemover2D_Lower.convolve_lower_electricField(E); 
    //interfaceNoiseRemover2D_Upper.convolve_upper_electricField(E);
}


void PIC2D::oneStepFreeXFreeY(
    thrust::device_vector<ConservationParameter>& UPast_Lower, 
    thrust::device_vector<ConservationParameter>& UPast_Upper, 
    thrust::device_vector<ConservationParameter>& UNext_Lower, 
    thrust::device_vector<ConservationParameter>& UNext_Upper, 
    Interface2D& interface2D_Lower, 
    Interface2D& interface2D_Upper, 
    InterfaceNoiseRemover2D& interfaceNoiseRemover2D_Lower, 
    InterfaceNoiseRemover2D& interfaceNoiseRemover2D_Upper, 
    int step, int substep, int totalSubstep
)
{
    double mixingRatio = (totalSubstep - substep) / totalSubstep;
    thrust::device_vector<ConservationParameter>& USub_Lower = interface2D_Lower.calculateAndGetSubU(UPast_Lower, UNext_Lower, mixingRatio);
    thrust::device_vector<ConservationParameter>& USub_Upper = interface2D_Upper.calculateAndGetSubU(UPast_Upper, UNext_Upper, mixingRatio);


    fieldSolver.timeEvolutionB(B, E, dt_PIC/2.0);
    boundaryPIC.freeBoundaryBX(B);
    boundaryPIC.freeBoundaryBY(B);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny_PIC + threadsPerBlock.y - 1) / threadsPerBlock.y);
    getCenterBE_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpB.data()), 
        thrust::raw_pointer_cast(tmpE.data()), 
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(E.data())
    );
    cudaDeviceSynchronize();
    boundaryPIC.freeBoundaryBX(tmpB);
    boundaryPIC.freeBoundaryBY(tmpB);
    boundaryPIC.freeBoundaryEX(tmpE);
    boundaryPIC.freeBoundaryEY(tmpE);
    
    particlePush.pushVelocity(
        particlesIon, particlesElectron, tmpB, tmpE, dt_PIC
    );
    
    particlePush.pushPosition(
        particlesIon, particlesElectron, dt_PIC/2.0
    );
    boundaryPIC.openBoundaryParticleX(
        particlesIon, particlesElectron
    );
    boundaryPIC.openBoundaryParticleY(
        particlesIon, particlesElectron
    );

    currentCalculator.resetCurrent(tmpCurrent);
    currentCalculator.calculateCurrent(
        tmpCurrent, particlesIon, particlesElectron
    );
    boundaryPIC.freeBoundaryCurrentY(tmpCurrent);
    getHalfCurrent_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(current.data()), 
        thrust::raw_pointer_cast(tmpCurrent.data())
    );
    boundaryPIC.freeBoundaryCurrentX(current);
    boundaryPIC.freeBoundaryCurrentY(current);
    interface2D_Lower.sendMHDtoPIC_currentField_yDirection(USub_Lower, current);
    interface2D_Upper.sendMHDtoPIC_currentField_yDirection(USub_Upper, current);
    boundaryPIC.freeBoundaryCurrentX(current);
    boundaryPIC.freeBoundaryCurrentY(current);
    //interfaceNoiseRemover2D_Lower.convolve_lower_currentField(current);
    //interfaceNoiseRemover2D_Upper.convolve_upper_currentField(current);

    fieldSolver.timeEvolutionB(B, E, dt_PIC/2.0);
    boundaryPIC.freeBoundaryBX(B);
    boundaryPIC.freeBoundaryBY(B);

    fieldSolver.timeEvolutionE(E, B, current, dt_PIC);
    boundaryPIC.freeBoundaryEX(E);
    boundaryPIC.freeBoundaryEY(E);
    filter.langdonMarderTypeCorrection(E, particlesIon, particlesElectron, dt_PIC);
    boundaryPIC.freeBoundaryEX(E);
    boundaryPIC.freeBoundaryEY(E);

    particlePush.pushPosition(
        particlesIon, particlesElectron, dt_PIC/2.0
    );
    boundaryPIC.openBoundaryParticleX(
        particlesIon, particlesElectron
    );
    boundaryPIC.openBoundaryParticleY(
        particlesIon, particlesElectron
    );


    interface2D_Lower.sendMHDtoPIC_magneticField_yDirection(USub_Lower, B);
    interface2D_Lower.sendMHDtoPIC_electricField_yDirection(USub_Lower, E);
    interface2D_Lower.sendMHDtoPIC_particle(USub_Lower, particlesIon, particlesElectron, step * totalSubstep + substep);
    interface2D_Upper.sendMHDtoPIC_magneticField_yDirection(USub_Upper, B);
    interface2D_Upper.sendMHDtoPIC_electricField_yDirection(USub_Upper, E);
    interface2D_Upper.sendMHDtoPIC_particle(USub_Upper, particlesIon, particlesElectron, step * totalSubstep + substep);

    boundaryPIC.freeBoundaryBX(B);
    boundaryPIC.freeBoundaryBY(B);
    boundaryPIC.freeBoundaryEX(E);
    boundaryPIC.freeBoundaryEY(E);
    boundaryPIC.openBoundaryParticleX(particlesIon, particlesElectron);
    boundaryPIC.openBoundaryParticleY(particlesIon, particlesElectron);

    //interfaceNoiseRemover2D_Lower.convolve_lower_magneticField(B);
    //interfaceNoiseRemover2D_Upper.convolve_upper_magneticField(B);
    //interfaceNoiseRemover2D_Lower.convolve_lower_electricField(E); 
    //interfaceNoiseRemover2D_Upper.convolve_upper_electricField(E);
}


void PIC2D::sortParticle()
{
    particleSorter.sortParticle(particlesIon, particlesElectron);
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
    double BEnergy = 0.0, EEnergy = 0.0;

    filenameB = directoryname + "/"
             + filenameWithoutStep + "_B_" + std::to_string(step)
             + ".bin";
    filenameE = directoryname + "/"
             + filenameWithoutStep + "_E_" + std::to_string(step)
             + ".bin";
    filenameCurrent = directoryname + "/"
             + filenameWithoutStep + "_current_" + std::to_string(step)
             + ".bin";
    filenameBEnergy = directoryname + "/"
             + filenameWithoutStep + "_BEnergy_" + std::to_string(step)
             + ".bin";
    filenameEEnergy = directoryname + "/"
             + filenameWithoutStep + "_EEnergy_" + std::to_string(step)
             + ".bin";


    std::ofstream ofsB(filenameB, std::ios::binary);
    ofsB << std::fixed << std::setprecision(6);
    for (int i = 0; i < nx_PIC; i++) {
        for (int j = 0; j < ny_PIC; j++) {
            ofsB.write(reinterpret_cast<const char*>(&host_B[j + ny_PIC * i].bX), sizeof(double));
            ofsB.write(reinterpret_cast<const char*>(&host_B[j + ny_PIC * i].bY), sizeof(double));
            ofsB.write(reinterpret_cast<const char*>(&host_B[j + ny_PIC * i].bZ), sizeof(double));
            BEnergy += host_B[j + ny_PIC * i].bX * host_B[j + ny_PIC * i].bX 
                     + host_B[j + ny_PIC * i].bY * host_B[j + ny_PIC * i].bY
                     + host_B[j + ny_PIC * i].bZ * host_B[j + ny_PIC * i].bZ;
        }
    }
    BEnergy *= 0.5f / mu0_PIC;

    std::ofstream ofsE(filenameE, std::ios::binary);
    ofsE << std::fixed << std::setprecision(6);
    for (int i = 0; i < nx_PIC; i++) {
        for (int j = 0; j < ny_PIC; j++) {
            ofsE.write(reinterpret_cast<const char*>(&host_E[j + ny_PIC * i].eX), sizeof(double));
            ofsE.write(reinterpret_cast<const char*>(&host_E[j + ny_PIC * i].eY), sizeof(double));
            ofsE.write(reinterpret_cast<const char*>(&host_E[j + ny_PIC * i].eZ), sizeof(double));
            EEnergy += host_E[j + ny_PIC * i].eX * host_E[j + ny_PIC * i].eX
                     + host_E[j + ny_PIC * i].eY * host_E[j + ny_PIC * i].eY
                     + host_E[j + ny_PIC * i].eZ * host_E[j + ny_PIC * i].eZ;
        }
    }
    EEnergy *= 0.5 * epsilon0_PIC;

    std::ofstream ofsCurrent(filenameCurrent, std::ios::binary);
    ofsCurrent << std::fixed << std::setprecision(6);
    for (int i = 0; i < nx_PIC; i++) {
        for (int j = 0; j < ny_PIC; j++) {
            ofsCurrent.write(reinterpret_cast<const char*>(&host_current[j + ny_PIC * i].jX), sizeof(double));
            ofsCurrent.write(reinterpret_cast<const char*>(&host_current[j + ny_PIC * i].jY), sizeof(double));
            ofsCurrent.write(reinterpret_cast<const char*>(&host_current[j + ny_PIC * i].jZ), sizeof(double));
        }
    }

    std::ofstream ofsBEnergy(filenameBEnergy, std::ios::binary);
    ofsBEnergy << std::fixed << std::setprecision(6);
    ofsBEnergy.write(reinterpret_cast<const char*>(&BEnergy), sizeof(double));

    std::ofstream ofsEEnergy(filenameEEnergy, std::ios::binary);
    ofsEEnergy << std::fixed << std::setprecision(6);
    ofsEEnergy.write(reinterpret_cast<const char*>(&EEnergy), sizeof(double));
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
        zerothMomentIon, particlesIon, existNumIon_PIC
    );
    momentCalculater.calculateZerothMomentOfOneSpecies(
        zerothMomentElectron, particlesElectron, existNumElectron_PIC
    );
}


void PIC2D::calculateFirstMoments()
{
    momentCalculater.calculateFirstMomentOfOneSpecies(
        firstMomentIon, particlesIon, existNumIon_PIC
    );
    momentCalculater.calculateFirstMomentOfOneSpecies(
        firstMomentElectron, particlesElectron, existNumElectron_PIC
    );
}


void PIC2D::calculateSecondMoments()
{
    momentCalculater.calculateSecondMomentOfOneSpecies(
        secondMomentIon, particlesIon, existNumIon_PIC
    );
    momentCalculater.calculateSecondMomentOfOneSpecies(
        secondMomentElectron, particlesElectron, existNumElectron_PIC
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
                            + ".bin";
    filenameZerothMomentElectron = directoryname + "/"
                                 + filenameWithoutStep + "_zeroth_moment_electron_" + std::to_string(step)
                                 + ".bin";
    

    std::ofstream ofsZerothMomentIon(filenameZerothMomentIon, std::ios::binary);
    ofsZerothMomentIon << std::fixed << std::setprecision(6);
    for (int i = 0; i < nx_PIC; i++) {
        for (int j = 0; j < ny_PIC; j++) {
            ofsZerothMomentIon.write(reinterpret_cast<const char*>(
                &host_zerothMomentIon[j + ny_PIC * i].n), sizeof(double)
            );
        }
    }

    std::ofstream ofsZerothMomentElectron(filenameZerothMomentElectron, std::ios::binary);
    ofsZerothMomentElectron << std::fixed << std::setprecision(6);
    for (int i = 0; i < nx_PIC; i++) {
        for (int j = 0; j < ny_PIC; j++) {
            ofsZerothMomentElectron.write(reinterpret_cast<const char*>(
                &host_zerothMomentElectron[j + ny_PIC * i].n), sizeof(double)
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
                           + ".bin";
    filenameFirstMomentElectron = directoryname + "/"
                                + filenameWithoutStep + "_first_moment_electron_" + std::to_string(step)
                                + ".bin";
    

    std::ofstream ofsFirstMomentIon(filenameFirstMomentIon, std::ios::binary);
    ofsFirstMomentIon << std::fixed << std::setprecision(6);
    for (int i = 0; i < nx_PIC; i++) {
        for (int j = 0; j < ny_PIC; j++) {
            ofsFirstMomentIon.write(reinterpret_cast<const char*>(
                &host_firstMomentIon[j + ny_PIC * i].x), sizeof(double)
            );
            ofsFirstMomentIon.write(reinterpret_cast<const char*>(
                &host_firstMomentIon[j + ny_PIC * i].y), sizeof(double)
            );
            ofsFirstMomentIon.write(reinterpret_cast<const char*>(
                &host_firstMomentIon[j + ny_PIC * i].z), sizeof(double)
            );
        }
    }

    std::ofstream ofsFirstMomentElectron(filenameFirstMomentElectron, std::ios::binary);
    ofsFirstMomentElectron << std::fixed << std::setprecision(6);
    for (int i = 0; i < nx_PIC; i++) {
        for (int j = 0; j < ny_PIC; j++) {
            ofsFirstMomentElectron.write(reinterpret_cast<const char*>(
                &host_firstMomentElectron[j + ny_PIC * i].x), sizeof(double)
            );
            ofsFirstMomentElectron.write(reinterpret_cast<const char*>(
                &host_firstMomentElectron[j + ny_PIC * i].y), sizeof(double)
            );
            ofsFirstMomentElectron.write(reinterpret_cast<const char*>(
                &host_firstMomentElectron[j + ny_PIC * i].z), sizeof(double)
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
                            + ".bin";
    filenameSecondMomentElectron = directoryname + "/"
                                 + filenameWithoutStep + "_second_moment_electron_" + std::to_string(step)
                                 + ".bin";
    

    std::ofstream ofsSecondMomentIon(filenameSecondMomentIon, std::ios::binary);
    ofsSecondMomentIon << std::fixed << std::setprecision(6);
    for (int i = 0; i < nx_PIC; i++) {
        for (int j = 0; j < ny_PIC; j++) {
            ofsSecondMomentIon.write(reinterpret_cast<const char*>(
                &host_secondMomentIon[j + ny_PIC * i].xx), sizeof(double)
            );
            ofsSecondMomentIon.write(reinterpret_cast<const char*>(
                &host_secondMomentIon[j + ny_PIC * i].yy), sizeof(double)
            );
            ofsSecondMomentIon.write(reinterpret_cast<const char*>(
                &host_secondMomentIon[j + ny_PIC * i].zz), sizeof(double)
            );
            ofsSecondMomentIon.write(reinterpret_cast<const char*>(
                &host_secondMomentIon[j + ny_PIC * i].xy), sizeof(double)
            );
            ofsSecondMomentIon.write(reinterpret_cast<const char*>(
                &host_secondMomentIon[j + ny_PIC * i].xz), sizeof(double)
            );
            ofsSecondMomentIon.write(reinterpret_cast<const char*>(
                &host_secondMomentIon[j + ny_PIC * i].yz), sizeof(double)
            );
        }
    }

    std::ofstream ofsSecondMomentElectron(filenameSecondMomentElectron, std::ios::binary);
    ofsSecondMomentElectron << std::fixed << std::setprecision(6);
    for (int i = 0; i < nx_PIC; i++) {
        for (int j = 0; j < ny_PIC; j++) {
            ofsSecondMomentElectron.write(reinterpret_cast<const char*>(
                &host_secondMomentElectron[j + ny_PIC * i].xx), sizeof(double)
            );
            ofsSecondMomentElectron.write(reinterpret_cast<const char*>(
                &host_secondMomentElectron[j + ny_PIC * i].yy), sizeof(double)
            );
            ofsSecondMomentElectron.write(reinterpret_cast<const char*>(
                &host_secondMomentElectron[j + ny_PIC * i].zz), sizeof(double)
            );
            ofsSecondMomentElectron.write(reinterpret_cast<const char*>(
                &host_secondMomentElectron[j + ny_PIC * i].xy), sizeof(double)
            );
            ofsSecondMomentElectron.write(reinterpret_cast<const char*>(
                &host_secondMomentElectron[j + ny_PIC * i].xz), sizeof(double)
            );
            ofsSecondMomentElectron.write(reinterpret_cast<const char*>(
                &host_secondMomentElectron[j + ny_PIC * i].yz), sizeof(double)
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
    std::string filenameKineticEnergy;

    filenameXIon = directoryname + "/"
             + filenameWithoutStep + "_x_ion_" + std::to_string(step)
             + ".bin";
    filenameXElectron = directoryname + "/"
             + filenameWithoutStep + "_x_electron_" + std::to_string(step)
             + ".bin";
    filenameVIon = directoryname + "/"
             + filenameWithoutStep + "_v_ion_" + std::to_string(step)
             + ".bin";
    filenameVElectron = directoryname + "/"
             + filenameWithoutStep + "_v_electron_" + std::to_string(step)
             + ".bin";
    filenameKineticEnergy = directoryname + "/"
             + filenameWithoutStep + "_KE_" + std::to_string(step)
             + ".bin";


    std::ofstream ofsXIon(filenameXIon, std::ios::binary);
    ofsXIon << std::fixed << std::setprecision(6);
    for (unsigned long long i = 0; i < existNumIon_PIC; i++) {
        ofsXIon.write(reinterpret_cast<const char*>(&host_particlesIon[i].x), sizeof(double));
        ofsXIon.write(reinterpret_cast<const char*>(&host_particlesIon[i].y), sizeof(double));
        ofsXIon.write(reinterpret_cast<const char*>(&host_particlesIon[i].z), sizeof(double));
    }

    std::ofstream ofsXElectron(filenameXElectron, std::ios::binary);
    ofsXElectron << std::fixed << std::setprecision(6);
    for (unsigned long long i = 0; i < existNumElectron_PIC; i++) {
        ofsXElectron.write(reinterpret_cast<const char*>(&host_particlesElectron[i].x), sizeof(double));
        ofsXElectron.write(reinterpret_cast<const char*>(&host_particlesElectron[i].y), sizeof(double));
        ofsXElectron.write(reinterpret_cast<const char*>(&host_particlesElectron[i].z), sizeof(double));
    }


    double vx, vy, vz, KineticEnergy = 0.0;

    std::ofstream ofsVIon(filenameVIon, std::ios::binary);
    ofsVIon << std::fixed << std::setprecision(6);
    for (unsigned long long i = 0; i < existNumIon_PIC; i++) {
        vx = host_particlesIon[i].vx;
        vy = host_particlesIon[i].vy;
        vz = host_particlesIon[i].vz;

        ofsVIon.write(reinterpret_cast<const char*>(&vx), sizeof(double));
        ofsVIon.write(reinterpret_cast<const char*>(&vy), sizeof(double));
        ofsVIon.write(reinterpret_cast<const char*>(&vz), sizeof(double));

        KineticEnergy += 0.5 * mIon_PIC * (vx * vx + vy * vy + vz * vz);
    }

    std::ofstream ofsVElectron(filenameVElectron, std::ios::binary);
    ofsVElectron << std::fixed << std::setprecision(6);
    for (unsigned long long i = 0; i < existNumElectron_PIC; i++) {
        vx = host_particlesElectron[i].vx;
        vy = host_particlesElectron[i].vy;
        vz = host_particlesElectron[i].vz;

        ofsVElectron.write(reinterpret_cast<const char*>(&vx), sizeof(double));
        ofsVElectron.write(reinterpret_cast<const char*>(&vy), sizeof(double));
        ofsVElectron.write(reinterpret_cast<const char*>(&vz), sizeof(double));
        
        KineticEnergy += 0.5 * mElectron_PIC * (vx * vx + vy * vy + vz * vz);
    }

    std::ofstream ofsKineticEnergy(filenameKineticEnergy, std::ios::binary);
    ofsKineticEnergy << std::fixed << std::setprecision(6);
    ofsKineticEnergy.write(reinterpret_cast<const char*>(&KineticEnergy), sizeof(double));
}


//////////////////////////////////////////////////


thrust::device_vector<MagneticField>& PIC2D::getBRef()
{
    return B;
}


thrust::device_vector<ElectricField>& PIC2D::getERef()
{
    return E;
}


thrust::device_vector<CurrentField>& PIC2D::getCurrentRef()
{
    return current;
}


thrust::device_vector<Particle>& PIC2D::getParticlesIonRef()
{
    return particlesIon;
}


thrust::device_vector<Particle>& PIC2D::getParticlesElectronRef()
{
    return particlesElectron;
}


