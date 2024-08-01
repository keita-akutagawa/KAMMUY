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
        tmpB[j + device_ny_PIC * i].bX = 0.5f * (B[j + device_ny_PIC * i].bX + B[j - 1 + device_ny_PIC * i].bX);
        tmpB[j + device_ny_PIC * i].bY = 0.5f * (B[j + device_ny_PIC * i].bY + B[j + device_ny_PIC * (i - 1)].bY);
        tmpB[j + device_ny_PIC * i].bZ = 0.25f * (B[j + device_ny_PIC * i].bZ + B[j + device_ny_PIC * (i - 1)].bZ
                                   + B[j - 1 + device_ny_PIC * i].bZ + B[j - 1 + device_ny_PIC * (i - 1)].bZ);
        tmpE[j + device_ny_PIC * i].eX = 0.5f * (E[j + device_ny_PIC * i].eX + E[j + device_ny_PIC * (i - 1)].eX);
        tmpE[j + device_ny_PIC * i].eY = 0.5f * (E[j + device_ny_PIC * i].eY + E[j - 1 + device_ny_PIC * i].eY);
        tmpE[j + device_ny_PIC * i].eZ = E[j + device_ny_PIC * i].eZ;
    }

    if ((i == 0) && (0 < j) && (j < device_ny_PIC)) {
        tmpB[j + device_ny_PIC * i].bX = 0.5f * (B[j + device_ny_PIC * i].bX + B[j - 1 + device_ny_PIC * i].bX);
        tmpB[j + device_ny_PIC * i].bY = 0.5f * (B[j + device_ny_PIC * i].bY + B[j + device_ny_PIC * (device_nx_PIC - 1)].bY);
        tmpB[j + device_ny_PIC * i].bZ = 0.25f * (B[j + device_ny_PIC * i].bZ + B[j + device_ny_PIC * (device_nx_PIC - 1)].bZ
                                   + B[j - 1 + device_ny_PIC * i].bZ + B[j - 1 + device_ny_PIC * (device_nx_PIC - 1)].bZ);
        tmpE[j + device_ny_PIC * i].eX = 0.5f * (E[j + device_ny_PIC * i].eX + E[j + device_ny_PIC * (device_nx_PIC - 1)].eX);
        tmpE[j + device_ny_PIC * i].eY = 0.5f * (E[j + device_ny_PIC * i].eY + E[j - 1 + device_ny_PIC * i].eY);
        tmpE[j + device_ny_PIC * i].eZ = E[j + device_ny_PIC * i].eZ;
    }

    if ((0 < i) && (i < device_nx_PIC) && (j == 0)) {
        tmpB[j + device_ny_PIC * i].bX = 0.5f * (B[j + device_ny_PIC * i].bX + B[device_ny_PIC - 1 + device_ny_PIC * i].bX);
        tmpB[j + device_ny_PIC * i].bY = 0.5f * (B[j + device_ny_PIC * i].bY + B[j + device_ny_PIC * (i - 1)].bY);
        tmpB[j + device_ny_PIC * i].bZ = 0.25f * (B[j + device_ny_PIC * i].bZ + B[j + device_ny_PIC * (i - 1)].bZ
                                   + B[device_ny_PIC - 1 + device_ny_PIC * i].bZ + B[device_ny_PIC - 1 + device_ny_PIC * (i - 1)].bZ);
        tmpE[j + device_ny_PIC * i].eX = 0.5f * (E[j + device_ny_PIC * i].eX + E[j + device_ny_PIC * (i - 1)].eX);
        tmpE[j + device_ny_PIC * i].eY = 0.5f * (E[j + device_ny_PIC * i].eY + E[device_ny_PIC - 1 + device_ny_PIC * i].eY);
        tmpE[j + device_ny_PIC * i].eZ = E[j + device_ny_PIC * i].eZ;
    }

    if (i == 0 && j == 0) {
        tmpB[j + device_ny_PIC * i].bX = 0.5f * (B[j + device_ny_PIC * i].bX + B[device_ny_PIC - 1 + device_ny_PIC * i].bX);
        tmpB[j + device_ny_PIC * i].bY = 0.5f * (B[j + device_ny_PIC * i].bY + B[j + device_ny_PIC * (device_nx_PIC - 1)].bY);
        tmpB[j + device_ny_PIC * i].bZ = 0.25f * (B[j + device_ny_PIC * i].bZ + B[j + device_ny_PIC * (device_nx_PIC - 1)].bZ
                                   + B[device_ny_PIC - 1 + device_ny_PIC * i].bZ + B[device_ny_PIC - 1 + device_ny_PIC * (device_nx_PIC - 1)].bZ);
        tmpE[j + device_ny_PIC * i].eX = 0.5f * (E[j + device_ny_PIC * i].eX + E[j + device_ny_PIC * (device_nx_PIC - 1)].eX);
        tmpE[j + device_ny_PIC * i].eY = 0.5f * (E[j + device_ny_PIC * i].eY + E[device_ny_PIC - 1 + device_ny_PIC * i].eY);
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
        current[j + device_ny_PIC * i].jX = 0.5f * (tmpCurrent[j + device_ny_PIC * i].jX + tmpCurrent[j + device_ny_PIC * (i + 1)].jX);
        current[j + device_ny_PIC * i].jY = 0.5f * (tmpCurrent[j + device_ny_PIC * i].jY + tmpCurrent[j + 1 + device_ny_PIC * i].jY);
        current[j + device_ny_PIC * i].jZ = tmpCurrent[j + device_ny_PIC * i].jZ;
    }

    if (i == device_nx_PIC - 1 && j < device_ny_PIC - 1) {
        current[j + device_ny_PIC * i].jX = 0.5f * (tmpCurrent[j + device_ny_PIC * i].jX + tmpCurrent[j + device_ny_PIC * 0].jX);
        current[j + device_ny_PIC * i].jY = 0.5f * (tmpCurrent[j + device_ny_PIC * i].jY + tmpCurrent[j + 1 + device_ny_PIC * i].jY);
        current[j + device_ny_PIC * i].jZ = tmpCurrent[j + device_ny_PIC * i].jZ;
    }

    if (i < device_nx_PIC - 1 && j == device_ny_PIC - 1) {
        current[j + device_ny_PIC * i].jX = 0.5f * (tmpCurrent[j + device_ny_PIC * i].jX + tmpCurrent[j + device_ny_PIC * (i + 1)].jX);
        current[j + device_ny_PIC * i].jY = 0.5f * (tmpCurrent[j + device_ny_PIC * i].jY + tmpCurrent[0 + device_ny_PIC * i].jY);
        current[j + device_ny_PIC * i].jZ = tmpCurrent[j + device_ny_PIC * i].jZ;
    }

    if (i == device_nx_PIC - 1 && j == device_ny_PIC - 1) {
        current[j + device_ny_PIC * i].jX = 0.5f * (tmpCurrent[j + device_ny_PIC * i].jX + tmpCurrent[j + device_ny_PIC * 0].jX);
        current[j + device_ny_PIC * i].jY = 0.5f * (tmpCurrent[j + device_ny_PIC * i].jY + tmpCurrent[0 + device_ny_PIC * i].jY);
        current[j + device_ny_PIC * i].jZ = tmpCurrent[j + device_ny_PIC * i].jZ;
    }
}


void PIC2D::oneStepSymmetricXWallY()
{
    fieldSolver.timeEvolutionB(B, E, dt_PIC/2.0f);
    boundaryPIC.symmetricBoundaryBX(B);
    boundaryPIC.conductingWallBoundaryBY(B);
    
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
    boundaryPIC.symmetricBoundaryBX(tmpB);
    boundaryPIC.conductingWallBoundaryBY(tmpB);
    boundaryPIC.symmetricBoundaryEX(tmpE);
    boundaryPIC.conductingWallBoundaryEY(tmpE);


    particlePush.pushVelocity(
        particlesIon, particlesElectron, tmpB, tmpE, dt_PIC
    );


    particlePush.pushPosition(
        particlesIon, particlesElectron, dt_PIC/2.0f
    );
    boundaryPIC.conductingWallBoundaryParticleX(
        particlesIon, particlesElectron
    );
    boundaryPIC.conductingWallBoundaryParticleY(
        particlesIon, particlesElectron
    );


    currentCalculator.resetCurrent(tmpCurrent);
    currentCalculator.calculateCurrent(
        tmpCurrent, particlesIon, particlesElectron
    );
    boundaryPIC.symmetricBoundaryCurrentX(tmpCurrent);
    boundaryPIC.conductingWallBoundaryCurrentY(tmpCurrent);
    getHalfCurrent_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(current.data()), 
        thrust::raw_pointer_cast(tmpCurrent.data())
    );
    boundaryPIC.symmetricBoundaryCurrentX(current);
    boundaryPIC.conductingWallBoundaryCurrentY(current);

    fieldSolver.timeEvolutionB(B, E, dt_PIC/2.0f);
    boundaryPIC.symmetricBoundaryBX(B);
    boundaryPIC.conductingWallBoundaryBY(B);

    fieldSolver.timeEvolutionE(E, B, current, dt_PIC);
    boundaryPIC.symmetricBoundaryEX(E);
    boundaryPIC.conductingWallBoundaryEY(E);
    filter.langdonMarderTypeCorrection(E, particlesIon, particlesElectron, dt_PIC);
    boundaryPIC.symmetricBoundaryEX(E);
    boundaryPIC.conductingWallBoundaryEY(E);

    particlePush.pushPosition(
        particlesIon, particlesElectron, dt_PIC/2.0f
    );
    boundaryPIC.conductingWallBoundaryParticleX(
        particlesIon, particlesElectron
    );
    boundaryPIC.conductingWallBoundaryParticleY(
        particlesIon, particlesElectron
    );
}


void PIC2D::oneStepWallXFreeY()
{
    fieldSolver.timeEvolutionB(B, E, dt_PIC/2.0f);
    boundaryPIC.symmetricBoundaryBX(B);
    boundaryPIC.conductingWallBoundaryBY(B);
    
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
    boundaryPIC.symmetricBoundaryBX(tmpB);
    boundaryPIC.conductingWallBoundaryBY(tmpB);
    boundaryPIC.symmetricBoundaryEX(tmpE);
    boundaryPIC.conductingWallBoundaryEY(tmpE);
    
    particlePush.pushVelocity(
        particlesIon, particlesElectron, tmpB, tmpE, dt_PIC
    );
    
    particlePush.pushPosition(
        particlesIon, particlesElectron, dt_PIC/2.0f
    );
    boundaryPIC.conductingWallBoundaryParticleX(
        particlesIon, particlesElectron
    );
    boundaryPIC.openBoundaryParticleY(
        particlesIon, particlesElectron
    );

    currentCalculator.resetCurrent(tmpCurrent);
    currentCalculator.calculateCurrent(
        tmpCurrent, particlesIon, particlesElectron
    );
    boundaryPIC.symmetricBoundaryCurrentX(tmpCurrent);
    boundaryPIC.conductingWallBoundaryCurrentY(tmpCurrent);
    getHalfCurrent_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(current.data()), 
        thrust::raw_pointer_cast(tmpCurrent.data())
    );
    boundaryPIC.symmetricBoundaryCurrentX(current);
    boundaryPIC.conductingWallBoundaryCurrentY(current);

    fieldSolver.timeEvolutionB(B, E, dt_PIC/2.0f);
    boundaryPIC.symmetricBoundaryBX(B);
    boundaryPIC.conductingWallBoundaryBY(B);

    fieldSolver.timeEvolutionE(E, B, current, dt_PIC);
    boundaryPIC.symmetricBoundaryEX(E);
    boundaryPIC.conductingWallBoundaryEY(E);
    filter.langdonMarderTypeCorrection(E, particlesIon, particlesElectron, dt_PIC);
    boundaryPIC.symmetricBoundaryEX(E);
    boundaryPIC.conductingWallBoundaryEY(E);

    particlePush.pushPosition(
        particlesIon, particlesElectron, dt_PIC/2.0f
    );
    boundaryPIC.conductingWallBoundaryParticleX(
        particlesIon, particlesElectron
    );
    boundaryPIC.openBoundaryParticleY(
        particlesIon, particlesElectron
    );
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
    float BEnergy = 0.0f, EEnergy = 0.0f;

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
            ofsB.write(reinterpret_cast<const char*>(&host_B[j + ny_PIC * i].bX), sizeof(float));
            ofsB.write(reinterpret_cast<const char*>(&host_B[j + ny_PIC * i].bY), sizeof(float));
            ofsB.write(reinterpret_cast<const char*>(&host_B[j + ny_PIC * i].bZ), sizeof(float));
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
            ofsE.write(reinterpret_cast<const char*>(&host_E[j + ny_PIC * i].eX), sizeof(float));
            ofsE.write(reinterpret_cast<const char*>(&host_E[j + ny_PIC * i].eY), sizeof(float));
            ofsE.write(reinterpret_cast<const char*>(&host_E[j + ny_PIC * i].eZ), sizeof(float));
            EEnergy += host_E[j + ny_PIC * i].eX * host_E[j + ny_PIC * i].eX
                     + host_E[j + ny_PIC * i].eY * host_E[j + ny_PIC * i].eY
                     + host_E[j + ny_PIC * i].eZ * host_E[j + ny_PIC * i].eZ;
        }
    }
    EEnergy *= 0.5f * epsilon0_PIC;

    std::ofstream ofsCurrent(filenameCurrent, std::ios::binary);
    ofsCurrent << std::fixed << std::setprecision(6);
    for (int i = 0; i < nx_PIC; i++) {
        for (int j = 0; j < ny_PIC; j++) {
            ofsCurrent.write(reinterpret_cast<const char*>(&host_current[j + ny_PIC * i].jX), sizeof(float));
            ofsCurrent.write(reinterpret_cast<const char*>(&host_current[j + ny_PIC * i].jY), sizeof(float));
            ofsCurrent.write(reinterpret_cast<const char*>(&host_current[j + ny_PIC * i].jZ), sizeof(float));
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
                &host_zerothMomentIon[j + ny_PIC * i].n), sizeof(float)
            );
        }
    }

    std::ofstream ofsZerothMomentElectron(filenameZerothMomentElectron, std::ios::binary);
    ofsZerothMomentElectron << std::fixed << std::setprecision(6);
    for (int i = 0; i < nx_PIC; i++) {
        for (int j = 0; j < ny_PIC; j++) {
            ofsZerothMomentElectron.write(reinterpret_cast<const char*>(
                &host_zerothMomentElectron[j + ny_PIC * i].n), sizeof(float)
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
                &host_firstMomentIon[j + ny_PIC * i].x), sizeof(float)
            );
            ofsFirstMomentIon.write(reinterpret_cast<const char*>(
                &host_firstMomentIon[j + ny_PIC * i].y), sizeof(float)
            );
            ofsFirstMomentIon.write(reinterpret_cast<const char*>(
                &host_firstMomentIon[j + ny_PIC * i].z), sizeof(float)
            );
        }
    }

    std::ofstream ofsFirstMomentElectron(filenameFirstMomentElectron, std::ios::binary);
    ofsFirstMomentElectron << std::fixed << std::setprecision(6);
    for (int i = 0; i < nx_PIC; i++) {
        for (int j = 0; j < ny_PIC; j++) {
            ofsFirstMomentElectron.write(reinterpret_cast<const char*>(
                &host_firstMomentElectron[j + ny_PIC * i].x), sizeof(float)
            );
            ofsFirstMomentElectron.write(reinterpret_cast<const char*>(
                &host_firstMomentElectron[j + ny_PIC * i].y), sizeof(float)
            );
            ofsFirstMomentElectron.write(reinterpret_cast<const char*>(
                &host_firstMomentElectron[j + ny_PIC * i].z), sizeof(float)
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
                &host_secondMomentIon[j + ny_PIC * i].xx), sizeof(float)
            );
            ofsSecondMomentIon.write(reinterpret_cast<const char*>(
                &host_secondMomentIon[j + ny_PIC * i].yy), sizeof(float)
            );
            ofsSecondMomentIon.write(reinterpret_cast<const char*>(
                &host_secondMomentIon[j + ny_PIC * i].zz), sizeof(float)
            );
            ofsSecondMomentIon.write(reinterpret_cast<const char*>(
                &host_secondMomentIon[j + ny_PIC * i].xy), sizeof(float)
            );
            ofsSecondMomentIon.write(reinterpret_cast<const char*>(
                &host_secondMomentIon[j + ny_PIC * i].xz), sizeof(float)
            );
            ofsSecondMomentIon.write(reinterpret_cast<const char*>(
                &host_secondMomentIon[j + ny_PIC * i].yz), sizeof(float)
            );
        }
    }

    std::ofstream ofsSecondMomentElectron(filenameSecondMomentElectron, std::ios::binary);
    ofsSecondMomentElectron << std::fixed << std::setprecision(6);
    for (int i = 0; i < nx_PIC; i++) {
        for (int j = 0; j < ny_PIC; j++) {
            ofsSecondMomentElectron.write(reinterpret_cast<const char*>(
                &host_secondMomentElectron[j + ny_PIC * i].xx), sizeof(float)
            );
            ofsSecondMomentElectron.write(reinterpret_cast<const char*>(
                &host_secondMomentElectron[j + ny_PIC * i].yy), sizeof(float)
            );
            ofsSecondMomentElectron.write(reinterpret_cast<const char*>(
                &host_secondMomentElectron[j + ny_PIC * i].zz), sizeof(float)
            );
            ofsSecondMomentElectron.write(reinterpret_cast<const char*>(
                &host_secondMomentElectron[j + ny_PIC * i].xy), sizeof(float)
            );
            ofsSecondMomentElectron.write(reinterpret_cast<const char*>(
                &host_secondMomentElectron[j + ny_PIC * i].xz), sizeof(float)
            );
            ofsSecondMomentElectron.write(reinterpret_cast<const char*>(
                &host_secondMomentElectron[j + ny_PIC * i].yz), sizeof(float)
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
        ofsXIon.write(reinterpret_cast<const char*>(&host_particlesIon[i].x), sizeof(float));
        ofsXIon.write(reinterpret_cast<const char*>(&host_particlesIon[i].y), sizeof(float));
        ofsXIon.write(reinterpret_cast<const char*>(&host_particlesIon[i].z), sizeof(float));
    }

    std::ofstream ofsXElectron(filenameXElectron, std::ios::binary);
    ofsXElectron << std::fixed << std::setprecision(6);
    for (unsigned long long i = 0; i < existNumElectron_PIC; i++) {
        ofsXElectron.write(reinterpret_cast<const char*>(&host_particlesElectron[i].x), sizeof(float));
        ofsXElectron.write(reinterpret_cast<const char*>(&host_particlesElectron[i].y), sizeof(float));
        ofsXElectron.write(reinterpret_cast<const char*>(&host_particlesElectron[i].z), sizeof(float));
    }


    float vx, vy, vz, KineticEnergy = 0.0f;

    std::ofstream ofsVIon(filenameVIon, std::ios::binary);
    ofsVIon << std::fixed << std::setprecision(6);
    for (unsigned long long i = 0; i < existNumIon_PIC; i++) {
        vx = host_particlesIon[i].vx;
        vy = host_particlesIon[i].vy;
        vz = host_particlesIon[i].vz;

        ofsVIon.write(reinterpret_cast<const char*>(&vx), sizeof(float));
        ofsVIon.write(reinterpret_cast<const char*>(&vy), sizeof(float));
        ofsVIon.write(reinterpret_cast<const char*>(&vz), sizeof(float));

        KineticEnergy += 0.5f * mIon_PIC * (vx * vx + vy * vy + vz * vz);
    }

    std::ofstream ofsVElectron(filenameVElectron, std::ios::binary);
    ofsVElectron << std::fixed << std::setprecision(6);
    for (unsigned long long i = 0; i < existNumElectron_PIC; i++) {
        vx = host_particlesElectron[i].vx;
        vy = host_particlesElectron[i].vy;
        vz = host_particlesElectron[i].vz;

        ofsVElectron.write(reinterpret_cast<const char*>(&vx), sizeof(float));
        ofsVElectron.write(reinterpret_cast<const char*>(&vy), sizeof(float));
        ofsVElectron.write(reinterpret_cast<const char*>(&vz), sizeof(float));
        
        KineticEnergy += 0.5f * mElectron_PIC * (vx * vx + vy * vy + vz * vz);
    }

    std::ofstream ofsKineticEnergy(filenameKineticEnergy, std::ios::binary);
    ofsKineticEnergy << std::fixed << std::setprecision(6);
    ofsKineticEnergy.write(reinterpret_cast<const char*>(&KineticEnergy), sizeof(float));
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


