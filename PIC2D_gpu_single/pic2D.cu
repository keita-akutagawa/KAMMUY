#include <fstream>
#include <iomanip>
#include <iostream>
#include <cstdio>
#include "PIC2D.hpp"


using namespace PIC2DConst;

PIC2D::PIC2D()
    : particlesIon(totalNumIon), 
      particlesElectron(totalNumElectron), 
      E(nx * ny), 
      tmpE(nx * ny), 
      B(nx * ny), 
      tmpB(nx * ny), 
      current(nx * ny), 
      tmpCurrent(nx * ny), 
      zerothMomentIon(nx * ny), 
      zerothMomentElectron(nx * ny), 
      firstMomentIon(nx * ny), 
      firstMomentElectron(nx * ny), 
      secondMomentIon(nx * ny), 
      secondMomentElectron(nx * ny),

      host_particlesIon(totalNumIon), 
      host_particlesElectron(totalNumElectron), 
      host_E(nx * ny), 
      host_B(nx * ny), 
      host_current(nx * ny), 
      host_zerothMomentIon(nx * ny), 
      host_zerothMomentElectron(nx * ny), 
      host_firstMomentIon(nx * ny), 
      host_firstMomentElectron(nx * ny), 
      host_secondMomentIon(nx * ny), 
      host_secondMomentElectron(nx * ny)
{
}


__global__ void getCenterBE_kernel(
    MagneticField* tmpB, ElectricField* tmpE, 
    const MagneticField* B, const ElectricField* E
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < device_nx) && (0 < j) && (j < device_ny)) {
        tmpB[j + device_ny * i].bX = 0.5f * (B[j + device_ny * i].bX + B[j - 1 + device_ny * i].bX);
        tmpB[j + device_ny * i].bY = 0.5f * (B[j + device_ny * i].bY + B[j + device_ny * (i - 1)].bY);
        tmpB[j + device_ny * i].bZ = 0.25f * (B[j + device_ny * i].bZ + B[j + device_ny * (i - 1)].bZ
                                   + B[j - 1 + device_ny * i].bZ + B[j - 1 + device_ny * (i - 1)].bZ);
        tmpE[j + device_ny * i].eX = 0.5f * (E[j + device_ny * i].eX + E[j + device_ny * (i - 1)].eX);
        tmpE[j + device_ny * i].eY = 0.5f * (E[j + device_ny * i].eY + E[j - 1 + device_ny * i].eY);
        tmpE[j + device_ny * i].eZ = E[j + device_ny * i].eZ;
    }

    if ((i == 0) && (0 < j) && (j < device_ny)) {
        tmpB[j + device_ny * i].bX = 0.5f * (B[j + device_ny * i].bX + B[j - 1 + device_ny * i].bX);
        tmpB[j + device_ny * i].bY = 0.5f * (B[j + device_ny * i].bY + B[j + device_ny * (device_nx - 1)].bY);
        tmpB[j + device_ny * i].bZ = 0.25f * (B[j + device_ny * i].bZ + B[j + device_ny * (device_nx - 1)].bZ
                                   + B[j - 1 + device_ny * i].bZ + B[j - 1 + device_ny * (device_nx - 1)].bZ);
        tmpE[j + device_ny * i].eX = 0.5f * (E[j + device_ny * i].eX + E[j + device_ny * (device_nx - 1)].eX);
        tmpE[j + device_ny * i].eY = 0.5f * (E[j + device_ny * i].eY + E[j - 1 + device_ny * i].eY);
        tmpE[j + device_ny * i].eZ = E[j + device_ny * i].eZ;
    }

    if ((0 < i) && (i < device_nx) && (j == 0)) {
        tmpB[j + device_ny * i].bX = 0.5f * (B[j + device_ny * i].bX + B[device_ny - 1 + device_ny * i].bX);
        tmpB[j + device_ny * i].bY = 0.5f * (B[j + device_ny * i].bY + B[j + device_ny * (i - 1)].bY);
        tmpB[j + device_ny * i].bZ = 0.25f * (B[j + device_ny * i].bZ + B[j + device_ny * (i - 1)].bZ
                                   + B[device_ny - 1 + device_ny * i].bZ + B[device_ny - 1 + device_ny * (i - 1)].bZ);
        tmpE[j + device_ny * i].eX = 0.5f * (E[j + device_ny * i].eX + E[j + device_ny * (i - 1)].eX);
        tmpE[j + device_ny * i].eY = 0.5f * (E[j + device_ny * i].eY + E[device_ny - 1 + device_ny * i].eY);
        tmpE[j + device_ny * i].eZ = E[j + device_ny * i].eZ;
    }

    if (i == 0 && j == 0) {
        tmpB[j + device_ny * i].bX = 0.5f * (B[j + device_ny * i].bX + B[device_ny - 1 + device_ny * i].bX);
        tmpB[j + device_ny * i].bY = 0.5f * (B[j + device_ny * i].bY + B[j + device_ny * (device_nx - 1)].bY);
        tmpB[j + device_ny * i].bZ = 0.25f * (B[j + device_ny * i].bZ + B[j + device_ny * (device_nx - 1)].bZ
                                   + B[device_ny - 1 + device_ny * i].bZ + B[device_ny - 1 + device_ny * (device_nx - 1)].bZ);
        tmpE[j + device_ny * i].eX = 0.5f * (E[j + device_ny * i].eX + E[j + device_ny * (device_nx - 1)].eX);
        tmpE[j + device_ny * i].eY = 0.5f * (E[j + device_ny * i].eY + E[device_ny - 1 + device_ny * i].eY);
        tmpE[j + device_ny * i].eZ = E[j + device_ny * i].eZ;
    }
}

__global__ void getHalfCurrent_kernel(
    CurrentField* current, const CurrentField* tmpCurrent
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < device_nx - 1 && j < device_ny - 1) {
        current[j + device_ny * i].jX = 0.5f * (tmpCurrent[j + device_ny * i].jX + tmpCurrent[j + device_ny * (i + 1)].jX);
        current[j + device_ny * i].jY = 0.5f * (tmpCurrent[j + device_ny * i].jY + tmpCurrent[j + 1 + device_ny * i].jY);
        current[j + device_ny * i].jZ = tmpCurrent[j + device_ny * i].jZ;
    }

    if (i == device_nx - 1 && j < device_ny - 1) {
        current[j + device_ny * i].jX = 0.5f * (tmpCurrent[j + device_ny * i].jX + tmpCurrent[j + device_ny * 0].jX);
        current[j + device_ny * i].jY = 0.5f * (tmpCurrent[j + device_ny * i].jY + tmpCurrent[j + 1 + device_ny * i].jY);
        current[j + device_ny * i].jZ = tmpCurrent[j + device_ny * i].jZ;
    }

    if (i < device_nx - 1 && j == device_ny - 1) {
        current[j + device_ny * i].jX = 0.5f * (tmpCurrent[j + device_ny * i].jX + tmpCurrent[j + device_ny * (i + 1)].jX);
        current[j + device_ny * i].jY = 0.5f * (tmpCurrent[j + device_ny * i].jY + tmpCurrent[0 + device_ny * i].jY);
        current[j + device_ny * i].jZ = tmpCurrent[j + device_ny * i].jZ;
    }

    if (i == device_nx - 1 && j == device_ny - 1) {
        current[j + device_ny * i].jX = 0.5f * (tmpCurrent[j + device_ny * i].jX + tmpCurrent[j + device_ny * 0].jX);
        current[j + device_ny * i].jY = 0.5f * (tmpCurrent[j + device_ny * i].jY + tmpCurrent[0 + device_ny * i].jY);
        current[j + device_ny * i].jZ = tmpCurrent[j + device_ny * i].jZ;
    }
}


void PIC2D::oneStepPeriodicXY()
{
    fieldSolver.timeEvolutionB(B, E, dt/2.0f);
    boundary.periodicBoundaryBX(B);
    boundary.periodicBoundaryBY(B);
    
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);
    getCenterBE_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpB.data()), 
        thrust::raw_pointer_cast(tmpE.data()), 
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(E.data())
    );
    cudaDeviceSynchronize();
    boundary.periodicBoundaryBX(tmpB);
    boundary.periodicBoundaryBY(tmpB);
    boundary.periodicBoundaryEX(tmpE);
    boundary.periodicBoundaryEY(tmpE);


    particlePush.pushVelocity(
        particlesIon, particlesElectron, tmpB, tmpE, dt
    );


    particlePush.pushPosition(
        particlesIon, particlesElectron, dt/2.0f
    );
    boundary.periodicBoundaryParticleX(
        particlesIon, particlesElectron
    );
    boundary.periodicBoundaryParticleY(
        particlesIon, particlesElectron
    );


    currentCalculator.resetCurrent(tmpCurrent);
    currentCalculator.calculateCurrent(
        tmpCurrent, particlesIon, particlesElectron
    );
    boundary.periodicBoundaryCurrentX(tmpCurrent);
    boundary.periodicBoundaryCurrentY(tmpCurrent);
    getHalfCurrent_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(current.data()), 
        thrust::raw_pointer_cast(tmpCurrent.data())
    );
    boundary.periodicBoundaryCurrentX(current);
    boundary.periodicBoundaryCurrentY(current);

    fieldSolver.timeEvolutionB(B, E, dt/2.0f);
    boundary.periodicBoundaryBX(B);
    boundary.periodicBoundaryBY(B);

    fieldSolver.timeEvolutionE(E, B, current, dt);
    boundary.periodicBoundaryEX(E);
    boundary.periodicBoundaryEY(E);

    particlePush.pushPosition(
        particlesIon, particlesElectron, dt/2.0f
    );
    boundary.periodicBoundaryParticleX(
        particlesIon, particlesElectron
    );
    boundary.periodicBoundaryParticleY(
        particlesIon, particlesElectron
    );
}


void PIC2D::oneStepSymmetricXWallY()
{
    fieldSolver.timeEvolutionB(B, E, dt/2.0f);
    boundary.symmetricBoundaryBX(B);
    boundary.conductingWallBoundaryBY(B);
    
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);
    getCenterBE_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpB.data()), 
        thrust::raw_pointer_cast(tmpE.data()), 
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(E.data())
    );
    cudaDeviceSynchronize();
    boundary.symmetricBoundaryBX(tmpB);
    boundary.conductingWallBoundaryBY(tmpB);
    boundary.symmetricBoundaryEX(tmpE);
    boundary.conductingWallBoundaryEY(tmpE);


    particlePush.pushVelocity(
        particlesIon, particlesElectron, tmpB, tmpE, dt
    );


    particlePush.pushPosition(
        particlesIon, particlesElectron, dt/2.0f
    );
    boundary.conductingWallBoundaryParticleX(
        particlesIon, particlesElectron
    );
    boundary.conductingWallBoundaryParticleY(
        particlesIon, particlesElectron
    );


    currentCalculator.resetCurrent(tmpCurrent);
    currentCalculator.calculateCurrent(
        tmpCurrent, particlesIon, particlesElectron
    );
    boundary.symmetricBoundaryCurrentX(tmpCurrent);
    boundary.conductingWallBoundaryCurrentY(tmpCurrent);
    getHalfCurrent_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(current.data()), 
        thrust::raw_pointer_cast(tmpCurrent.data())
    );
    boundary.symmetricBoundaryCurrentX(current);
    boundary.conductingWallBoundaryCurrentY(current);

    fieldSolver.timeEvolutionB(B, E, dt/2.0f);
    boundary.symmetricBoundaryBX(B);
    boundary.conductingWallBoundaryBY(B);

    fieldSolver.timeEvolutionE(E, B, current, dt);
    boundary.symmetricBoundaryEX(E);
    boundary.conductingWallBoundaryEY(E);
    filter.langdonMarderTypeCorrection(E, particlesIon, particlesElectron, dt);
    boundary.symmetricBoundaryEX(E);
    boundary.conductingWallBoundaryEY(E);

    particlePush.pushPosition(
        particlesIon, particlesElectron, dt/2.0f
    );
    boundary.conductingWallBoundaryParticleX(
        particlesIon, particlesElectron
    );
    boundary.conductingWallBoundaryParticleY(
        particlesIon, particlesElectron
    );
}


void PIC2D::oneStepPeriodicXWallY()
{
    fieldSolver.timeEvolutionB(B, E, dt/2.0f);
    boundary.periodicBoundaryBX(B);
    boundary.conductingWallBoundaryBY(B);
    
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);
    getCenterBE_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpB.data()), 
        thrust::raw_pointer_cast(tmpE.data()), 
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(E.data())
    );
    cudaDeviceSynchronize();
    boundary.periodicBoundaryBX(tmpB);
    boundary.conductingWallBoundaryBY(tmpB);
    boundary.periodicBoundaryEX(tmpE);
    boundary.conductingWallBoundaryEY(tmpE);

    particlePush.pushVelocity(
        particlesIon, particlesElectron, tmpB, tmpE, dt
    );

    particlePush.pushPosition(
        particlesIon, particlesElectron, dt/2.0f
    );
    boundary.periodicBoundaryParticleX(
        particlesIon, particlesElectron
    );
    boundary.conductingWallBoundaryParticleY(
        particlesIon, particlesElectron
    );


    currentCalculator.resetCurrent(tmpCurrent);
    currentCalculator.calculateCurrent(
        tmpCurrent, particlesIon, particlesElectron
    );
    boundary.periodicBoundaryCurrentX(tmpCurrent);
    boundary.conductingWallBoundaryCurrentY(tmpCurrent);
    getHalfCurrent_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(current.data()), 
        thrust::raw_pointer_cast(tmpCurrent.data())
    );
    boundary.periodicBoundaryCurrentX(current);
    boundary.conductingWallBoundaryCurrentY(current);

    fieldSolver.timeEvolutionB(B, E, dt/2.0f);
    boundary.periodicBoundaryBX(B);
    boundary.conductingWallBoundaryBY(B);

    fieldSolver.timeEvolutionE(E, B, current, dt);
    boundary.periodicBoundaryEX(E);
    boundary.conductingWallBoundaryEY(E);
    filter.langdonMarderTypeCorrection(E, particlesIon, particlesElectron, dt);
    boundary.periodicBoundaryEX(E);
    boundary.conductingWallBoundaryEY(E);

    particlePush.pushPosition(
        particlesIon, particlesElectron, dt/2.0f
    );
    boundary.periodicBoundaryParticleX(
        particlesIon, particlesElectron
    );
    boundary.conductingWallBoundaryParticleY(
        particlesIon, particlesElectron
    );
}


void PIC2D::oneStepPeriodicXFreeWallY()
{
    fieldSolver.timeEvolutionB(B, E, dt/2.0f);
    boundary.periodicBoundaryBX(B);
    boundary.freeBoundaryBY(B);
    
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);
    getCenterBE_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpB.data()), 
        thrust::raw_pointer_cast(tmpE.data()), 
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(E.data())
    );
    cudaDeviceSynchronize();
    boundary.periodicBoundaryBX(tmpB);
    boundary.freeBoundaryBY(tmpB);
    boundary.periodicBoundaryEX(tmpE);
    boundary.freeBoundaryEY(tmpE);

    particlePush.pushVelocity(
        particlesIon, particlesElectron, tmpB, tmpE, dt
    );

    particlePush.pushPosition(
        particlesIon, particlesElectron, dt/2.0f
    );
    boundary.periodicBoundaryParticleX(
        particlesIon, particlesElectron
    );
    boundary.conductingWallBoundaryParticleY(
        particlesIon, particlesElectron
    );


    currentCalculator.resetCurrent(tmpCurrent);
    currentCalculator.calculateCurrent(
        tmpCurrent, particlesIon, particlesElectron
    );
    boundary.periodicBoundaryCurrentX(tmpCurrent);
    boundary.freeBoundaryCurrentY(tmpCurrent);
    getHalfCurrent_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(current.data()), 
        thrust::raw_pointer_cast(tmpCurrent.data())
    );
    boundary.periodicBoundaryCurrentX(current);
    boundary.freeBoundaryCurrentY(current);

    fieldSolver.timeEvolutionB(B, E, dt/2.0f);
    boundary.periodicBoundaryBX(B);
    boundary.freeBoundaryBY(B);

    fieldSolver.timeEvolutionE(E, B, current, dt);
    boundary.periodicBoundaryEX(E);
    boundary.freeBoundaryEY(E);
    filter.langdonMarderTypeCorrection(E, particlesIon, particlesElectron, dt);
    boundary.periodicBoundaryEX(E);
    boundary.freeBoundaryEY(E);

    particlePush.pushPosition(
        particlesIon, particlesElectron, dt/2.0f
    );
    boundary.periodicBoundaryParticleX(
        particlesIon, particlesElectron
    );
    boundary.conductingWallBoundaryParticleY(
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
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            ofsB.write(reinterpret_cast<const char*>(&host_B[j + ny * i].bX), sizeof(float));
            ofsB.write(reinterpret_cast<const char*>(&host_B[j + ny * i].bY), sizeof(float));
            ofsB.write(reinterpret_cast<const char*>(&host_B[j + ny * i].bZ), sizeof(float));
            BEnergy += host_B[j + ny * i].bX * host_B[j + ny * i].bX 
                     + host_B[j + ny * i].bY * host_B[j + ny * i].bY
                     + host_B[j + ny * i].bZ * host_B[j + ny * i].bZ;
        }
    }
    BEnergy *= 0.5f / mu0;

    std::ofstream ofsE(filenameE, std::ios::binary);
    ofsE << std::fixed << std::setprecision(6);
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            ofsE.write(reinterpret_cast<const char*>(&host_E[j + ny * i].eX), sizeof(float));
            ofsE.write(reinterpret_cast<const char*>(&host_E[j + ny * i].eY), sizeof(float));
            ofsE.write(reinterpret_cast<const char*>(&host_E[j + ny * i].eZ), sizeof(float));
            EEnergy += host_E[j + ny * i].eX * host_E[j + ny * i].eX
                     + host_E[j + ny * i].eY * host_E[j + ny * i].eY
                     + host_E[j + ny * i].eZ * host_E[j + ny * i].eZ;
        }
    }
    EEnergy *= 0.5f * epsilon0;

    std::ofstream ofsCurrent(filenameCurrent, std::ios::binary);
    ofsCurrent << std::fixed << std::setprecision(6);
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            ofsCurrent.write(reinterpret_cast<const char*>(&host_current[j + ny * i].jX), sizeof(float));
            ofsCurrent.write(reinterpret_cast<const char*>(&host_current[j + ny * i].jY), sizeof(float));
            ofsCurrent.write(reinterpret_cast<const char*>(&host_current[j + ny * i].jZ), sizeof(float));
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
        zerothMomentIon, particlesIon, existNumIon
    );
    momentCalculater.calculateZerothMomentOfOneSpecies(
        zerothMomentElectron, particlesElectron, existNumElectron
    );
}


void PIC2D::calculateFirstMoments()
{
    momentCalculater.calculateFirstMomentOfOneSpecies(
        firstMomentIon, particlesIon, existNumIon
    );
    momentCalculater.calculateFirstMomentOfOneSpecies(
        firstMomentElectron, particlesElectron, existNumElectron
    );
}


void PIC2D::calculateSecondMoments()
{
    momentCalculater.calculateSecondMomentOfOneSpecies(
        secondMomentIon, particlesIon, existNumIon
    );
    momentCalculater.calculateSecondMomentOfOneSpecies(
        secondMomentElectron, particlesElectron, existNumElectron
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
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            ofsZerothMomentIon.write(reinterpret_cast<const char*>(
                &host_zerothMomentIon[j + ny * i].n), sizeof(float)
            );
        }
    }

    std::ofstream ofsZerothMomentElectron(filenameZerothMomentElectron, std::ios::binary);
    ofsZerothMomentElectron << std::fixed << std::setprecision(6);
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            ofsZerothMomentElectron.write(reinterpret_cast<const char*>(
                &host_zerothMomentElectron[j + ny * i].n), sizeof(float)
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
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            ofsFirstMomentIon.write(reinterpret_cast<const char*>(
                &host_firstMomentIon[j + ny * i].x), sizeof(float)
            );
            ofsFirstMomentIon.write(reinterpret_cast<const char*>(
                &host_firstMomentIon[j + ny * i].y), sizeof(float)
            );
            ofsFirstMomentIon.write(reinterpret_cast<const char*>(
                &host_firstMomentIon[j + ny * i].z), sizeof(float)
            );
        }
    }

    std::ofstream ofsFirstMomentElectron(filenameFirstMomentElectron, std::ios::binary);
    ofsFirstMomentElectron << std::fixed << std::setprecision(6);
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            ofsFirstMomentElectron.write(reinterpret_cast<const char*>(
                &host_firstMomentElectron[j + ny * i].x), sizeof(float)
            );
            ofsFirstMomentElectron.write(reinterpret_cast<const char*>(
                &host_firstMomentElectron[j + ny * i].y), sizeof(float)
            );
            ofsFirstMomentElectron.write(reinterpret_cast<const char*>(
                &host_firstMomentElectron[j + ny * i].z), sizeof(float)
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
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            ofsSecondMomentIon.write(reinterpret_cast<const char*>(
                &host_secondMomentIon[j + ny * i].xx), sizeof(float)
            );
            ofsSecondMomentIon.write(reinterpret_cast<const char*>(
                &host_secondMomentIon[j + ny * i].yy), sizeof(float)
            );
            ofsSecondMomentIon.write(reinterpret_cast<const char*>(
                &host_secondMomentIon[j + ny * i].zz), sizeof(float)
            );
            ofsSecondMomentIon.write(reinterpret_cast<const char*>(
                &host_secondMomentIon[j + ny * i].xy), sizeof(float)
            );
            ofsSecondMomentIon.write(reinterpret_cast<const char*>(
                &host_secondMomentIon[j + ny * i].xz), sizeof(float)
            );
            ofsSecondMomentIon.write(reinterpret_cast<const char*>(
                &host_secondMomentIon[j + ny * i].yz), sizeof(float)
            );
        }
    }

    std::ofstream ofsSecondMomentElectron(filenameSecondMomentElectron, std::ios::binary);
    ofsSecondMomentElectron << std::fixed << std::setprecision(6);
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            ofsSecondMomentElectron.write(reinterpret_cast<const char*>(
                &host_secondMomentElectron[j + ny * i].xx), sizeof(float)
            );
            ofsSecondMomentElectron.write(reinterpret_cast<const char*>(
                &host_secondMomentElectron[j + ny * i].yy), sizeof(float)
            );
            ofsSecondMomentElectron.write(reinterpret_cast<const char*>(
                &host_secondMomentElectron[j + ny * i].zz), sizeof(float)
            );
            ofsSecondMomentElectron.write(reinterpret_cast<const char*>(
                &host_secondMomentElectron[j + ny * i].xy), sizeof(float)
            );
            ofsSecondMomentElectron.write(reinterpret_cast<const char*>(
                &host_secondMomentElectron[j + ny * i].xz), sizeof(float)
            );
            ofsSecondMomentElectron.write(reinterpret_cast<const char*>(
                &host_secondMomentElectron[j + ny * i].yz), sizeof(float)
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
    for (unsigned long long i = 0; i < existNumIon; i++) {
        ofsXIon.write(reinterpret_cast<const char*>(&host_particlesIon[i].x), sizeof(float));
        ofsXIon.write(reinterpret_cast<const char*>(&host_particlesIon[i].y), sizeof(float));
        ofsXIon.write(reinterpret_cast<const char*>(&host_particlesIon[i].z), sizeof(float));
    }

    std::ofstream ofsXElectron(filenameXElectron, std::ios::binary);
    ofsXElectron << std::fixed << std::setprecision(6);
    for (unsigned long long i = 0; i < existNumElectron; i++) {
        ofsXElectron.write(reinterpret_cast<const char*>(&host_particlesElectron[i].x), sizeof(float));
        ofsXElectron.write(reinterpret_cast<const char*>(&host_particlesElectron[i].y), sizeof(float));
        ofsXElectron.write(reinterpret_cast<const char*>(&host_particlesElectron[i].z), sizeof(float));
    }


    float vx, vy, vz, KineticEnergy = 0.0f;

    std::ofstream ofsVIon(filenameVIon, std::ios::binary);
    ofsVIon << std::fixed << std::setprecision(6);
    for (unsigned long long i = 0; i < existNumIon; i++) {
        vx = host_particlesIon[i].vx;
        vy = host_particlesIon[i].vy;
        vz = host_particlesIon[i].vz;

        ofsVIon.write(reinterpret_cast<const char*>(&vx), sizeof(float));
        ofsVIon.write(reinterpret_cast<const char*>(&vy), sizeof(float));
        ofsVIon.write(reinterpret_cast<const char*>(&vz), sizeof(float));

        KineticEnergy += 0.5f * mIon * (vx * vx + vy * vy + vz * vz);
    }

    std::ofstream ofsVElectron(filenameVElectron, std::ios::binary);
    ofsVElectron << std::fixed << std::setprecision(6);
    for (unsigned long long i = 0; i < existNumElectron; i++) {
        vx = host_particlesElectron[i].vx;
        vy = host_particlesElectron[i].vy;
        vz = host_particlesElectron[i].vz;

        ofsVElectron.write(reinterpret_cast<const char*>(&vx), sizeof(float));
        ofsVElectron.write(reinterpret_cast<const char*>(&vy), sizeof(float));
        ofsVElectron.write(reinterpret_cast<const char*>(&vz), sizeof(float));
        
        KineticEnergy += 0.5f * mElectron * (vx * vx + vy * vy + vz * vz);
    }

    std::ofstream ofsKineticEnergy(filenameKineticEnergy, std::ios::binary);
    ofsKineticEnergy << std::fixed << std::setprecision(6);
    ofsKineticEnergy.write(reinterpret_cast<const char*>(&KineticEnergy), sizeof(float));
}

