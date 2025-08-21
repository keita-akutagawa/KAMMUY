#include "reloader.hpp"


Reloader::Reloader(
    PIC2DMPI::MPIInfo& mPIInfoPIC, 
    IdealMHD2DMPI::MPIInfo& mPIInfoMHD
)   : mPIInfoPIC(mPIInfoPIC), 
      mPIInfoMHD(mPIInfoMHD)
{
}


void Reloader::reloadPICData(
    thrust::host_vector<Particle>& host_particlesIon, 
    thrust::host_vector<Particle>& host_particlesElectron, 
    thrust::host_vector<MagneticField>& host_B, 
    thrust::host_vector<ElectricField>& host_E, 
    std::string savedDirectoryName, 
    std::string filenameWithoutStep, 
    int step
)
{
    std::string filenameB, filenameE;
    filenameB = savedDirectoryName + "/"
             + filenameWithoutStep + "_B_" + std::to_string(step)
             + "_" + std::to_string(mPIInfoPIC.rank)
             + ".bin";
    filenameE = savedDirectoryName + "/"
             + filenameWithoutStep + "_E_" + std::to_string(step)
             + "_" + std::to_string(mPIInfoPIC.rank)
             + ".bin";
    

    std::ifstream ifsB(filenameB, std::ios::binary);
    for (int i = 0; i < mPIInfoPIC.localSizeX; i++) {
        for (int j = 0; j < PIC2DConst::ny; j++) {
            ifsB.read(reinterpret_cast<char*>(&host_B[j + i * PIC2DConst::ny].bX), sizeof(double));
            ifsB.read(reinterpret_cast<char*>(&host_B[j + i * PIC2DConst::ny].bY), sizeof(double));
            ifsB.read(reinterpret_cast<char*>(&host_B[j + i * PIC2DConst::ny].bZ), sizeof(double));
        }
    }

    std::ifstream ifsE(filenameE, std::ios::binary);
    for (int i = 0; i < mPIInfoPIC.localSizeX; i++) {
        for (int j = 0; j < PIC2DConst::ny; j++) {
            ifsE.read(reinterpret_cast<char*>(&host_E[j + i * PIC2DConst::ny].eX), sizeof(double));
            ifsE.read(reinterpret_cast<char*>(&host_E[j + i * PIC2DConst::ny].eY), sizeof(double));
            ifsE.read(reinterpret_cast<char*>(&host_E[j + i * PIC2DConst::ny].eZ), sizeof(double));
        }
    }


    std::string filenameParticleNumIon, filenameParticleNumElectron;  
    filenameParticleNumIon = savedDirectoryName + "/"
                           + filenameWithoutStep + "_num_ion_" + std::to_string(step)
                           + "_" + std::to_string(mPIInfoPIC.rank)
                           + ".bin";
    filenameParticleNumElectron = savedDirectoryName + "/"
                                + filenameWithoutStep + "_num_electron_" + std::to_string(step)
                                + "_" + std::to_string(mPIInfoPIC.rank)
                                + ".bin";
                
    std::ifstream ifsParticleNumIon(filenameParticleNumIon, std::ios::binary);
    std::ifstream ifsParticleNumElectron(filenameParticleNumElectron, std::ios::binary);

    ifsParticleNumIon.read(reinterpret_cast<char*>(&mPIInfoPIC.existNumIonPerProcs), sizeof(unsigned long long));
    ifsParticleNumElectron.read(reinterpret_cast<char*>(&mPIInfoPIC.existNumElectronPerProcs), sizeof(unsigned long long));


    std::string filenameParticleIonX, filenameParticleIonV;  
    std::string filenameParticleElectronX, filenameParticleElectronV;  
    filenameParticleIonX = savedDirectoryName + "/"
                         + filenameWithoutStep + "_x_ion_" + std::to_string(step)
                         + "_" + std::to_string(mPIInfoPIC.rank)
                         + ".bin";
    filenameParticleIonV = savedDirectoryName + "/"
                         + filenameWithoutStep + "_v_ion_" + std::to_string(step)
                         + "_" + std::to_string(mPIInfoPIC.rank)
                         + ".bin";
    filenameParticleElectronX = savedDirectoryName + "/"
                              + filenameWithoutStep + "_x_electron_" + std::to_string(step)
                              + "_" + std::to_string(mPIInfoPIC.rank)
                              + ".bin";
    filenameParticleElectronV = savedDirectoryName + "/"
                              + filenameWithoutStep + "_v_electron_" + std::to_string(step)
                              + "_" + std::to_string(mPIInfoPIC.rank)
                              + ".bin";
    
    std::ifstream ifsParticleIonX(filenameParticleIonX, std::ios::binary);
    std::ifstream ifsParticleIonV(filenameParticleIonV, std::ios::binary);
    for (int i = 0; i < mPIInfoPIC.existNumIonPerProcs; i++) {
        double x, y, z, vx, vy, vz, gamma; 
        ifsParticleIonX.read(reinterpret_cast<char*>(&x), sizeof(double));
        ifsParticleIonX.read(reinterpret_cast<char*>(&y), sizeof(double));
        ifsParticleIonX.read(reinterpret_cast<char*>(&z), sizeof(double));
        ifsParticleIonV.read(reinterpret_cast<char*>(&vx), sizeof(double));
        ifsParticleIonV.read(reinterpret_cast<char*>(&vy), sizeof(double));
        ifsParticleIonV.read(reinterpret_cast<char*>(&vz), sizeof(double));
        gamma = 1.0 / sqrt(1.0 - (vx * vx + vy * vy + vz * vz) / pow(PIC2DConst::c, 2)); 

        host_particlesIon[i].x = x; 
        host_particlesIon[i].y = y; 
        host_particlesIon[i].z = z; 
        host_particlesIon[i].vx = vx; 
        host_particlesIon[i].vy = vy; 
        host_particlesIon[i].vz = vz; 
        host_particlesIon[i].gamma = gamma; 
        host_particlesIon[i].isExist = true;
    }

    std::ifstream ifsParticleElectronX(filenameParticleElectronX, std::ios::binary);
    std::ifstream ifsParticleElectronV(filenameParticleElectronV, std::ios::binary);
    for (int i = 0; i < mPIInfoPIC.existNumElectronPerProcs; i++) {
        double x, y, z, vx, vy, vz, gamma; 
        ifsParticleElectronX.read(reinterpret_cast<char*>(&x), sizeof(double));
        ifsParticleElectronX.read(reinterpret_cast<char*>(&y), sizeof(double));
        ifsParticleElectronX.read(reinterpret_cast<char*>(&z), sizeof(double));
        ifsParticleElectronV.read(reinterpret_cast<char*>(&vx), sizeof(double));
        ifsParticleElectronV.read(reinterpret_cast<char*>(&vy), sizeof(double));
        ifsParticleElectronV.read(reinterpret_cast<char*>(&vz), sizeof(double));
        gamma = 1.0 / sqrt(1.0 - (vx * vx + vy * vy + vz * vz) / pow(PIC2DConst::c, 2)); 

        host_particlesElectron[i].x = x; 
        host_particlesElectron[i].y = y; 
        host_particlesElectron[i].z = z; 
        host_particlesElectron[i].vx = vx; 
        host_particlesElectron[i].vy = vy; 
        host_particlesElectron[i].vz = vz; 
        host_particlesElectron[i].gamma = gamma; 
        host_particlesElectron[i].isExist = true;
    }
}


void Reloader::reloadMHDData(
    thrust::host_vector<ConservationParameter>& host_U, 
    std::string savedDirectoryName, 
    std::string filenameWithoutStep, 
    int step
)
{
    std::string filenameU;
    filenameU = savedDirectoryName + "/"
              + filenameWithoutStep + "_U_" + std::to_string(step)
              + "_" + std::to_string(mPIInfoMHD.rank)
              + ".bin";
    
    std::ifstream ifsU(filenameU, std::ios::binary);
    for (int i = 0; i < mPIInfoMHD.localSizeX; i++) {
        for (int j = 0; j < IdealMHD2DConst::ny; j++) {
            ifsU.read(reinterpret_cast<char*>(&host_U[j + i * IdealMHD2DConst::ny].rho), sizeof(double));
            ifsU.read(reinterpret_cast<char*>(&host_U[j + i * IdealMHD2DConst::ny].rhoU), sizeof(double));
            ifsU.read(reinterpret_cast<char*>(&host_U[j + i * IdealMHD2DConst::ny].rhoV), sizeof(double));
            ifsU.read(reinterpret_cast<char*>(&host_U[j + i * IdealMHD2DConst::ny].rhoW), sizeof(double));
            ifsU.read(reinterpret_cast<char*>(&host_U[j + i * IdealMHD2DConst::ny].bX), sizeof(double));
            ifsU.read(reinterpret_cast<char*>(&host_U[j + i * IdealMHD2DConst::ny].bY), sizeof(double));
            ifsU.read(reinterpret_cast<char*>(&host_U[j + i * IdealMHD2DConst::ny].bZ), sizeof(double));
            ifsU.read(reinterpret_cast<char*>(&host_U[j + i * IdealMHD2DConst::ny].e), sizeof(double));
        }
    }
}

