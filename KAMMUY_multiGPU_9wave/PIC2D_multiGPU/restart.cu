#include "restart.hpp"


RestartPIC::RestartPIC(PIC2DMPI::MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo)
{
}


void RestartPIC::loadFields(
    thrust::host_vector<MagneticField>& host_B, 
    thrust::host_vector<ElectricField>& host_E, 
    thrust::device_vector<MagneticField>& B, 
    thrust::device_vector<ElectricField>& E, 
    std::string directoryname, 
    std::string filenameWithoutStep, 
    int step
)
{
    std::string filenameB, filenameE; 
    filenameB = directoryname + "/"
             + filenameWithoutStep + "_B_" + std::to_string(step)
             + "_" + std::to_string(mPIInfo.rank)
             + ".bin";
    filenameE = directoryname + "/"
             + filenameWithoutStep + "_E_" + std::to_string(step)
             + "_" + std::to_string(mPIInfo.rank)
             + ".bin";

    std::ifstream ifsB(filenameB, std::ios::binary);
    if (!ifsB) {
        throw std::runtime_error("Failed to open file: " + filenameB);
    }
    std::ifstream ifsE(filenameE, std::ios::binary);
    if (!ifsE) {
        throw std::runtime_error("Failed to open file: " + filenameE);
    }

    for (int i = mPIInfo.buffer; i < mPIInfo.localNx + mPIInfo.buffer; i++) {
        for (int j = mPIInfo.buffer; j < mPIInfo.localNy + mPIInfo.buffer; j++) {
            int index = j + mPIInfo.localSizeY * i;
            ifsB.read(reinterpret_cast<char*>(&host_B[index].bX), sizeof(float));
            ifsB.read(reinterpret_cast<char*>(&host_B[index].bY), sizeof(float));
            ifsB.read(reinterpret_cast<char*>(&host_B[index].bZ), sizeof(float));
        }
    }
    for (int i = mPIInfo.buffer; i < mPIInfo.localNx + mPIInfo.buffer; i++) {
        for (int j = mPIInfo.buffer; j < mPIInfo.localNy + mPIInfo.buffer; j++) {
            int index = j + mPIInfo.localSizeY * i;
            ifsE.read(reinterpret_cast<char*>(&host_E[index].eX), sizeof(float));
            ifsE.read(reinterpret_cast<char*>(&host_E[index].eY), sizeof(float));
            ifsE.read(reinterpret_cast<char*>(&host_E[index].eZ), sizeof(float));
        }
    }

    B = host_B; 
    E = host_E; 

    MPI_Barrier(MPI_COMM_WORLD);
    PIC2DMPI::sendrecv_field(B, mPIInfo, mPIInfo.mpi_fieldType);
    PIC2DMPI::sendrecv_field(E, mPIInfo, mPIInfo.mpi_fieldType);
    MPI_Barrier(MPI_COMM_WORLD);
    
}


void RestartPIC::loadParticles(
    thrust::host_vector<Particle>& host_particlesIon, 
    thrust::host_vector<Particle>& host_particlesElectron, 
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron, 
    std::string directoryname, 
    std::string filenameWithoutStep, 
    int step
)
{
    std::string filenameXIon, filenameXElectron;
    std::string filenameVIon, filenameVElectron;
    std::string filenameNumIon, filenameNumElectron;

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
    
    std::ifstream ifsXIon(filenameXIon, std::ios::binary);
    if (!ifsXIon) {
        throw std::runtime_error("Failed to open file: " + filenameXIon);
    }
    std::ifstream ifsXElectron(filenameXElectron, std::ios::binary);
    if (!ifsXElectron) {
        throw std::runtime_error("Failed to open file: " + filenameXElectron);
    }
    std::ifstream ifsVIon(filenameVIon, std::ios::binary);
    if (!ifsVIon) {
        throw std::runtime_error("Failed to open file: " + filenameVIon);
    }
    std::ifstream ifsVElectron(filenameVElectron, std::ios::binary);
    if (!ifsVElectron) {
        throw std::runtime_error("Failed to open file: " + filenameVElectron);
    }
    std::ifstream ifsNumIon(filenameNumIon, std::ios::binary);
    if (!ifsNumIon) {
        throw std::runtime_error("Failed to open file: " + filenameNumIon);
    }
    std::ifstream ifsNumElectron(filenameNumElectron, std::ios::binary);
    if (!ifsNumElectron) {
        throw std::runtime_error("Failed to open file: " + filenameNumElectron);
    }

    unsigned long long existNumIonPerProcs, existNumElectronPerProcs;
    ifsNumIon.read(reinterpret_cast<char*>(&existNumIonPerProcs), sizeof(unsigned long long));
    ifsNumElectron.read(reinterpret_cast<char*>(&existNumElectronPerProcs), sizeof(unsigned long long));
    mPIInfo.existNumIonPerProcs = existNumIonPerProcs;
    mPIInfo.existNumElectronPerProcs = existNumElectronPerProcs; 

    for (unsigned long long i = 0; i < existNumIonPerProcs; i++) {
        float x, y, z, vx, vy, vz, gamma; 

        ifsXIon.read(reinterpret_cast<char*>(&x),  sizeof(float));
        ifsXIon.read(reinterpret_cast<char*>(&y),  sizeof(float));
        ifsXIon.read(reinterpret_cast<char*>(&z),  sizeof(float));
        ifsVIon.read(reinterpret_cast<char*>(&vx), sizeof(float));
        ifsVIon.read(reinterpret_cast<char*>(&vy), sizeof(float));
        ifsVIon.read(reinterpret_cast<char*>(&vz), sizeof(float));

        gamma = 1.0f / (sqrt(1.0 - (vx * vx + vy * vy + vz * vz) / pow(PIC2DConst::c, 2)));

        host_particlesIon[i].x = x;
        host_particlesIon[i].y = y;
        host_particlesIon[i].z = z; 
        host_particlesIon[i].vx = vx * gamma;
        host_particlesIon[i].vy = vy * gamma;
        host_particlesIon[i].vz = vz * gamma;
        host_particlesIon[i].gamma = gamma;
        host_particlesIon[i].isExist = true;
    }

    for (unsigned long long i = 0; i < existNumElectronPerProcs; i++) {
        float x, y, z, vx, vy, vz, gamma; 

        ifsXElectron.read(reinterpret_cast<char*>(&x),  sizeof(float));
        ifsXElectron.read(reinterpret_cast<char*>(&y),  sizeof(float));
        ifsXElectron.read(reinterpret_cast<char*>(&z),  sizeof(float));
        ifsVElectron.read(reinterpret_cast<char*>(&vx), sizeof(float));
        ifsVElectron.read(reinterpret_cast<char*>(&vy), sizeof(float));
        ifsVElectron.read(reinterpret_cast<char*>(&vz), sizeof(float));

        gamma = 1.0f / (sqrt(1.0 - (vx * vx + vy * vy + vz * vz) / pow(PIC2DConst::c, 2)));

        host_particlesElectron[i].x = x;
        host_particlesElectron[i].y = y;
        host_particlesElectron[i].z = z; 
        host_particlesElectron[i].vx = vx * gamma;
        host_particlesElectron[i].vy = vy * gamma;
        host_particlesElectron[i].vz = vz * gamma;
        host_particlesElectron[i].gamma = gamma;
        host_particlesElectron[i].isExist = true;
    }

    particlesIon = host_particlesIon; 
    particlesElectron = host_particlesElectron; 
}


