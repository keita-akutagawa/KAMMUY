#include "restart.hpp"


RestartMHD::RestartMHD(IdealMHD2DMPI::MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo)
{
}


void RestartMHD::loadU(
    thrust::host_vector<ConservationParameter>& hU, 
    thrust::device_vector<ConservationParameter>& U, 
    std::string directoryname, 
    std::string filenameWithoutStep, 
    int step
)
{
    std::string filename;
    filename = directoryname + "/"
             + filenameWithoutStep + "_" + std::to_string(step)
             + "_" + std::to_string(mPIInfo.rank)
             + ".bin";

    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    for (int i = mPIInfo.buffer; i < mPIInfo.localNx + mPIInfo.buffer; i++) {
        for (int j = mPIInfo.buffer; j < mPIInfo.localNy + mPIInfo.buffer; j++) {
            ifs.read(reinterpret_cast<char*>(&hU[j + i * mPIInfo.localSizeY].rho),  sizeof(double));
            ifs.read(reinterpret_cast<char*>(&hU[j + i * mPIInfo.localSizeY].rhoU), sizeof(double));
            ifs.read(reinterpret_cast<char*>(&hU[j + i * mPIInfo.localSizeY].rhoV), sizeof(double));
            ifs.read(reinterpret_cast<char*>(&hU[j + i * mPIInfo.localSizeY].rhoW), sizeof(double));
            ifs.read(reinterpret_cast<char*>(&hU[j + i * mPIInfo.localSizeY].bX),   sizeof(double));
            ifs.read(reinterpret_cast<char*>(&hU[j + i * mPIInfo.localSizeY].bY),   sizeof(double));
            ifs.read(reinterpret_cast<char*>(&hU[j + i * mPIInfo.localSizeY].bZ),   sizeof(double));
            ifs.read(reinterpret_cast<char*>(&hU[j + i * mPIInfo.localSizeY].e),    sizeof(double));
        }
    }

    ifs.close();

    U = hU; 

}

