#include "binary_output.hpp"


BinaryOutput::BinaryOutput(
    const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
    const MHDConstParameter& mHDConstParameter
)
  : NX(NX), 
    NY(NY), 
    mHDConstParameter(mHDConstParameter), 
    host_U(NX * NY)
{
}


void BinaryOutput::save(
    const thrust::device_vector<MHDValue>& U, 
    std::string addName  
) 
{
    host_U = U;

    std::string filename;
    filename = mHDConstParameter.SAVE_DIRNAME + "/"
             + mHDConstParameter.SAVE_FILENAME_WITHOUT_STEP + addName + "_" + std::to_string(mHDConstParameter.CURRENT_STEP)
             + ".bin";

    std::ofstream ofs(filename, std::ios::binary);
    ofs << std::fixed << std::setprecision(6);

    for (MHDUnsignedInt i = 0; i < NX; i++) {
        for (MHDUnsignedInt j = 0; j < NY; j++) {
            MHDUnsignedLongLong index = getIndex<MHDUnsignedLongLong>(i, j, NX, NY);

            ofs.write(reinterpret_cast<const char*>(&host_U[index].rho), sizeof(MHDFloat));
            ofs.write(reinterpret_cast<const char*>(&host_U[index].u),   sizeof(MHDFloat));
            ofs.write(reinterpret_cast<const char*>(&host_U[index].v),   sizeof(MHDFloat));
            ofs.write(reinterpret_cast<const char*>(&host_U[index].w),   sizeof(MHDFloat));
            ofs.write(reinterpret_cast<const char*>(&host_U[index].bX),  sizeof(MHDFloat));
            ofs.write(reinterpret_cast<const char*>(&host_U[index].bY),  sizeof(MHDFloat));
            ofs.write(reinterpret_cast<const char*>(&host_U[index].bZ),  sizeof(MHDFloat));
            ofs.write(reinterpret_cast<const char*>(&host_U[index].pXX), sizeof(MHDFloat));
            ofs.write(reinterpret_cast<const char*>(&host_U[index].pYY), sizeof(MHDFloat));
            ofs.write(reinterpret_cast<const char*>(&host_U[index].pZZ), sizeof(MHDFloat));
            ofs.write(reinterpret_cast<const char*>(&host_U[index].pXY), sizeof(MHDFloat));
            ofs.write(reinterpret_cast<const char*>(&host_U[index].pXZ), sizeof(MHDFloat));
            ofs.write(reinterpret_cast<const char*>(&host_U[index].pYZ), sizeof(MHDFloat));
            ofs.write(reinterpret_cast<const char*>(&host_U[index].psi), sizeof(MHDFloat));
        }
    }
}

