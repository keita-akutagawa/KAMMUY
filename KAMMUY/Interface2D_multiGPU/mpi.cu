#include "mpi.hpp"


int Interface2DMPI::MPIInfo::getRank(int dx)
{
    int rankX = (localGridX + dx + gridX) % gridX;
    return rankX;
}


void Interface2DMPI::setupInfo(MPIInfo& mPIInfo, int buffer)
{
    int rank = 0, procs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    mPIInfo.rank = rank;
    mPIInfo.procs = procs;
    mPIInfo.gridX = procs;
    mPIInfo.localGridX = rank;
    mPIInfo.localNx = Interface2DConst::nx / mPIInfo.gridX;
    mPIInfo.buffer = buffer;
    mPIInfo.localSizeX = mPIInfo.localNx + 2 * mPIInfo.buffer;
}


