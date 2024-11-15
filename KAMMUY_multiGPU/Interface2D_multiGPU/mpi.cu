#include "mpi.hpp"


int Interface2DMPI::MPIInfo::getRank(int dx, int dy)
{
    int rankX = (localGridX + dx + gridX) % gridX;
    int rankY = (localGridY + dy + gridY) % gridY;
    return rankY + rankX * gridY;
}


bool Interface2DMPI::MPIInfo::isInside(int globalX, int globalY)
{
    int startX = localNx * localGridX;
    int endX = startX + localNx;
    int startY = localNy * localGridY;
    int endY = startY + localNy;

    if (globalX < startX) return false;
    if (globalX >= endX) return false;
    if (globalY < startY) return false;
    if (globalY >= endY) return false;

    return true;
}


int Interface2DMPI::MPIInfo::globalToLocal(int globalX, int globalY)
{
    int startX = localNx * localGridX;
    int x = globalX - startX;

    int startY = localNy * localGridY;
    int y = globalY - startY;

    return y + buffer + (x + buffer) * localSizeY;
}


void Interface2DMPI::setupInfo(MPIInfo& mPIInfo, int buffer, int gridX, int gridY)
{
    int rank = 0, procs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    mPIInfo.rank = rank;
    mPIInfo.procs = procs;
    mPIInfo.gridX = gridX;
    mPIInfo.gridY = gridY;
    mPIInfo.localGridX = rank / mPIInfo.gridY;
    mPIInfo.localGridY = rank % mPIInfo.gridY;
    mPIInfo.localNx = Interface2DConst::nx / mPIInfo.gridX;
    mPIInfo.localNy = Interface2DConst::ny / mPIInfo.gridY;
    mPIInfo.buffer = buffer;
    mPIInfo.localSizeX = mPIInfo.localNx + 2 * mPIInfo.buffer;
    mPIInfo.localSizeY = mPIInfo.localNy; //not need buffer!


    int block_lengths_reloadParticlesData[5] = {1, 1, 1, 1, 1};
    MPI_Aint offsets_reloadParticlesData[5];
    offsets_reloadParticlesData[0] = offsetof(ReloadParticlesData, numAndIndex);
    offsets_reloadParticlesData[1] = offsetof(ReloadParticlesData, u);
    offsets_reloadParticlesData[2] = offsetof(ReloadParticlesData, v);
    offsets_reloadParticlesData[3] = offsetof(ReloadParticlesData, w);
    offsets_reloadParticlesData[4] = offsetof(ReloadParticlesData, vth);
    MPI_Datatype types_reloadParticlesData[5] = {MPI_UNSIGNED_LONG, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
    MPI_Type_create_struct(3, block_lengths_reloadParticlesData, offsets_reloadParticlesData, types_reloadParticlesData, &mPIInfo.mpi_reloadParticlesDataType);
    MPI_Type_commit(&mPIInfo.mpi_reloadParticlesDataType);
}


void Interface2DMPI::sendrecv_reloadParticlesData_x(
    thrust::device_vector<ReloadParticlesData> reloadParticlesDataSpecies, 
    MPIInfo& mPIInfo
)
{
    int localNx = mPIInfo.localNx;
    int localNy = mPIInfo.localNy;
    //int localSizeX = mPIInfo.localSizeX;
    int localSizeY = mPIInfo.localSizeY;

    int left  = mPIInfo.getRank(-1, 0);
    int right = mPIInfo.getRank(1, 0);
    MPI_Status st;

    thrust::host_vector<ReloadParticlesData> sendDataLeft(mPIInfo.buffer * localNy), sendDataRight(mPIInfo.buffer * localNy);
    thrust::host_vector<ReloadParticlesData> recvDataLeft(mPIInfo.buffer * localNy), recvDataRight(mPIInfo.buffer * localNy);

    for (int i = 0; i < mPIInfo.buffer; i++) {
        for (int j = 0; j < localNy; j++) {
            sendDataLeft[ j + i * localNy] = reloadParticlesDataSpecies[j + mPIInfo.buffer + (mPIInfo.buffer + i) * localSizeY];
            sendDataRight[j + i * localNy] = reloadParticlesDataSpecies[j + mPIInfo.buffer + (localNx + i)        * localSizeY];
        }
    }

    MPI_Sendrecv(sendDataLeft.data(),  sendDataLeft.size(),  mPIInfo.mpi_reloadParticlesDataType, left,  0, 
                 recvDataRight.data(), recvDataRight.size(), mPIInfo.mpi_reloadParticlesDataType, right, 0, 
                 MPI_COMM_WORLD, &st);
    MPI_Sendrecv(sendDataRight.data(), sendDataRight.size(), mPIInfo.mpi_reloadParticlesDataType, right, 0, 
                 recvDataLeft.data(),  recvDataLeft.size(),  mPIInfo.mpi_reloadParticlesDataType, left,  0, 
                 MPI_COMM_WORLD, &st);

    for (int i = 0; i < mPIInfo.buffer; i++) {
        for (int j = 0; j < localNy; j++) {
            reloadParticlesDataSpecies[j + mPIInfo.buffer + i                              * localSizeY] = recvDataLeft[ j + i * localNy];
            reloadParticlesDataSpecies[j + mPIInfo.buffer + (localNx + mPIInfo.buffer + i) * localSizeY] = recvDataRight[j + i * localNy];
        }
    }
}


