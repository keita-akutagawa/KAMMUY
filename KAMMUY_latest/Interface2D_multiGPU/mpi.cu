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


    int block_lengths_reloadParticlesData[5] = {1, 1, 1, 1, 1};
    MPI_Aint offsets_reloadParticlesData[5];
    offsets_reloadParticlesData[0] = offsetof(ReloadParticlesData, numAndIndex);
    offsets_reloadParticlesData[1] = offsetof(ReloadParticlesData, u);
    offsets_reloadParticlesData[2] = offsetof(ReloadParticlesData, v);
    offsets_reloadParticlesData[3] = offsetof(ReloadParticlesData, w);
    offsets_reloadParticlesData[4] = offsetof(ReloadParticlesData, vth);
    MPI_Datatype types_reloadParticlesData[5] = {MPI_UNSIGNED_LONG, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT};
    MPI_Type_create_struct(5, block_lengths_reloadParticlesData, offsets_reloadParticlesData, types_reloadParticlesData, &mPIInfo.mpi_reloadParticlesDataType);
    MPI_Type_commit(&mPIInfo.mpi_reloadParticlesDataType);
}


void Interface2DMPI::sendrecv_reloadParticlesData_x(
    thrust::device_vector<ReloadParticlesData>& reloadParticlesDataSpecies, 
    MPIInfo& mPIInfo
)
{
    int localNx = mPIInfo.localNx;

    int left  = mPIInfo.getRank(-1);
    int right = mPIInfo.getRank(1);
    MPI_Status st;

    thrust::device_vector<ReloadParticlesData> sendDataLeft(mPIInfo.buffer * Interface2DConst::interfaceLength), sendDataRight(mPIInfo.buffer * Interface2DConst::interfaceLength);
    thrust::device_vector<ReloadParticlesData> recvDataLeft(mPIInfo.buffer * Interface2DConst::interfaceLength), recvDataRight(mPIInfo.buffer * Interface2DConst::interfaceLength);

    for (int i = 0; i < mPIInfo.buffer; i++) {
        for (int j = 0; j < Interface2DConst::interfaceLength; j++) {
            sendDataLeft[ j + i * Interface2DConst::interfaceLength] = reloadParticlesDataSpecies[j + (mPIInfo.buffer + i) * Interface2DConst::interfaceLength];
            sendDataRight[j + i * Interface2DConst::interfaceLength] = reloadParticlesDataSpecies[j + (localNx + i)        * Interface2DConst::interfaceLength];
        }
    }

    MPI_Sendrecv(thrust::raw_pointer_cast(sendDataLeft.data()),  sendDataLeft.size(),  mPIInfo.mpi_reloadParticlesDataType, left,  0, 
                 thrust::raw_pointer_cast(recvDataRight.data()), recvDataRight.size(), mPIInfo.mpi_reloadParticlesDataType, right, 0, 
                 MPI_COMM_WORLD, &st);
    MPI_Sendrecv(thrust::raw_pointer_cast(sendDataRight.data()), sendDataRight.size(), mPIInfo.mpi_reloadParticlesDataType, right, 0, 
                 thrust::raw_pointer_cast(recvDataLeft.data()),  recvDataLeft.size(),  mPIInfo.mpi_reloadParticlesDataType, left,  0, 
                 MPI_COMM_WORLD, &st);

    for (int i = 0; i < mPIInfo.buffer; i++) {
        for (int j = 0; j < Interface2DConst::interfaceLength; j++) {
            reloadParticlesDataSpecies[j + i                              * Interface2DConst::interfaceLength] = recvDataLeft[ j + i * Interface2DConst::interfaceLength];
            reloadParticlesDataSpecies[j + (localNx + mPIInfo.buffer + i) * Interface2DConst::interfaceLength] = recvDataRight[j + i * Interface2DConst::interfaceLength];
        }
    }
}


