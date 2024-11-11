#include "mpi.hpp"


int IdealMHD2DMPI::MPIInfo::getRank(int dx, int dy)
{
    int rankX = (localGridX + dx + gridX) % gridX;
    int rankY = (localGridY + dy + gridY) % gridY;
    return rankY + rankX * gridY;
}


bool IdealMHD2DMPI::MPIInfo::isInside(int globalX, int globalY)
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


int IdealMHD2DMPI::MPIInfo::globalToLocal(int globalX, int globalY)
{
    int startX = localNx * localGridX;
    int x = globalX - startX;

    int startY = localNy * localGridY;
    int y = globalY - startY;

    return y + buffer + (x + buffer) * localSizeY;
}


void IdealMHD2DMPI::setupInfo(MPIInfo& mPIInfo, int buffer, int gridX, int gridY)
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
    mPIInfo.localNx = IdealMHD2DConst::nx / mPIInfo.gridX;
    mPIInfo.localNy = IdealMHD2DConst::ny / mPIInfo.gridY;
    mPIInfo.buffer = buffer;
    mPIInfo.localSizeX = mPIInfo.localNx + 2 * mPIInfo.buffer;
    mPIInfo.localSizeY = mPIInfo.localNy + 2 * mPIInfo.buffer;


    int block_lengths_conservation_parameter[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    MPI_Aint offsets_conservation_parameter[8];
    offsets_conservation_parameter[0] = offsetof(ConservationParameter, rho);
    offsets_conservation_parameter[1] = offsetof(ConservationParameter, rhoU);
    offsets_conservation_parameter[2] = offsetof(ConservationParameter, rhoV);
    offsets_conservation_parameter[3] = offsetof(ConservationParameter, rhoW);
    offsets_conservation_parameter[4] = offsetof(ConservationParameter, bX);
    offsets_conservation_parameter[5] = offsetof(ConservationParameter, bY);
    offsets_conservation_parameter[6] = offsetof(ConservationParameter, bZ);
    offsets_conservation_parameter[7] = offsetof(ConservationParameter, e);

    MPI_Datatype types_conservation_parameter[8] = {
        MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, 
        MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, 
        MPI_DOUBLE, MPI_DOUBLE
    };

    MPI_Type_create_struct(8, block_lengths_conservation_parameter, offsets_conservation_parameter, types_conservation_parameter, &mPIInfo.mpi_conservation_parameter_type);
    MPI_Type_commit(&mPIInfo.mpi_conservation_parameter_type);


    int block_lengths_flux[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    MPI_Aint offsets_flux[8];
    offsets_flux[0] = offsetof(Flux, f0);
    offsets_flux[1] = offsetof(Flux, f1);
    offsets_flux[2] = offsetof(Flux, f2);
    offsets_flux[3] = offsetof(Flux, f3);
    offsets_flux[4] = offsetof(Flux, f4);
    offsets_flux[5] = offsetof(Flux, f5);
    offsets_flux[6] = offsetof(Flux, f6);
    offsets_flux[7] = offsetof(Flux, f7);

    MPI_Datatype types_flux[8] = {
        MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, 
        MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, 
        MPI_DOUBLE, MPI_DOUBLE
    };

    MPI_Type_create_struct(8, block_lengths_flux, offsets_flux, types_flux, &mPIInfo.mpi_flux_type);
    MPI_Type_commit(&mPIInfo.mpi_flux_type);
}


void IdealMHD2DMPI::sendrecv_U_x(thrust::device_vector<ConservationParameter>& U, MPIInfo& mPIInfo)
{
    int localNx = mPIInfo.localNx;
    int localNy = mPIInfo.localNy;
    //int localSizeX = mPIInfo.localSizeX;
    int localSizeY = mPIInfo.localSizeY;

    int left = mPIInfo.getRank(-1, 0);
    int right = mPIInfo.getRank(1, 0);
    MPI_Status st;

    thrust::host_vector<ConservationParameter> sendULeft(mPIInfo.buffer * localNy), sendURight(mPIInfo.buffer * localNy);
    thrust::host_vector<ConservationParameter> recvULeft(mPIInfo.buffer * localNy), recvURight(mPIInfo.buffer * localNy);

    for (int i = 0; i < mPIInfo.buffer; i++) {
        for (int j = 0; j < localNy; j++) {
            sendULeft[ j + i * localNy] = U[j + mPIInfo.buffer + (mPIInfo.buffer + i) * localSizeY];
            sendURight[j + i * localNy] = U[j + mPIInfo.buffer + (localNx + i)        * localSizeY];
        }
    }

    MPI_Sendrecv(sendULeft.data(),  sendULeft.size(),  mPIInfo.mpi_conservation_parameter_type, left,  0, 
                 recvURight.data(), recvURight.size(), mPIInfo.mpi_conservation_parameter_type, right, 0, 
                 MPI_COMM_WORLD, &st);
    MPI_Sendrecv(sendURight.data(), sendURight.size(), mPIInfo.mpi_conservation_parameter_type, right, 0, 
                 recvULeft.data(),  recvULeft.size(),  mPIInfo.mpi_conservation_parameter_type, left,  0, 
                 MPI_COMM_WORLD, &st);

    for (int i = 0; i < mPIInfo.buffer; i++) {
        for (int j = 0; j < localNy; j++) {
            U[j + mPIInfo.buffer + i                              * localSizeY] = recvULeft[ j + i * localNy];
            U[j + mPIInfo.buffer + (localNx + mPIInfo.buffer + i) * localSizeY] = recvURight[j + i * localNy];
        }
    }
}


void IdealMHD2DMPI::sendrecv_U_y(thrust::device_vector<ConservationParameter>& U, MPIInfo& mPIInfo)
{
    //int localNx = mPIInfo.localNx;
    int localNy = mPIInfo.localNy;
    int localSizeX = mPIInfo.localSizeX;
    int localSizeY = mPIInfo.localSizeY;

    int down = mPIInfo.getRank(0, -1);
    int up   = mPIInfo.getRank(0, 1);
    MPI_Status st;

    thrust::host_vector<ConservationParameter> sendUDown(mPIInfo.buffer * localSizeX), sendUUp(mPIInfo.buffer * localSizeX);
    thrust::host_vector<ConservationParameter> recvUDown(mPIInfo.buffer * localSizeX), recvUUp(mPIInfo.buffer * localSizeX);

    for (int i = 0; i < localSizeX; i++) {
        for (int j = 0; j < mPIInfo.buffer; j++) {
            sendUDown[j + i * mPIInfo.buffer] = U[j + mPIInfo.buffer + i * localSizeY];
            sendUUp[  j + i * mPIInfo.buffer] = U[j + localNy        + i * localSizeY];
        }
    }

    MPI_Sendrecv(sendUDown.data(), sendUDown.size(), mPIInfo.mpi_conservation_parameter_type, down, 0, 
                 recvUUp.data(),   recvUUp.size(),   mPIInfo.mpi_conservation_parameter_type, up,   0, 
                 MPI_COMM_WORLD, &st);
    MPI_Sendrecv(sendUUp.data(),   sendUUp.size(),   mPIInfo.mpi_conservation_parameter_type, up,   0, 
                 recvUDown.data(), recvUDown.size(), mPIInfo.mpi_conservation_parameter_type, down, 0, 
                 MPI_COMM_WORLD, &st);

    for (int i = 0; i < localSizeX; i++) {
        for (int j = 0; j < mPIInfo.buffer; j++) {
            U[j                            + i * localSizeY] = recvUDown[j + i * mPIInfo.buffer];
            U[j + localNy + mPIInfo.buffer + i * localSizeY] = recvUUp[  j + i * mPIInfo.buffer];
        }
    }
}


void IdealMHD2DMPI::sendrecv_U(thrust::device_vector<ConservationParameter>& U, MPIInfo& mPIInfo)
{
    sendrecv_U_x(U, mPIInfo);
    sendrecv_U_y(U, mPIInfo);
}

