#include "mpi.hpp"


int IdealMHD2DMPI::MPIInfo::getRank(int dx)
{
    int rankX = (localGridX + dx + gridX) % gridX;
    return rankX;
}


bool IdealMHD2DMPI::MPIInfo::isInside(int globalX)
{
    int startX = localNx * localGridX;
    int endX = startX + localNx;

    if (globalX < startX) return false;
    if (globalX >= endX) return false;

    return true;
}


__device__
int IdealMHD2DMPI::MPIInfo::globalToLocal(int globalX, int globalY)
{
    int startX = localNx * localGridX;
    int x = globalX - startX;

    int y = globalY;

    return y + (x + buffer) * IdealMHD2DConst::device_ny;
}


void IdealMHD2DMPI::setupInfo(IdealMHD2DMPI::MPIInfo& mPIInfo, int buffer)
{
    int rank = 0, procs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    mPIInfo.rank = rank;
    mPIInfo.procs = procs;
    mPIInfo.gridX = procs;
    mPIInfo.localGridX = rank; 
    mPIInfo.localNx = IdealMHD2DConst::nx / mPIInfo.gridX;
    mPIInfo.buffer = buffer;
    mPIInfo.localSizeX = mPIInfo.localNx + 2 * mPIInfo.buffer;


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

}


void IdealMHD2DMPI::sendrecv_U_x(
    thrust::device_vector<ConservationParameter>& U, 
    thrust::device_vector<ConservationParameter>& sendULeft, 
    thrust::device_vector<ConservationParameter>& sendURight, 
    thrust::device_vector<ConservationParameter>& recvULeft, 
    thrust::device_vector<ConservationParameter>& recvURight, 
    IdealMHD2DMPI::MPIInfo& mPIInfo)
{
    int localNx = mPIInfo.localNx;

    int left = mPIInfo.getRank(-1);
    int right = mPIInfo.getRank(1);
    MPI_Status st;

    for (int i = 0; i < mPIInfo.buffer; i++) {
        for (int j = 0; j < IdealMHD2DConst::ny; j++) {
            sendULeft[ j + i * IdealMHD2DConst::ny] = U[j + (mPIInfo.buffer + i) * IdealMHD2DConst::ny];
            sendURight[j + i * IdealMHD2DConst::ny] = U[j + (localNx + i)        * IdealMHD2DConst::ny];
        }
    }

    MPI_Sendrecv(thrust::raw_pointer_cast(sendULeft.data()),  sendULeft.size(),  mPIInfo.mpi_conservation_parameter_type, left,  0, 
                 thrust::raw_pointer_cast(recvURight.data()), recvURight.size(), mPIInfo.mpi_conservation_parameter_type, right, 0, 
                 MPI_COMM_WORLD, &st);
    MPI_Sendrecv(thrust::raw_pointer_cast(sendURight.data()), sendURight.size(), mPIInfo.mpi_conservation_parameter_type, right, 0, 
                 thrust::raw_pointer_cast(recvULeft.data()),  recvULeft.size(),  mPIInfo.mpi_conservation_parameter_type, left,  0, 
                 MPI_COMM_WORLD, &st);

    for (int i = 0; i < mPIInfo.buffer; i++) {
        for (int j = 0; j < IdealMHD2DConst::ny; j++) {
            U[j + i                              * IdealMHD2DConst::ny] = recvULeft[ j + i * IdealMHD2DConst::ny];
            U[j + (localNx + mPIInfo.buffer + i) * IdealMHD2DConst::ny] = recvURight[j + i * IdealMHD2DConst::ny];
        }
    }
}




