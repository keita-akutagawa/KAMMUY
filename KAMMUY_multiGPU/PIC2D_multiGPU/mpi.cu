#include "mpi.hpp"


int PIC2DMPI::MPIInfo::getRank(int dx, int dy)
{
    int rankX = (localGridX + dx + gridX) % gridX;
    int rankY = (localGridY + dy + gridY) % gridY;
    return rankY + rankX * gridY;
}


bool PIC2DMPI::MPIInfo::isInside(int globalX, int globalY)
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


int PIC2DMPI::MPIInfo::globalToLocal(int globalX, int globalY)
{
    int startX = localNx * localGridX;
    int x = globalX - startX;

    int startY = localNy * localGridY;
    int y = globalY - startY;

    return y + buffer + (x + buffer) * localSizeY;
}


void PIC2DMPI::setupInfo(MPIInfo& mPIInfo, int buffer, int gridX, int gridY)
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
    mPIInfo.localNx = PIC2DConst::nx / mPIInfo.gridX;
    mPIInfo.localNy = PIC2DConst::ny / mPIInfo.gridY;
    mPIInfo.buffer = buffer;
    mPIInfo.localSizeX = mPIInfo.localNx + 2 * mPIInfo.buffer;
    mPIInfo.localSizeY = mPIInfo.localNy + 2 * mPIInfo.buffer;


    int block_lengths_particle[12] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    MPI_Aint offsets_particle[12];
    offsets_particle[0]  = offsetof(Particle, x);
    offsets_particle[1]  = offsetof(Particle, y);
    offsets_particle[2]  = offsetof(Particle, z);
    offsets_particle[3]  = offsetof(Particle, vx);
    offsets_particle[4]  = offsetof(Particle, vy);
    offsets_particle[5]  = offsetof(Particle, vz);
    offsets_particle[6]  = offsetof(Particle, gamma);
    offsets_particle[7]  = offsetof(Particle, isExist);
    offsets_particle[8]  = offsetof(Particle, isMPISendLeft);
    offsets_particle[9]  = offsetof(Particle, isMPISendRight);
    offsets_particle[10] = offsetof(Particle, isMPISendDown);
    offsets_particle[11] = offsetof(Particle, isMPISendUp);

    MPI_Datatype types_particle[12] = {
        MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, 
        MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, 
        MPI_FLOAT, MPI_C_BOOL, 
        MPI_C_BOOL, MPI_C_BOOL, MPI_C_BOOL, MPI_C_BOOL
    };

    MPI_Type_create_struct(12, block_lengths_particle, offsets_particle, types_particle, &mPIInfo.mpi_particleType);
    MPI_Type_commit(&mPIInfo.mpi_particleType);

    // MagneticField, ElectricField, CurrentField共通
    int block_lengths_field[3] = {1, 1, 1};
    MPI_Aint offsets_field[3];
    offsets_field[0] = offsetof(MagneticField, bX);
    offsets_field[1] = offsetof(MagneticField, bY);
    offsets_field[2] = offsetof(MagneticField, bZ);
    MPI_Datatype types_field[3] = {MPI_FLOAT, MPI_FLOAT, MPI_FLOAT};
    MPI_Type_create_struct(3, block_lengths_field, offsets_field, types_field, &mPIInfo.mpi_fieldType);
    MPI_Type_commit(&mPIInfo.mpi_fieldType);

    int block_lengths_zerothMoment[1] = {1};
    MPI_Aint offsets_zerothMoment[1];
    offsets_zerothMoment[0] = offsetof(ZerothMoment, n);
    MPI_Datatype types_zerothMoment[1] = {MPI_FLOAT};
    MPI_Type_create_struct(1, block_lengths_zerothMoment, offsets_zerothMoment, types_zerothMoment, &mPIInfo.mpi_zerothMomentType);
    MPI_Type_commit(&mPIInfo.mpi_zerothMomentType);

    int block_lengths_firstMoment[3] = {1, 1, 1};
    MPI_Aint offsets_firstMoment[3];
    offsets_firstMoment[0] = offsetof(FirstMoment, x);
    offsets_firstMoment[1] = offsetof(FirstMoment, y);
    offsets_firstMoment[2] = offsetof(FirstMoment, z);
    MPI_Datatype types_firstMoment[3] = {MPI_FLOAT, MPI_FLOAT, MPI_FLOAT};
    MPI_Type_create_struct(3, block_lengths_firstMoment, offsets_firstMoment, types_firstMoment, &mPIInfo.mpi_firstMomentType);
    MPI_Type_commit(&mPIInfo.mpi_firstMomentType);
}


//////////////////////////////////////////////////

// sendrecv(field用)はヘッダーファイルにある。
// templeteを使用したため

//////////////////////////////////////////////////

void PIC2DMPI::sendrecv_numParticle_x(
    const unsigned int& numForSendParticlesSpeciesLeft, 
    const unsigned int& numForSendParticlesSpeciesRight, 
    unsigned int& numForRecvParticlesSpeciesLeft, 
    unsigned int& numForRecvParticlesSpeciesRight, 
    MPIInfo& mPIInfo
)
{
    int left  = mPIInfo.getRank(-1, 0);
    int right = mPIInfo.getRank(1, 0);
    MPI_Status st;

    MPI_Sendrecv(
        &(numForSendParticlesSpeciesLeft), 
        1, 
        MPI_UNSIGNED,  
        left, 0, 
        &(numForRecvParticlesSpeciesRight), 
        1, 
        MPI_UNSIGNED, 
        right, 0, 
        MPI_COMM_WORLD, &st
    );

    MPI_Sendrecv(
        &(numForSendParticlesSpeciesRight), 
        1, 
        MPI_UNSIGNED, 
        right, 0, 
        &(numForRecvParticlesSpeciesLeft), 
        1, 
        MPI_UNSIGNED, 
        left, 0, 
        MPI_COMM_WORLD, &st
    );
}


void PIC2DMPI::sendrecv_particle_x(
    thrust::host_vector<Particle>& host_sendParticlesSpeciesLeft, 
    thrust::host_vector<Particle>& host_sendParticlesSpeciesRight, 
    thrust::host_vector<Particle>& host_recvParticlesSpeciesLeft, 
    thrust::host_vector<Particle>& host_recvParticlesSpeciesRight, 
    MPIInfo& mPIInfo
)
{
    int left  = mPIInfo.getRank(-1, 0);
    int right = mPIInfo.getRank(1, 0);
    MPI_Status st;


    unsigned int maxNumSendLeftRecvRightForProcs = max(
        host_sendParticlesSpeciesLeft.size(), 
        host_recvParticlesSpeciesRight.size()
    );
    unsigned int maxNumSendRightRecvLeftForProcs = max(
        host_sendParticlesSpeciesRight.size(), 
        host_recvParticlesSpeciesLeft.size()
    );

    unsigned int maxNumSendLeftRecvRight = 0, maxNumSendRightRecvLeft = 0;
    MPI_Allreduce(&maxNumSendLeftRecvRightForProcs, &maxNumSendLeftRecvRight, 1, MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&maxNumSendRightRecvLeftForProcs, &maxNumSendRightRecvLeft, 1, MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD);
    
    thrust::host_vector<Particle> sendbufLeft(maxNumSendLeftRecvRight);
    thrust::host_vector<Particle> recvbufRight(maxNumSendLeftRecvRight);
    thrust::host_vector<Particle> sendbufRight(maxNumSendRightRecvLeft);
    thrust::host_vector<Particle> recvbufLeft(maxNumSendRightRecvLeft);

    for (unsigned int i = 0; i < host_sendParticlesSpeciesLeft.size(); i++) {
        sendbufLeft[i] = host_sendParticlesSpeciesLeft[i];
    }
    for (unsigned int i = 0; i < host_sendParticlesSpeciesRight.size(); i++) {
        sendbufRight[i] = host_sendParticlesSpeciesRight[i];
    }
    

    MPI_Sendrecv(
        thrust::raw_pointer_cast(sendbufLeft.data()), 
        sendbufLeft.size(), 
        mPIInfo.mpi_particleType, 
        left, 0, 
        thrust::raw_pointer_cast(recvbufRight.data()), 
        recvbufRight.size(), 
        mPIInfo.mpi_particleType, 
        right, 0, 
        MPI_COMM_WORLD, &st
    );

    MPI_Sendrecv(
        thrust::raw_pointer_cast(sendbufRight.data()), 
        sendbufRight.size(), 
        mPIInfo.mpi_particleType, 
        right, 0, 
        thrust::raw_pointer_cast(recvbufLeft.data()),
        recvbufLeft.size(), 
        mPIInfo.mpi_particleType, 
        left, 0, 
        MPI_COMM_WORLD, &st
    );


    for (unsigned int i = 0; i < host_recvParticlesSpeciesLeft.size(); i++) {
        host_recvParticlesSpeciesLeft[i] = recvbufLeft[i];
    }
    for (unsigned int i = 0; i < host_recvParticlesSpeciesRight.size(); i++) {
        host_recvParticlesSpeciesRight[i] = recvbufRight[i];
    }
}


void PIC2DMPI::sendrecv_numParticle_corner(
    const unsigned int& numForSendParticlesSpeciesCornerLeftDown, 
    const unsigned int& numForSendParticlesSpeciesCornerRightDown, 
    const unsigned int& numForSendParticlesSpeciesCornerLeftUp, 
    const unsigned int& numForSendParticlesSpeciesCornerRightUp, 
    unsigned int& numForRecvParticlesSpeciesCornerLeftDown, 
    unsigned int& numForRecvParticlesSpeciesCornerRightDown, 
    unsigned int& numForRecvParticlesSpeciesCornerLeftUp, 
    unsigned int& numForRecvParticlesSpeciesCornerRightUp, 
    MPIInfo& mPIInfo
)
{
    int left  = mPIInfo.getRank(-1, 0);
    int right = mPIInfo.getRank(1, 0);
    MPI_Status st;

    MPI_Sendrecv(
        &(numForSendParticlesSpeciesCornerLeftDown), 
        1, 
        MPI_UNSIGNED,  
        left, 0, 
        &(numForRecvParticlesSpeciesCornerRightDown), 
        1, 
        MPI_UNSIGNED, 
        right, 0, 
        MPI_COMM_WORLD, &st
    );

    MPI_Sendrecv(
        &(numForSendParticlesSpeciesCornerRightDown), 
        1, 
        MPI_UNSIGNED, 
        right, 0, 
        &(numForRecvParticlesSpeciesCornerLeftDown), 
        1, 
        MPI_UNSIGNED, 
        left, 0, 
        MPI_COMM_WORLD, &st
    );

    MPI_Sendrecv(
        &(numForSendParticlesSpeciesCornerLeftUp), 
        1, 
        MPI_UNSIGNED,  
        left, 0, 
        &(numForRecvParticlesSpeciesCornerRightUp), 
        1, 
        MPI_UNSIGNED, 
        right, 0, 
        MPI_COMM_WORLD, &st
    );

    MPI_Sendrecv(
        &(numForSendParticlesSpeciesCornerRightUp), 
        1, 
        MPI_UNSIGNED, 
        right, 0, 
        &(numForRecvParticlesSpeciesCornerLeftUp), 
        1, 
        MPI_UNSIGNED, 
        left, 0, 
        MPI_COMM_WORLD, &st
    );
}


void PIC2DMPI::sendrecv_numParticle_y(
    const unsigned int& numForSendParticlesSpeciesDown, 
    const unsigned int& numForSendParticlesSpeciesUp, 
    unsigned int& numForRecvParticlesSpeciesDown, 
    unsigned int& numForRecvParticlesSpeciesUp, 
    MPIInfo& mPIInfo
)
{
    int down = mPIInfo.getRank(0, -1);
    int up   = mPIInfo.getRank(0, 1);
    MPI_Status st;


    MPI_Sendrecv(
        &(numForSendParticlesSpeciesDown), 
        1, 
        MPI_UNSIGNED,  
        down, 0, 
        &(numForRecvParticlesSpeciesUp), 
        1, 
        MPI_UNSIGNED, 
        up, 0, 
        MPI_COMM_WORLD, &st
    );

    MPI_Sendrecv(
        &(numForSendParticlesSpeciesUp), 
        1, 
        MPI_UNSIGNED, 
        up, 0, 
        &(numForRecvParticlesSpeciesDown), 
        1, 
        MPI_UNSIGNED, 
        down, 0, 
        MPI_COMM_WORLD, &st
    );
}


void PIC2DMPI::sendrecv_particle_y(
    thrust::host_vector<Particle>& host_sendParticlesSpeciesDown, 
    thrust::host_vector<Particle>& host_sendParticlesSpeciesUp, 
    thrust::host_vector<Particle>& host_recvParticlesSpeciesDown, 
    thrust::host_vector<Particle>& host_recvParticlesSpeciesUp, 
    MPIInfo& mPIInfo
)
{
    int down  = mPIInfo.getRank(0, -1);
    int up = mPIInfo.getRank(0, 1);
    MPI_Status st;


    unsigned int maxNumSendDownRecvUpForProcs = max(
        host_sendParticlesSpeciesDown.size(), 
        host_recvParticlesSpeciesUp.size()
    );
    unsigned int maxNumSendUpRecvDownForProcs = max(
        host_sendParticlesSpeciesUp.size(), 
        host_recvParticlesSpeciesDown.size()
    );

    unsigned int maxNumSendDownRecvUp = 0, maxNumSendUpRecvDown = 0;
    MPI_Allreduce(&maxNumSendDownRecvUpForProcs, &maxNumSendDownRecvUp, 1, MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&maxNumSendUpRecvDownForProcs, &maxNumSendUpRecvDown, 1, MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD);
   

    thrust::host_vector<Particle> sendbufDown(maxNumSendDownRecvUp);
    thrust::host_vector<Particle> recvbufUp(maxNumSendDownRecvUp);
    thrust::host_vector<Particle> sendbufUp(maxNumSendUpRecvDown);
    thrust::host_vector<Particle> recvbufDown(maxNumSendUpRecvDown);
    

    for (unsigned int i = 0; i < host_sendParticlesSpeciesDown.size(); i++) {
        sendbufDown[i] = host_sendParticlesSpeciesDown[i];
    }
    for (unsigned int i = 0; i < host_sendParticlesSpeciesUp.size(); i++) {
        sendbufUp[i] = host_sendParticlesSpeciesUp[i];
    }


    MPI_Sendrecv(
        thrust::raw_pointer_cast(sendbufDown.data()), 
        sendbufDown.size(), 
        mPIInfo.mpi_particleType, 
        down, 0, 
        thrust::raw_pointer_cast(recvbufUp.data()), 
        recvbufUp.size(), 
        mPIInfo.mpi_particleType, 
        up, 0, 
        MPI_COMM_WORLD, &st
    );

    MPI_Sendrecv(
        thrust::raw_pointer_cast(sendbufUp.data()), 
        sendbufUp.size(), 
        mPIInfo.mpi_particleType, 
        up, 0, 
        thrust::raw_pointer_cast(recvbufDown.data()),
        recvbufDown.size(), 
        mPIInfo.mpi_particleType, 
        down, 0, 
        MPI_COMM_WORLD, &st
    );


    for (unsigned int i = 0; i < host_recvParticlesSpeciesDown.size(); i++) {
        host_recvParticlesSpeciesDown[i] = recvbufDown[i];
    }
    for (unsigned int i = 0; i < host_recvParticlesSpeciesUp.size(); i++) {
        host_recvParticlesSpeciesUp[i] = recvbufUp[i];
    }
}

