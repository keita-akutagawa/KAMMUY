#include "mpi.hpp"


int PIC2DMPI::MPIInfo::getRank(int dx)
{
    int rankX = (localGridX + dx + gridX) % gridX;
    return rankX;
}


bool PIC2DMPI::MPIInfo::isInside(int globalX)
{
    int startX = localNx * localGridX;
    int endX = startX + localNx;

    if (globalX < startX) return false;
    if (globalX >= endX) return false;

    return true;
}


__device__
int PIC2DMPI::MPIInfo::globalToLocal(int globalX, int globalY)
{
    int startX = localNx * localGridX;
    int x = globalX - startX;

    int y = globalY;

    return y + (x + buffer) * PIC2DConst::device_ny;
}


void PIC2DMPI::setupInfo(MPIInfo& mPIInfo, int buffer, int mpiBufNumParticles)
{
    int rank = 0, procs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    mPIInfo.rank = rank;
    mPIInfo.procs = procs;
    mPIInfo.gridX = procs;
    mPIInfo.localGridX = rank;
    mPIInfo.localNx = PIC2DConst::nx / mPIInfo.gridX;
    mPIInfo.buffer = buffer;
    mPIInfo.localSizeX = mPIInfo.localNx + 2 * mPIInfo.buffer;
    mPIInfo.mpiBufNumParticles = mpiBufNumParticles; 


    int block_lengths_particle[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    MPI_Aint offsets_particle[8];
    offsets_particle[0]  = offsetof(Particle, x);
    offsets_particle[1]  = offsetof(Particle, y);
    offsets_particle[2]  = offsetof(Particle, z);
    offsets_particle[3]  = offsetof(Particle, vx);
    offsets_particle[4]  = offsetof(Particle, vy);
    offsets_particle[5]  = offsetof(Particle, vz);
    offsets_particle[6]  = offsetof(Particle, gamma);
    offsets_particle[7]  = offsetof(Particle, isExist);

    MPI_Datatype types_particle[8] = {
        MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, 
        MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, 
        MPI_FLOAT, MPI_C_BOOL
    };

    MPI_Type_create_struct(8, block_lengths_particle, offsets_particle, types_particle, &mPIInfo.mpi_particleType);
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

void PIC2DMPI::sendrecv_magneticField_x(
    thrust::device_vector<MagneticField>& B, 
    thrust::device_vector<MagneticField>& sendMagneticFieldLeft, 
    thrust::device_vector<MagneticField>& sendMagneticFieldRight, 
    thrust::device_vector<MagneticField>& recvMagneticFieldLeft, 
    thrust::device_vector<MagneticField>& recvMagneticFieldRight, 
    MPIInfo& mPIInfo
)
{
    PIC2DMPI::sendrecv_field_x(
        B, 
        sendMagneticFieldLeft, sendMagneticFieldRight, 
        recvMagneticFieldLeft, recvMagneticFieldRight, 
        mPIInfo, mPIInfo.mpi_fieldType
    );
}


void PIC2DMPI::sendrecv_electricField_x(
    thrust::device_vector<ElectricField>& E, 
    thrust::device_vector<ElectricField>& sendElectricFieldLeft, 
    thrust::device_vector<ElectricField>& sendElectricFieldRight, 
    thrust::device_vector<ElectricField>& recvElectricFieldLeft, 
    thrust::device_vector<ElectricField>& recvElectricFieldRight,
    MPIInfo& mPIInfo
)
{
    PIC2DMPI::sendrecv_field_x(
        E, 
        sendElectricFieldLeft, sendElectricFieldRight, 
        recvElectricFieldLeft, recvElectricFieldRight, 
        mPIInfo, mPIInfo.mpi_fieldType
    );
}


void PIC2DMPI::sendrecv_currentField_x(
    thrust::device_vector<CurrentField>& current,  
    thrust::device_vector<CurrentField>& sendCurrentFieldLeft, 
    thrust::device_vector<CurrentField>& sendCurrentFieldRight, 
    thrust::device_vector<CurrentField>& recvCurrentFieldLeft, 
    thrust::device_vector<CurrentField>& recvCurrentFieldRight, 
    MPIInfo& mPIInfo
)
{
    PIC2DMPI::sendrecv_field_x(
        current, 
        sendCurrentFieldLeft, sendCurrentFieldRight, 
        recvCurrentFieldLeft, recvCurrentFieldRight, 
        mPIInfo, mPIInfo.mpi_fieldType
    );
}


void PIC2DMPI::sendrecv_zerothMoment_x(
    thrust::device_vector<ZerothMoment>& zerothMoment,  
    thrust::device_vector<ZerothMoment>& sendZerothMomentLeft, 
    thrust::device_vector<ZerothMoment>& sendZerothMomentRight, 
    thrust::device_vector<ZerothMoment>& recvZerothMomentLeft, 
    thrust::device_vector<ZerothMoment>& recvZerothMomentRight, 
    MPIInfo& mPIInfo
)
{
    PIC2DMPI::sendrecv_field_x(
        zerothMoment, 
        sendZerothMomentLeft, sendZerothMomentRight, 
        recvZerothMomentLeft, recvZerothMomentRight, 
        mPIInfo, mPIInfo.mpi_zerothMomentType
    );
}


void PIC2DMPI::sendrecv_firstMoment_x(
    thrust::device_vector<FirstMoment>& firstMoment,  
    thrust::device_vector<FirstMoment>& sendFirstMomentLeft, 
    thrust::device_vector<FirstMoment>& sendFirstMomentRight, 
    thrust::device_vector<FirstMoment>& recvFirstMomentLeft, 
    thrust::device_vector<FirstMoment>& recvFirstMomentRight, 
    MPIInfo& mPIInfo
)
{
    PIC2DMPI::sendrecv_field_x(
        firstMoment, 
        sendFirstMomentLeft, sendFirstMomentRight, 
        recvFirstMomentLeft, recvFirstMomentRight, 
        mPIInfo, mPIInfo.mpi_firstMomentType
    );
}

//////////////////////////////////////////////////

void PIC2DMPI::sendrecv_numParticle_x(
    const unsigned int& numForSendParticlesSpeciesLeft, 
    const unsigned int& numForSendParticlesSpeciesRight, 
    unsigned int& numForRecvParticlesSpeciesLeft, 
    unsigned int& numForRecvParticlesSpeciesRight, 
    MPIInfo& mPIInfo
)
{
    int left  = mPIInfo.getRank(-1);
    int right = mPIInfo.getRank(1);
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
    thrust::device_vector<Particle>& sendParticlesSpeciesLeft, 
    thrust::device_vector<Particle>& sendParticlesSpeciesRight, 
    thrust::device_vector<Particle>& recvParticlesSpeciesLeft, 
    thrust::device_vector<Particle>& recvParticlesSpeciesRight, 
    const unsigned int& numForSendParticlesSpeciesLeft, 
    const unsigned int& numForSendParticlesSpeciesRight, 
    const unsigned int& numForRecvParticlesSpeciesLeft, 
    const unsigned int& numForRecvParticlesSpeciesRight, 
    MPIInfo& mPIInfo
)
{
    int left  = mPIInfo.getRank(-1);
    int right = mPIInfo.getRank(1);
    MPI_Status st;


    unsigned int maxNumSendLeftRecvRightForProcs = max(
        numForSendParticlesSpeciesLeft, 
        numForRecvParticlesSpeciesRight 
    );
    unsigned int maxNumSendRightRecvLeftForProcs = max(
        numForSendParticlesSpeciesRight, 
        numForRecvParticlesSpeciesLeft
    );

    unsigned int maxNumSendLeftRecvRight = 0, maxNumSendRightRecvLeft = 0;
    MPI_Allreduce(&maxNumSendLeftRecvRightForProcs, &maxNumSendLeftRecvRight, 1, MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&maxNumSendRightRecvLeftForProcs, &maxNumSendRightRecvLeft, 1, MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD);

    MPI_Sendrecv(
        thrust::raw_pointer_cast(sendParticlesSpeciesLeft.data()), 
        numForSendParticlesSpeciesLeft, 
        mPIInfo.mpi_particleType, 
        left, 0, 
        thrust::raw_pointer_cast(recvParticlesSpeciesRight.data()), 
        numForRecvParticlesSpeciesRight, 
        mPIInfo.mpi_particleType, 
        right, 0, 
        MPI_COMM_WORLD, &st
    );

    MPI_Sendrecv(
        thrust::raw_pointer_cast(sendParticlesSpeciesRight.data()), 
        numForSendParticlesSpeciesRight, 
        mPIInfo.mpi_particleType, 
        right, 0, 
        thrust::raw_pointer_cast(recvParticlesSpeciesLeft.data()),
        numForRecvParticlesSpeciesLeft, 
        mPIInfo.mpi_particleType, 
        left, 0, 
        MPI_COMM_WORLD, &st
    );
}

