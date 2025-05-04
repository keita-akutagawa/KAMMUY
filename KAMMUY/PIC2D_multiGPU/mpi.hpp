#ifndef MPI_PIC_H
#define MPI_PIC_H

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <mpi.h>
#include "const.hpp"
#include "field_parameter_struct.hpp"
#include "moment_struct.hpp" 
#include "particle_struct.hpp"


namespace PIC2DMPI
{

    struct MPIInfo
    {
        int rank = 0;
        int procs = 0;
        int gridX = 0;
        int localGridX = 0;
        int localNx = 0; 
        int buffer = 0;
        int localSizeX = 0;
        int mpiBufNumParticles = 0; 

        unsigned long long existNumIonPerProcs = 0;
        unsigned long long existNumElectronPerProcs = 0;
        unsigned long long totalNumIonPerProcs = 0;
        unsigned long long totalNumElectronPerProcs = 0;

        float xminForProcs = 0.0f;
        float xmaxForProcs = 0.0f;

        unsigned long long numForSendParticlesIonLeft = 0;
        unsigned long long numForSendParticlesIonRight = 0;
        unsigned long long numForRecvParticlesIonLeft = 0;
        unsigned long long numForRecvParticlesIonRight = 0;

        unsigned long long numForSendParticlesElectronLeft = 0;
        unsigned long long numForSendParticlesElectronRight = 0;
        unsigned long long numForRecvParticlesElectronLeft = 0;
        unsigned long long numForRecvParticlesElectronRight = 0;

        MPI_Datatype mpi_particleType;
        MPI_Datatype mpi_fieldType;
        MPI_Datatype mpi_zerothMomentType;
        MPI_Datatype mpi_firstMomentType;
        MPI_Datatype mpi_secondMomentType;


        __host__ __device__
        int getRank(int dx);

        __host__ __device__
        bool isInside(int globalX);

        __device__
        unsigned long long globalToLocal(int globalX, int globalY);
    };


    void setupInfo(MPIInfo& mPIInfo, int buffer, int mpiBufNumParticles);


    template <typename FieldType>
    void sendrecv_field_x(
        thrust::device_vector<FieldType>& field, 
        thrust::device_vector<FieldType>& sendFieldLeft, 
        thrust::device_vector<FieldType>& sendFieldRight, 
        thrust::device_vector<FieldType>& recvFieldLeft, 
        thrust::device_vector<FieldType>& recvFieldRight, 
        MPIInfo& mPIInfo, 
        MPI_Datatype mpi_dataType
    )
    {
        int localNx = mPIInfo.localNx;

        int left  = mPIInfo.getRank(-1);
        int right = mPIInfo.getRank(1);
        MPI_Status st;

        for (int i = 0; i < mPIInfo.buffer; i++) {
            for (int j = 0; j < PIC2DConst::ny; j++) {
                sendFieldLeft[ j + i * PIC2DConst::ny] = field[j + (mPIInfo.buffer + i) * PIC2DConst::ny];
                sendFieldRight[j + i * PIC2DConst::ny] = field[j + (localNx + i)        * PIC2DConst::ny];
            }
        }
    
        MPI_Sendrecv(thrust::raw_pointer_cast(sendFieldLeft.data()),  sendFieldLeft.size(),  mpi_dataType, left,  0, 
                     thrust::raw_pointer_cast(recvFieldRight.data()), recvFieldRight.size(), mpi_dataType, right, 0, 
                     MPI_COMM_WORLD, &st);
        MPI_Sendrecv(thrust::raw_pointer_cast(sendFieldRight.data()), sendFieldRight.size(), mpi_dataType, right, 1, 
                     thrust::raw_pointer_cast(recvFieldLeft.data()),  recvFieldLeft.size(),  mpi_dataType, left,  1, 
                     MPI_COMM_WORLD, &st);
    
        for (int i = 0; i < mPIInfo.buffer; i++) {
            for (int j = 0; j < PIC2DConst::ny; j++) {
                field[j + i                              * PIC2DConst::ny] = recvFieldLeft[ j + i * PIC2DConst::ny];
                field[j + (localNx + mPIInfo.buffer + i) * PIC2DConst::ny] = recvFieldRight[j + i * PIC2DConst::ny];
            }
        }
    }


    void sendrecv_magneticField_x(
        thrust::device_vector<MagneticField>& B, 
        thrust::device_vector<MagneticField>& sendMagneticFieldLeft, 
        thrust::device_vector<MagneticField>& sendMagneticFieldRight, 
        thrust::device_vector<MagneticField>& recvMagneticFieldLeft, 
        thrust::device_vector<MagneticField>& recvMagneticFieldRight, 
        MPIInfo& mPIInfo
    ); 


    void sendrecv_electricField_x(
        thrust::device_vector<ElectricField>& E, 
        thrust::device_vector<ElectricField>& sendElectricFieldLeft, 
        thrust::device_vector<ElectricField>& sendElectricFieldRight, 
        thrust::device_vector<ElectricField>& recvElectricFieldLeft, 
        thrust::device_vector<ElectricField>& recvElectricFieldRight, 
        MPIInfo& mPIInfo
    ); 

    void sendrecv_currentField_x(
        thrust::device_vector<CurrentField>& current, 
        thrust::device_vector<CurrentField>& sendCurrentFieldLeft, 
        thrust::device_vector<CurrentField>& sendCurrentFieldRight, 
        thrust::device_vector<CurrentField>& recvCurrentFieldLeft, 
        thrust::device_vector<CurrentField>& recvCurrentFieldRight, 
        MPIInfo& mPIInfo
    ); 


    void sendrecv_zerothMoment_x(
        thrust::device_vector<ZerothMoment>& zerothMoment, 
        thrust::device_vector<ZerothMoment>& sendZerothMomentLeft, 
        thrust::device_vector<ZerothMoment>& sendZerothMomentRight, 
        thrust::device_vector<ZerothMoment>& recvZerothMomentLeft, 
        thrust::device_vector<ZerothMoment>& recvZerothMomentRight, 
        MPIInfo& mPIInfo
    ); 


    void sendrecv_firstMoment_x(
        thrust::device_vector<FirstMoment>& firstMoment, 
        thrust::device_vector<FirstMoment>& sendFirstMomentLeft, 
        thrust::device_vector<FirstMoment>& sendFirstMomentRight, 
        thrust::device_vector<FirstMoment>& recvFirstMomentLeft, 
        thrust::device_vector<FirstMoment>& recvFirstMomentRight, 
        MPIInfo& mPIInfo
    ); 

    void sendrecv_secondMoment_x(
        thrust::device_vector<SecondMoment>& secondMoment, 
        thrust::device_vector<SecondMoment>& sendSecondMomentLeft, 
        thrust::device_vector<SecondMoment>& sendSecondMomentRight, 
        thrust::device_vector<SecondMoment>& recvSecondMomentLeft, 
        thrust::device_vector<SecondMoment>& recvSecondMomentRight, 
        MPIInfo& mPIInfo
    ); 


    void sendrecv_numParticle_x(
        const unsigned long long& numForSendParticlesSpeciesLeft, 
        const unsigned long long& numForSendParticlesSpeciesRight, 
        unsigned long long& numForRecvParticlesSpeciesLeft, 
        unsigned long long& numForRecvParticlesSpeciesRight, 
        MPIInfo& mPIInfo
    );

    void sendrecv_particle_x(
        thrust::device_vector<Particle>& sendParticlesSpeciesLeft,
        thrust::device_vector<Particle>& sendParticlesSpeciesRight,  
        thrust::device_vector<Particle>& recvParticlesSpeciesLeft,
        thrust::device_vector<Particle>& recvParticlesSpeciesRight,
        const unsigned long long& numForSendParticlesSpeciesLeft, 
        const unsigned long long& numForSendParticlesSpeciesRight, 
        const unsigned long long& numForRecvParticlesSpeciesLeft, 
        const unsigned long long& numForRecvParticlesSpeciesRight, 
        MPIInfo& mPIInfo
    );
    
}

#endif


