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

        unsigned int numForSendParticlesIonLeft = 0;
        unsigned int numForSendParticlesIonRight = 0;
        unsigned int numForRecvParticlesIonLeft = 0;
        unsigned int numForRecvParticlesIonRight = 0;

        unsigned int numForSendParticlesElectronLeft = 0;
        unsigned int numForSendParticlesElectronRight = 0;
        unsigned int numForRecvParticlesElectronLeft = 0;
        unsigned int numForRecvParticlesElectronRight = 0;

        MPI_Datatype mpi_particleType;
        MPI_Datatype mpi_fieldType;
        MPI_Datatype mpi_zerothMomentType;
        MPI_Datatype mpi_firstMomentType;


        __host__ __device__
        int getRank(int dx);

        __host__ __device__
        bool isInside(int globalX);

        __device__
        int globalToLocal(int globalX, int globalY);
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
        MPI_Datatype mpi_dataType)
    {
        int localNx = mPIInfo.localNx;

        int left  = mPIInfo.getRank(-1);
        int right = mPIInfo.getRank(1);
        MPI_Status st;

        FieldType* d_field = thrust::raw_pointer_cast(field.data());
        FieldType* d_sendFieldLeft = thrust::raw_pointer_cast(sendFieldLeft.data());
        FieldType* d_sendFieldRight = thrust::raw_pointer_cast(sendFieldRight.data());
        for (int i = 0; i < mPIInfo.buffer; i++) {
            cudaMemcpy(
                d_sendFieldLeft + i * PIC2DConst::ny, 
                d_field + (mPIInfo.buffer + i) * PIC2DConst::ny + mPIInfo.buffer,
                PIC2DConst::ny * sizeof(FieldType),
                cudaMemcpyDeviceToDevice
            );
            cudaMemcpy(
                d_sendFieldRight + i * PIC2DConst::ny,
                d_field + (localNx + i) * PIC2DConst::ny + mPIInfo.buffer,
                PIC2DConst::ny * sizeof(FieldType),
                cudaMemcpyDeviceToDevice
            );
        }

        MPI_Sendrecv(thrust::raw_pointer_cast(sendFieldLeft.data()),  sendFieldLeft.size(),  mpi_dataType, left,  0, 
                    thrust::raw_pointer_cast(recvFieldRight.data()), recvFieldRight.size(), mpi_dataType, right, 0, 
                    MPI_COMM_WORLD, &st);
        MPI_Sendrecv(thrust::raw_pointer_cast(sendFieldRight.data()), sendFieldRight.size(), mpi_dataType, right, 0, 
                    thrust::raw_pointer_cast(recvFieldLeft.data()),  recvFieldLeft.size(),  mpi_dataType, left,  0, 
                    MPI_COMM_WORLD, &st);

        FieldType* d_recvFieldLeft = thrust::raw_pointer_cast(recvFieldLeft.data());
        FieldType* d_recvFieldRight = thrust::raw_pointer_cast(recvFieldRight.data());
        for (int i = 0; i < mPIInfo.buffer; i++) {
            cudaMemcpy(
                d_field + i * PIC2DConst::ny + mPIInfo.buffer,
                d_recvFieldLeft + i * PIC2DConst::ny,
                PIC2DConst::ny * sizeof(FieldType),
                cudaMemcpyDeviceToDevice
            ); 
            cudaMemcpy(
                d_field + (localNx + mPIInfo.buffer + i) * PIC2DConst::ny + mPIInfo.buffer,
                d_recvFieldRight + i * PIC2DConst::ny,
                PIC2DConst::ny * sizeof(FieldType),
                cudaMemcpyDeviceToDevice
            ); 
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


    void sendrecv_numParticle_x(
        const unsigned int& numForSendParticlesSpeciesLeft, 
        const unsigned int& numForSendParticlesSpeciesRight, 
        unsigned int& numForRecvParticlesSpeciesLeft, 
        unsigned int& numForRecvParticlesSpeciesRight, 
        MPIInfo& mPIInfo
    );

    void sendrecv_particle_x(
        thrust::device_vector<Particle>& sendParticlesSpeciesLeft,
        thrust::device_vector<Particle>& sendParticlesSpeciesRight,  
        thrust::device_vector<Particle>& recvParticlesSpeciesLeft,
        thrust::device_vector<Particle>& recvParticlesSpeciesRight,
        const unsigned int& numForSendParticlesSpeciesLeft, 
        const unsigned int& numForSendParticlesSpeciesRight, 
        const unsigned int& numForRecvParticlesSpeciesLeft, 
        const unsigned int& numForRecvParticlesSpeciesRight, 
        MPIInfo& mPIInfo
    );
    
}

#endif


