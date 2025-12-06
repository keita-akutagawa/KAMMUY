#ifndef MPI_H
#define MPI_H

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <mpi.h>
#include "const.hpp"
#include "conservation_parameter_struct.hpp"
#include "flux_struct.hpp"


namespace IdealMHD2DMPI 
{
    struct MPIInfo
    {
        int rank;
        int procs;
        int gridX;
        int localGridX;
        int localNx; 
        int buffer;
        int localSizeX; 

        MPI_Datatype mpi_conservation_parameter_type;


        __host__ __device__
        int getRank(int dx);

        __host__ __device__
        bool isInside(int globalX);

        __device__
        unsigned long long globalToLocal(int globalX, int globalY);
    };


    void setupInfo(IdealMHD2DMPI::MPIInfo& mPIInfo, int buffer);


    void sendrecv_U_x(
        thrust::device_vector<ConservationParameter>& U, 
        thrust::device_vector<ConservationParameter>& sendULeft, 
        thrust::device_vector<ConservationParameter>& sendURight, 
        thrust::device_vector<ConservationParameter>& recvULeft, 
        thrust::device_vector<ConservationParameter>& recvURight, 
        IdealMHD2DMPI::MPIInfo& mPIInfo
    );
}


#endif
