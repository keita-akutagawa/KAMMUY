#ifndef MPI_INTERFACE_H
#define MPI_INTERFACE_H

#include <mpi.h>
#include "const.hpp" 
#include "reload_particles_data_struct.hpp"
#include "../PIC2D_multiGPU/mpi.hpp"


namespace Interface2DMPI
{
    struct MPIInfo
    {
        int rank = 0;
        int procs = 0;
        int gridX, gridY = 0;
        int localGridX, localGridY = 0;
        int localNx = 0, localNy = 0; 
        int buffer = 0;
        int localSizeX = 0, localSizeY = 0;

        MPI_Datatype mpi_reloadParticlesDataType;

        __host__ __device__
        int getRank(int dx, int dy);

        __host__ __device__
        bool isInside(int globalX, int globalY);

        __host__ __device__
        int globalToLocal(int globalX, int globalY);
    }; 

    void setupInfo(MPIInfo& mPIInfo, int buffer, int gridX, int gridY);

    void sendrecv_reloadParticlesData_x(
        thrust::device_vector<ReloadParticlesData> reloadParticlesDataSpecies, 
        MPIInfo& mPIInfo
    );
}

#endif
