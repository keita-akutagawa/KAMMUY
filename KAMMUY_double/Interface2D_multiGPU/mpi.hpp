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
        int gridX = 0;
        int localGridX = 0;

        __host__ __device__
        int getRank(int dx);
    }; 

    void setupInfo(MPIInfo& mPIInfo);
}

#endif
