#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "const.hpp"
#include "particle_struct.hpp"
#include "mpi.hpp" 


class InitializeParticle
{
private:
    PIC2DMPI::MPIInfo& mPIInfo;

public:
    InitializeParticle(PIC2DMPI::MPIInfo& mPIInfo);

    void uniformForPosition_x(
        unsigned long long nStart, 
        unsigned long long nEnd, 
        double xmin, 
        double xmax, 
        unsigned long long seed, 
        thrust::device_vector<Particle>& particlesSpecies
    );

    void uniformForPosition_y(
        unsigned long long nStart, 
        unsigned long long nEnd, 
        double ymin, 
        double ymax, 
        unsigned long long seed, 
        thrust::device_vector<Particle>& particlesSpecies
    );

    void maxwellDistributionForVelocity(
        double bulkVxSpecies, 
        double bulkVySpecies, 
        double bulkVzSpecies, 
        double vxThSpecies, 
        double vyThSpecies, 
        double vzThSpecies, 
        unsigned long long nStart, 
        unsigned long long nEnd, 
        unsigned long long seed, 
        thrust::device_vector<Particle>& particlesSpecies
    );

    void uniformForPosition_xy_maxwellDistributionForVelocity_eachCell(
        double xmin, double xmax, double ymin, double ymax, 
        double bulkVxSpecies, double bulkVySpecies, double bulkVzSpecies, 
        double vxThSpecies, double vyThSpecies, double vzThSpecies, 
        unsigned long long nStart, unsigned long long nEnd, 
        unsigned long long seed, 
        thrust::device_vector<Particle>& particlesSpecies
    );

    void harrisForPosition_y(
        unsigned long long nStart, 
        unsigned long long nEnd, 
        unsigned long long seed, 
        double sheatThickness, 
        thrust::device_vector<Particle>& particlesSpecies
    );

    void harrisBackgroundForPosition_y(
        unsigned long long nStart, 
        unsigned long long nEnd, 
        unsigned long long seed, 
        double sheatThickness, 
        thrust::device_vector<Particle>& particlesSpecies
    );

private:

};

