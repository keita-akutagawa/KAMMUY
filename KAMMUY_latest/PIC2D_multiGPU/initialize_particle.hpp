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
        float xmin, 
        float xmax, 
        int seed, 
        thrust::device_vector<Particle>& particlesSpecies
    );

    void uniformForPosition_y(
        unsigned long long nStart, 
        unsigned long long nEnd, 
        float ymin, 
        float ymax, 
        int seed, 
        thrust::device_vector<Particle>& particlesSpecies
    );

    void maxwellDistributionForVelocity(
        float bulkVxSpecies, 
        float bulkVySpecies, 
        float bulkVzSpecies, 
        float vxThSpecies, 
        float vyThSpecies, 
        float vzThSpecies, 
        unsigned long long nStart, 
        unsigned long long nEnd, 
        int seed, 
        thrust::device_vector<Particle>& particlesSpecies
    );

    void uniformForPosition_xy_maxwellDistributionForVelocity_eachCell(
        float xmin, float xmax, float ymin, float ymax, 
        float bulkVxSpecies, float bulkVySpecies, float bulkVzSpecies, 
        float vxThSpecies, float vyThSpecies, float vzThSpecies, 
        unsigned long long nStart, unsigned long long nEnd, 
        int seed, 
        thrust::device_vector<Particle>& particlesSpecies
    );

    void harrisForPosition_y(
        unsigned long long nStart, 
        unsigned long long nEnd, 
        int seed, 
        float sheatThickness, 
        thrust::device_vector<Particle>& particlesSpecies
    );

    void harrisBackgroundForPosition_y(
        unsigned long long nStart, 
        unsigned long long nEnd, 
        int seed, 
        float sheatThickness, 
        thrust::device_vector<Particle>& particlesSpecies
    );

private:

};

