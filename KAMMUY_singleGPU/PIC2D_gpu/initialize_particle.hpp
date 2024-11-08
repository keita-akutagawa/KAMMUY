#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "const.hpp"
#include "particle_struct.hpp"


class InitializeParticle
{
private:

public:
    void uniformForPositionXY_maxwellDistributionForVelocity_detail(
        double xmin, double ymin,  
        double bulkVxSpecies, 
        double bulkVySpecies, 
        double bulkVzSpecies, 
        double vxThSpecies, 
        double vyThSpecies, 
        double vzThSpecies, 
        unsigned long long nStart, 
        unsigned long long nEnd, 
        int seed, 
        thrust::device_vector<Particle>& particlesSpecies
    );

    void uniformForPositionX(
        unsigned long long nStart, 
        unsigned long long nEnd, 
        int seed, 
        thrust::device_vector<Particle>& particlesSpecies
    );

    void uniformForPositionY(
        unsigned long long nStart, 
        unsigned long long nEnd, 
        int seed, 
        thrust::device_vector<Particle>& particlesSpecies
    );

    void uniformForPositionY_detail(
        unsigned long long nStart, 
        unsigned long long nEnd, 
        int seed, 
        double ymin, 
        double ymax, 
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
        int seed, 
        thrust::device_vector<Particle>& particlesSpecies
    );

    void harrisForPositionY(
        unsigned long long nStart, 
        unsigned long long nEnd, 
        int seed, 
        double sheatThickness, 
        thrust::device_vector<Particle>& particlesSpecies
    );

    void harrisBackgroundForPositionY(
        unsigned long long nStart, 
        unsigned long long nEnd, 
        int seed, 
        double sheatThickness, 
        thrust::device_vector<Particle>& particlesSpecies
    );

private:

};

