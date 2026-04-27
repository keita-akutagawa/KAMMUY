#include "boundary.hpp"


void BoundaryPIC::boundaryParticle(
    thrust::device_vector<Particle>& particlesIon, 
    unsigned long long& existNumIon,
    thrust::device_vector<Particle>& particlesElectron,
    unsigned long long& existNumElectron
)
{
    boundaryParticleSpecies(particlesIon, existNumIon);
    boundaryParticleSpecies(particlesElectron, existNumElectron);
}

