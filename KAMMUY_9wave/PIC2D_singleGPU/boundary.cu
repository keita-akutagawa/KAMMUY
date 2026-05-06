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

    if (PIC2DConst::existNumIon > PIC2DConst::totalNumIon) std::cout << "BROKEN" << std::endl;
    if (PIC2DConst::existNumElectron > PIC2DConst::totalNumElectron) std::cout << "BROKEN" << std::endl;
}

