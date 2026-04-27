#include "boundary.hpp"


BoundaryPIC::BoundaryPIC()
    : bufferParticlesSpecies(max(PIC2DConst::nx, PIC2DConst::ny) * max(PIC2DConst::numberDensityIon, PIC2DConst::numberDensityElectron) * 10)
{
}

