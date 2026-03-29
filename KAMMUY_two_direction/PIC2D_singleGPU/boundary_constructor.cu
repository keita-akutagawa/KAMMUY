#include "boundary.hpp"


BoundaryPIC::BoundaryPIC()
    : bufferParticlesSpeciesX(PIC2DConst::nx * PIC2DConst::numberDensityIon * 10), 
      bufferParticlesSpeciesY(PIC2DConst::ny * PIC2DConst::numberDensityIon * 10)
{
}

