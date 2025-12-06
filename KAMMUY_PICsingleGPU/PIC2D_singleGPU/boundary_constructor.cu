#include "boundary.hpp"


BoundaryPIC::BoundaryPIC()
    : sendParticlesSpeciesYDown(PIC2DConst::nx * PIC2DConst::numberDensityIon * 10), 
      sendParticlesSpeciesYUp(PIC2DConst::nx * PIC2DConst::numberDensityIon * 10), 
      recvParticlesSpeciesYDown(PIC2DConst::nx * PIC2DConst::numberDensityIon * 10), 
      recvParticlesSpeciesYUp(PIC2DConst::nx * PIC2DConst::numberDensityIon * 10)
{
}

