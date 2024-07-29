#include "KAMMUY2D.hpp"


KAMMUY2D::KAMMUY2D()
    : UPast(IdealMHD2DConst::nx * IdealMHD2DConst::ny)
{
}

void KAMMUY2D::oneStep(

)
{
    idealMHD2D.oneStepRK2();


}


