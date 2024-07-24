#include "interface.hpp"
#include <cmath>


Interface2D::Interface2D()
    :  interlockingFunctionX(Interface2DConst::interfaceLength), 
       interlockingFunctionY(Interface2DConst::interfaceLength), 
       host_interlockingFunctionX(Interface2DConst::interfaceLength), 
       host_interlockingFunctionY(Interface2DConst::interfaceLength)
{
    for(int i = 0; Interface2DConst::interfaceLength; i++) {
        host_interlockingFunctionX[i] = 0.5f * (
            1.0f + cos(Interface2DConst::PI * (i - 0) / (Interface2DConst::interfaceLength - i))
        );
        host_interlockingFunctionY[i] = 0.5f * (
            1.0f + cos(Interface2DConst::PI * (i - 0) / (Interface2DConst::interfaceLength - i))
        );
    }

    interlockingFunctionX = host_interlockingFunctionX;
    interlockingFunctionY = host_interlockingFunctionY;
}


void Interface2D::sendMHDtoPIC_MagneticField(
    thrust::device_vector<ConservationParameter>& U, 
    thrust::device_vector<MagneticField>& B
)
{

}



