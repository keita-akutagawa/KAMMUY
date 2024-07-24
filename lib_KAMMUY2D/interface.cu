#include "interface.hpp"
#include <cmath>


Interface2D::Interface2D()
    :  interlockingFunctionX(interfaceLength), 
       interlockingFunctionY(interfaceLength), 
       host_interlockingFunctionX(interfaceLength), 
       host_interlockingFunctionY(interfaceLength)
{
    for(int i = 0; interfaceLength; i++) {
        host_interlockingFunctionX[i] = 0.5f * (1.0f + cos(PI * (i - 0) / (interfaceLength - i)));
        host_interlockingFunctionY[i] = 0.5f * (1.0f + cos(PI * (i - 0) / (interfaceLength - i)));
    }

    interlockingFunctionX = host_interlockingFunctionX;
    interlockingFunctionY = host_interlockingFunctionY;
}




