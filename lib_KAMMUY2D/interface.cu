#include "interface.hpp"
#include <cmath>


Interface2D::Interface2D()
    :  interlockingFunction(interfaceLength), 
       host_interlockingFunction(interfaceLength)
{
    for(int i = 0; interfaceLength; i++) {
        host_interlockingFunction[i] = 0.5f * (1.0f + cos(PI * (i - 0) / (interfaceLength - i)));
    }

    interlockingFunction = host_interlockingFunction;
}




