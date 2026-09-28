#ifndef INTERFACE_CONST_PARAMETER_HPP
#define INTERFACE_CONST_PARAMETER_HPP

#include "type.hpp"
#include "../utils/json_function.hpp"


struct InterfaceConstParameter
{
    const InterfaceFloat EPS; 
    const InterfaceFloat PI; 

    const InterfaceUnsignedInt CONVOLUTION_INTERVAL; 

    const InterfaceFloat DELTA_INTERLOCKING_FUNCTION; 


    InterfaceConstParameter(
        const nlohmann::json& InterfaceConstJSON
    ) 
      : EPS(getRequiredForJSON<InterfaceFloat>(InterfaceConstJSON, "EPS")),  
        PI(getRequiredForJSON<InterfaceFloat>(InterfaceConstJSON, "PI")),  
        
        CONVOLUTION_INTERVAL(getRequiredForJSON<InterfaceUnsignedInt>(InterfaceConstJSON, "convolution_interval")), 

        DELTA_INTERLOCKING_FUNCTION(getRequiredForJSON<InterfaceFloat>(InterfaceConstJSON, "delta_interlocking_function"))
    {
        
    }
};

#endif
