#ifndef PIC_GRID_PARAMETER_HPP 
#define PIC_GRID_PARAMETER_HPP

#include "type.hpp"
#include "../utils/json_function.hpp"


struct PICGridParameter 
{
    const PICUnsignedInt BUFFER; 

    const PICUnsignedInt NX; 
    const PICUnsignedInt NY; 

    const PICFloat DX; 
    const PICFloat DY; 

    const PICFloat XMIN; 
    const PICFloat YMIN;
    const PICFloat XMAX; 
    const PICFloat YMAX; 


    PICGridParameter(
        const nlohmann::json& gridJSON
    ) 
      : BUFFER(getRequiredForJSON<PICUnsignedInt>(gridJSON, "buffer")), 
        
        NX(getRequiredForJSON<PICUnsignedInt>(gridJSON, "nx")), 
        NY(getRequiredForJSON<PICUnsignedInt>(gridJSON, "ny")), 

        DX(getRequiredForJSON<PICFloat>(gridJSON, "dx")), 
        DY(getRequiredForJSON<PICFloat>(gridJSON, "dy")), 

        XMIN(getRequiredForJSON<PICFloat>(gridJSON, "xmin")), 
        YMIN(getRequiredForJSON<PICFloat>(gridJSON, "ymin")), 
        XMAX(getRequiredForJSON<PICFloat>(gridJSON, "xmax")), 
        YMAX(getRequiredForJSON<PICFloat>(gridJSON, "ymax")) 
    {   
        
    }
};

#endif 
