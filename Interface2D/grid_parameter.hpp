#ifndef INTERFACE_GRID_PARAMETER_HPP
#define INTERFACE_GRID_PARAMETER_HPP

#include "type.hpp"
#include "../utils/json_function.hpp"


struct InterfaceGridParameter
{
    const InterfaceUnsignedInt GRID_SIZE_RATIO; 

    const InterfaceUnsignedInt START_INDEX_IN_MHD_X;  
    const InterfaceUnsignedInt START_INDEX_IN_MHD_Y;  

    InterfaceGridParameter(
        const nlohmann::json& InterfaceGridJSON
    ) 
      : GRID_SIZE_RATIO(getRequiredForJSON<InterfaceUnsignedInt>(InterfaceGridJSON, "grid_size_ratio")),  
        
        START_INDEX_IN_MHD_X(getRequiredForJSON<InterfaceUnsignedInt>(InterfaceGridJSON, "start_index_in_mhd_x")),  
        START_INDEX_IN_MHD_Y(getRequiredForJSON<InterfaceUnsignedInt>(InterfaceGridJSON, "start_index_in_mhd_y"))
    {
        
    }
};

#endif
