#ifndef MHD_GRID_PARAMETER_HPP 
#define MHD_GRID_PARAMETER_HPP

#include "type.hpp"
#include "../utils/json_function.hpp"


struct MHDGridParameter 
{
    const MHDUnsignedInt BUFFER; 

    const std::vector<MHDUnsignedInt> NX; 
    const std::vector<MHDUnsignedInt> NY; 

    const std::vector<MHDFloat> DX; 
    const std::vector<MHDFloat> DY; 

    const std::vector<MHDFloat> XMIN; 
    const std::vector<MHDFloat> YMIN;
    const std::vector<MHDFloat> XMAX; 
    const std::vector<MHDFloat> YMAX; 

    const std::vector<MHDUnsignedInt> START_INDEX_X; 
    const std::vector<MHDUnsignedInt> START_INDEX_Y; 

    const std::vector<MHDUnsignedInt> END_INDEX_X; 
    const std::vector<MHDUnsignedInt> END_INDEX_Y; 

    const bool SMR_AVAIL; 
    const MHDUnsignedInt NUMBER_OF_LEVELS;

    MHDGridParameter(
        const nlohmann::json& MHDGridJSON
    ) 
      : BUFFER(getRequiredForJSON<MHDUnsignedInt>(MHDGridJSON, "buffer")), 
        
        NX(getRequiredForJSON<std::vector<MHDUnsignedInt>>(MHDGridJSON, "nx")), 
        NY(getRequiredForJSON<std::vector<MHDUnsignedInt>>(MHDGridJSON, "ny")), 

        DX(getRequiredForJSON<std::vector<MHDFloat>>(MHDGridJSON, "dx")), 
        DY(getRequiredForJSON<std::vector<MHDFloat>>(MHDGridJSON, "dy")), 

        XMIN(getRequiredForJSON<std::vector<MHDFloat>>(MHDGridJSON, "xmin")), 
        YMIN(getRequiredForJSON<std::vector<MHDFloat>>(MHDGridJSON, "ymin")), 
        XMAX(getRequiredForJSON<std::vector<MHDFloat>>(MHDGridJSON, "xmax")), 
        YMAX(getRequiredForJSON<std::vector<MHDFloat>>(MHDGridJSON, "ymax")), 

        START_INDEX_X(getRequiredForJSON<std::vector<MHDUnsignedInt>>(MHDGridJSON, "start_index_x")), 
        START_INDEX_Y(getRequiredForJSON<std::vector<MHDUnsignedInt>>(MHDGridJSON, "start_index_y")), 
        END_INDEX_X(getRequiredForJSON<std::vector<MHDUnsignedInt>>(MHDGridJSON, "end_index_x")), 
        END_INDEX_Y(getRequiredForJSON<std::vector<MHDUnsignedInt>>(MHDGridJSON, "end_index_y")), 

        SMR_AVAIL(getRequiredForJSON<bool>(MHDGridJSON, "smr_avail")), 
        NUMBER_OF_LEVELS(getRequiredForJSON<MHDUnsignedInt>(MHDGridJSON, "number_of_levels")) 
    {   
        if (static_cast<MHDUnsignedInt>(NX.size()) != NUMBER_OF_LEVELS ||
            static_cast<MHDUnsignedInt>(NY.size()) != NUMBER_OF_LEVELS) {
            throw std::runtime_error("size of grid size array does not match number_of_levels.");
        }

        if (SMR_AVAIL && NUMBER_OF_LEVELS < 1) {
            throw std::runtime_error("number_of_levels should be more than 2 if SMR is used.");
        }
        
        for (MHDUnsignedInt level = 1; level < NUMBER_OF_LEVELS; level++) {
            if (START_INDEX_X[level] + NX[level] / 2 > NX[level - 1]) {
                throw std::invalid_argument("SMR grid in the x direction exceeds simulation box.");
            }
            if (START_INDEX_Y[level] + NY[level] / 2 > NY[level - 1]) {
                throw std::invalid_argument("SMR grid in the y direction exceeds simulation box.");
            }

            if (START_INDEX_X[level] + NX[level] / 2 != END_INDEX_X[level]) {
                throw std::invalid_argument("SMR grid information in the x direction does not match.");
            }
            if (START_INDEX_Y[level] + NY[level] / 2 != END_INDEX_Y[level]) {
                throw std::invalid_argument("SMR grid information in the y direction does not match.");
            }
        }
    }
};

#endif 
