#ifndef PIC_CONST_PARAMETER_HPP
#define PIC_CONST_PARAMETER_HPP

#include "type.hpp"
#include "../utils/json_function.hpp"


struct PICConstParameter
{

    const PICFloat C;
    const PICFloat EPSILON0;
    const PICFloat MU0;
    const PICFloat DCOEF_LM;
    const PICFloat EPS;
    const PICFloat PI; 

    PICUnsignedLongLong EXIST_NUM_ION;
    PICUnsignedLongLong EXIST_NUM_ELECTRON;

    const PICUnsignedLongLong TOTAL_NUM_ION;
    const PICUnsignedLongLong TOTAL_NUM_ELECTRON;

    const PICFloat M_ION;
    const PICFloat M_ELECTRON;
    const PICFloat Q_ION;
    const PICFloat Q_ELECTRON;
    const PICUnsignedInt NUMBER_DENSITY_ION;
    const PICUnsignedInt NUMBER_DENSITY_ELECTRON;
    const PICFloat B0;
    const PICFloat P0;

    const PICFloat OMEGA_PE; 

    PICUnsignedInt CURRENT_STEP; 
    const PICUnsignedInt RECORD_STEP; 
    const PICUnsignedInt TOTAL_STEP; 

    const PICFloat DT;
    PICFloat TOTAL_TIME;

    const std::string SAVE_DIRNAME; 
    const std::string SAVE_FILENAME_WITHOUT_STEP; 

    PICConstParameter(
        const nlohmann::json& PICConstJSON
    )
      : C(getRequiredForJSON<PICFloat>(PICConstJSON, "c")),  
        EPSILON0(getRequiredForJSON<PICFloat>(PICConstJSON, "epsilon0")),  
        MU0(getRequiredForJSON<PICFloat>(PICConstJSON, "mu0")),  
        DCOEF_LM(getRequiredForJSON<PICFloat>(PICConstJSON, "dcoef_lm")),  
        EPS(getRequiredForJSON<PICFloat>(PICConstJSON, "EPS")),  
        PI(getRequiredForJSON<PICFloat>(PICConstJSON, "PI")),  

        EXIST_NUM_ION(getRequiredForJSON<PICUnsignedLongLong>(PICConstJSON, "exist_num_ion")),  
        EXIST_NUM_ELECTRON(getRequiredForJSON<PICUnsignedLongLong>(PICConstJSON, "exist_num_electron")),  

        TOTAL_NUM_ION(getRequiredForJSON<PICUnsignedLongLong>(PICConstJSON, "total_num_ion")),  
        TOTAL_NUM_ELECTRON(getRequiredForJSON<PICUnsignedLongLong>(PICConstJSON, "total_num_electron")),  

        M_ION(getRequiredForJSON<PICFloat>(PICConstJSON, "m_ion")),  
        M_ELECTRON(getRequiredForJSON<PICFloat>(PICConstJSON, "m_electron")),  
        Q_ION(getRequiredForJSON<PICFloat>(PICConstJSON, "q_ion")),  
        Q_ELECTRON(getRequiredForJSON<PICFloat>(PICConstJSON, "q_electron")),  
        NUMBER_DENSITY_ION(getRequiredForJSON<PICUnsignedInt>(PICConstJSON, "number_density_ion")),  
        NUMBER_DENSITY_ELECTRON(getRequiredForJSON<PICUnsignedInt>(PICConstJSON, "number_density_electron")),  
        B0(getRequiredForJSON<PICFloat>(PICConstJSON, "B0")),  
        P0(getRequiredForJSON<PICFloat>(PICConstJSON, "p0")),  
        
        OMEGA_PE(getRequiredForJSON<PICFloat>(PICConstJSON, "omega_pe")),  
        
        CURRENT_STEP(0), 
        RECORD_STEP(getRequiredForJSON<PICUnsignedInt>(PICConstJSON, "record_step")),
        TOTAL_STEP(getRequiredForJSON<PICUnsignedInt>(PICConstJSON, "total_step")),

        DT(getRequiredForJSON<PICFloat>(PICConstJSON, "dt")), 
        TOTAL_TIME(0.0), 

        SAVE_DIRNAME(getRequiredForJSON<std::string>(PICConstJSON, "save_dirname")), 
        SAVE_FILENAME_WITHOUT_STEP(getRequiredForJSON<std::string>(PICConstJSON, "save_filename_without_step")
    )
    {

    }
};

#endif
