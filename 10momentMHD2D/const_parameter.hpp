#ifndef MHD_CONST_PARAMETER_HPP 
#define MHD_CONST_PARAMETER_HPP

#include "type.hpp"
#include "../utils/json_function.hpp"


struct MHDConstParameter 
{
    const MHDFloat EPS; 
    const MHDFloat PI; 

    const MHDFloat NUG_COEF; 
    
    const bool ACTIVATE_ISOTROPIC_EFFECT; 
    const bool ACTIVATE_GYROTROPIC_EFFECT; 
    const bool ACTIVATE_HALL_EFFECT; 
    
    const MHDFloat CR;
    MHDFloat CH; 
    MHDFloat CP; 

    const MHDFloat CDIFF; 

    const MHDFloat RHO0; 
    const MHDFloat B0; 
    const MHDFloat P0; 

    const MHDFloat M_ION; 
    const MHDFloat M_ELECTRON; 
    const MHDFloat Q_ELECTRON; 

    MHDUnsignedInt CURRENT_STEP; 
    const MHDUnsignedInt RECORD_STEP; 
    const MHDUnsignedInt TOTAL_STEP; 

    MHDFloat DT; 
    MHDFloat TOTAL_TIME; 

    const std::string SAVE_DIRNAME; 
    const std::string SAVE_FILENAME_WITHOUT_STEP; 


    MHDConstParameter(
        const nlohmann::json& MHDConstJSON
    ) 
      : EPS(getRequiredForJSON<MHDFloat>(MHDConstJSON, "EPS")),  
        PI(getRequiredForJSON<MHDFloat>(MHDConstJSON, "PI")),  
        
        NUG_COEF(getRequiredForJSON<MHDFloat>(MHDConstJSON, "nug_coef")), 
        
        ACTIVATE_ISOTROPIC_EFFECT(getRequiredForJSON<bool>(MHDConstJSON, "activate_isotropic_effect")), 
        ACTIVATE_GYROTROPIC_EFFECT(getRequiredForJSON<bool>(MHDConstJSON, "activate_gyrotropic_effect")), 
        ACTIVATE_HALL_EFFECT(getRequiredForJSON<bool>(MHDConstJSON, "activate_hall_effect")), 

        CR(getRequiredForJSON<MHDFloat>(MHDConstJSON, "cr")), 
        CH(0.0), 
        CP(0.0),

        CDIFF(getRequiredForJSON<MHDFloat>(MHDConstJSON, "c_diff")), 

        RHO0(getRequiredForJSON<MHDFloat>(MHDConstJSON, "rho0")), 
        B0(getRequiredForJSON<MHDFloat>(MHDConstJSON, "B0")), 
        P0(getRequiredForJSON<MHDFloat>(MHDConstJSON, "p0")), 

        M_ION(getRequiredForJSON<MHDFloat>(MHDConstJSON, "m_ion")),
        M_ELECTRON(getRequiredForJSON<MHDFloat>(MHDConstJSON, "m_electron")),
        Q_ELECTRON(getRequiredForJSON<MHDFloat>(MHDConstJSON, "q_electron")),
        
        CURRENT_STEP(0), 
        RECORD_STEP(getRequiredForJSON<MHDUnsignedInt>(MHDConstJSON, "record_step")),
        TOTAL_STEP(getRequiredForJSON<MHDUnsignedInt>(MHDConstJSON, "total_step")),

        DT(0.0), 
        TOTAL_TIME(0.0), 

        SAVE_DIRNAME(getRequiredForJSON<std::string>(MHDConstJSON, "save_dirname")), 
        SAVE_FILENAME_WITHOUT_STEP(getRequiredForJSON<std::string>(MHDConstJSON, "save_filename_without_step")
    )
    {
    }
};

#endif 
