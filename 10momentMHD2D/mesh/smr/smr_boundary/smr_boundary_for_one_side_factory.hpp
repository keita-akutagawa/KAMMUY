#ifndef MHD_SMR_BOUNDARY_FOR_ONE_SIDE_FACTORY_HPP
#define MHD_SMR_BOUNDARY_FOR_ONE_SIDE_FACTORY_HPP


#include <memory> 

#include "../../../const_parameter.hpp"
#include "../../../grid_parameter.hpp"
#include "../../../../utils/json_function.hpp"
#include "interpolate/interpolate_boundary.hpp"
#include "custom/custom_boundary.hpp"


class SMRBoundaryXLeftFactory 
{
private: 

public: 
    static std::unique_ptr<SMRBoundaryForOneSide> create(
        const nlohmann::json& configJSON, 
        const MHDInt level,  
        MHDConstParameter& mHDConstParameter, 
        const MHDGridParameter& mHDGridParameter
    );

private:

};


class SMRBoundaryXRightFactory 
{
private: 

public: 
    static std::unique_ptr<SMRBoundaryForOneSide> create(
        const nlohmann::json& configJSON, 
        const MHDInt level,  
        MHDConstParameter& mHDConstParameter, 
        const MHDGridParameter& mHDGridParameter
    );

private:

};


class SMRBoundaryYDownFactory 
{
private: 

public: 
    static std::unique_ptr<SMRBoundaryForOneSide> create(
        const nlohmann::json& configJSON, 
        const MHDInt level,  
        MHDConstParameter& mHDConstParameter, 
        const MHDGridParameter& mHDGridParameter
    );

private:

};


class SMRBoundaryYUpFactory 
{
private: 

public: 
    static std::unique_ptr<SMRBoundaryForOneSide> create(
        const nlohmann::json& configJSON, 
        const MHDInt level,  
        MHDConstParameter& mHDConstParameter, 
        const MHDGridParameter& mHDGridParameter
    );

private:

};

#endif
