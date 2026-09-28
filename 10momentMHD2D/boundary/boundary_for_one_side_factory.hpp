#ifndef MHD_BOUNDARY_FOR_ONE_SIDE_FACTORY_HPP
#define MHD_BOUNDARY_FOR_ONE_SIDE_FACTORY_HPP


#include <memory> 

#include "../const_parameter.hpp"
#include "../grid_parameter.hpp"
#include "../../utils/json_function.hpp"
#include "boundary_for_one_side.hpp"

#include "periodic/periodic_boundary.hpp"
#include "free/free_boundary.hpp"
#include "custom/custom_boundary.hpp"


class BoundaryXLeftFactory 
{
private: 

public: 
    static std::unique_ptr<BoundaryForOneSide> create(
        const nlohmann::json& configJSON, 
        const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
        MHDConstParameter& mHDConstParameter, 
        const MHDGridParameter& mHDGridParameter
    );

private:

};


class BoundaryXRightFactory 
{
private: 

public: 
    static std::unique_ptr<BoundaryForOneSide> create(
        const nlohmann::json& configJSON, 
        const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
        MHDConstParameter& mHDConstParameter, 
        const MHDGridParameter& mHDGridParameter
    );

private:

};


class BoundaryYDownFactory 
{
private: 

public: 
    static std::unique_ptr<BoundaryForOneSide> create(
        const nlohmann::json& configJSON, 
        const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
        MHDConstParameter& mHDConstParameter, 
        const MHDGridParameter& mHDGridParameter
    );

private:

};


class BoundaryYUpFactory 
{
private: 

public: 
    static std::unique_ptr<BoundaryForOneSide> create(
        const nlohmann::json& configJSON, 
        const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
        MHDConstParameter& mHDConstParameter, 
        const MHDGridParameter& mHDGridParameter
    );

private:

};

#endif
