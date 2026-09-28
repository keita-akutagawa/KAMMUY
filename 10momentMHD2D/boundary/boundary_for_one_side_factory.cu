#include "boundary_for_one_side_factory.hpp"


std::unique_ptr<BoundaryForOneSide> BoundaryXLeftFactory::create(
    const nlohmann::json& configJSON, 
    const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
    MHDConstParameter& mHDConstParameter, 
    const MHDGridParameter& mHDGridParameter
)
{
    const std::string boundaryXLeftType = getRequiredForJSON<std::string>(configJSON, {"boundary", "x_left"});

    if (boundaryXLeftType == "periodic") {
        return std::make_unique<PeriodicBoundaryXLeft>(NX, NY, mHDConstParameter, mHDGridParameter);
    } else if (boundaryXLeftType == "free") {
        return std::make_unique<FreeBoundaryXLeft>(NX, NY, mHDConstParameter, mHDGridParameter);
    } else if (boundaryXLeftType == "custom") {
        return std::make_unique<CustomBoundaryXLeft>(NX, NY, mHDConstParameter, mHDGridParameter);
    } else {
        throw std::invalid_argument("Unknown boundary condition: " + boundaryXLeftType);
    }
}


std::unique_ptr<BoundaryForOneSide> BoundaryXRightFactory::create(
    const nlohmann::json& configJSON, 
    const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
    MHDConstParameter& mHDConstParameter, 
    const MHDGridParameter& mHDGridParameter
)
{
    const std::string boundaryXRightType = getRequiredForJSON<std::string>(configJSON, {"boundary", "x_right"});

    if (boundaryXRightType == "periodic") {
        return std::make_unique<PeriodicBoundaryXRight>(NX, NY, mHDConstParameter, mHDGridParameter);
    } else if (boundaryXRightType == "free") {
        return std::make_unique<FreeBoundaryXRight>(NX, NY, mHDConstParameter, mHDGridParameter);
    } else if (boundaryXRightType == "custom") {
        return std::make_unique<CustomBoundaryXRight>(NX, NY, mHDConstParameter, mHDGridParameter);
    } else {
        throw std::invalid_argument("Unknown boundary condition: " + boundaryXRightType);
    }
}


std::unique_ptr<BoundaryForOneSide> BoundaryYDownFactory::create(
    const nlohmann::json& configJSON, 
    const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
    MHDConstParameter& mHDConstParameter, 
    const MHDGridParameter& mHDGridParameter
)
{
    const std::string boundaryYDownType = getRequiredForJSON<std::string>(configJSON, {"boundary", "y_down"});


    if (boundaryYDownType == "periodic") {
        return std::make_unique<PeriodicBoundaryYDown>(NX, NY, mHDConstParameter, mHDGridParameter);
    } else if (boundaryYDownType == "free") {
        return std::make_unique<FreeBoundaryYDown>(NX, NY, mHDConstParameter, mHDGridParameter);
    } else if (boundaryYDownType == "custom") {
        return std::make_unique<CustomBoundaryYDown>(NX, NY, mHDConstParameter, mHDGridParameter);
    } else {
        throw std::invalid_argument("Unknown boundary condition: " + boundaryYDownType);
    }
}


std::unique_ptr<BoundaryForOneSide> BoundaryYUpFactory::create(
    const nlohmann::json& configJSON, 
    const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
    MHDConstParameter& mHDConstParameter, 
    const MHDGridParameter& mHDGridParameter
)
{
    const std::string boundaryYUpType = getRequiredForJSON<std::string>(configJSON, {"boundary", "y_up"});

    if (boundaryYUpType == "periodic") {
        return std::make_unique<PeriodicBoundaryYUp>(NX, NY, mHDConstParameter, mHDGridParameter);
    } else if (boundaryYUpType == "free") {
        return std::make_unique<FreeBoundaryYUp>(NX, NY, mHDConstParameter, mHDGridParameter);
    } else if (boundaryYUpType == "custom") {
        return std::make_unique<CustomBoundaryYUp>(NX, NY, mHDConstParameter, mHDGridParameter);
    } else {
        throw std::invalid_argument("Unknown boundary condition: " + boundaryYUpType);
    }
}
