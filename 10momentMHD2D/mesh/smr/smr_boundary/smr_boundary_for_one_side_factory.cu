#include "smr_boundary_for_one_side_factory.hpp"


std::unique_ptr<SMRBoundaryForOneSide> SMRBoundaryXLeftFactory::create(
    const nlohmann::json& configJSON, 
    const MHDInt level, 
    MHDConstParameter& mHDConstParameter, 
    const MHDGridParameter& mHDGridParameter
)
{
    const std::vector<std::string> SMRBoundaryXLeftType = getRequiredForJSON<std::vector<std::string>>(configJSON, {"smr_boundary", "x_left"});

    if (SMRBoundaryXLeftType[level] == "interpolate") {
        return std::make_unique<SMRInterpolateBoundaryXLeft>(level, mHDConstParameter, mHDGridParameter);
    } else if (SMRBoundaryXLeftType[level] == "custom") {
        return std::make_unique<SMRCustomBoundaryXLeft>(level, mHDConstParameter, mHDGridParameter);
    } else {
        throw std::invalid_argument("Unknown smr boundary condition: " + SMRBoundaryXLeftType[level]);
    }
}


std::unique_ptr<SMRBoundaryForOneSide> SMRBoundaryXRightFactory::create(
    const nlohmann::json& configJSON, 
    const MHDInt level, 
    MHDConstParameter& mHDConstParameter, 
    const MHDGridParameter& mHDGridParameter
)
{
    const std::vector<std::string> SMRBoundaryXRightType = getRequiredForJSON<std::vector<std::string>>(configJSON, {"smr_boundary", "x_right"});

    if (SMRBoundaryXRightType[level] == "interpolate") {
        return std::make_unique<SMRInterpolateBoundaryXRight>(level, mHDConstParameter, mHDGridParameter);
    } else if (SMRBoundaryXRightType[level] == "custom") {
        return std::make_unique<SMRCustomBoundaryXRight>(level, mHDConstParameter, mHDGridParameter);
    } else {
        throw std::invalid_argument("Unknown smr boundary condition: " + SMRBoundaryXRightType[level]);
    }
}


std::unique_ptr<SMRBoundaryForOneSide> SMRBoundaryYDownFactory::create(
    const nlohmann::json& configJSON, 
    const MHDInt level, 
    MHDConstParameter& mHDConstParameter, 
    const MHDGridParameter& mHDGridParameter
)
{
    const std::vector<std::string> SMRBoundaryYDownType = getRequiredForJSON<std::vector<std::string>>(configJSON, {"smr_boundary", "y_down"});

    if (SMRBoundaryYDownType[level] == "interpolate") {
        return std::make_unique<SMRInterpolateBoundaryYDown>(level, mHDConstParameter, mHDGridParameter);
    } else if (SMRBoundaryYDownType[level] == "custom") {
        return std::make_unique<SMRCustomBoundaryYDown>(level, mHDConstParameter, mHDGridParameter);
    } else {
        throw std::invalid_argument("Unknown smr boundary condition: " + SMRBoundaryYDownType[level]);
    }
}


std::unique_ptr<SMRBoundaryForOneSide> SMRBoundaryYUpFactory::create(
    const nlohmann::json& configJSON, 
    const MHDInt level, 
    MHDConstParameter& mHDConstParameter, 
    const MHDGridParameter& mHDGridParameter
)
{
    const std::vector<std::string> SMRBoundaryYUpType = getRequiredForJSON<std::vector<std::string>>(configJSON, {"smr_boundary", "y_up"});

    if (SMRBoundaryYUpType[level] == "interpolate") {
        return std::make_unique<SMRInterpolateBoundaryYUp>(level, mHDConstParameter, mHDGridParameter);
    } else if (SMRBoundaryYUpType[level] == "custom") {
        return std::make_unique<SMRCustomBoundaryYUp>(level, mHDConstParameter, mHDGridParameter);
    } else {
        throw std::invalid_argument("Unknown smr boundary condition: " + SMRBoundaryYUpType[level]);
    }
}
