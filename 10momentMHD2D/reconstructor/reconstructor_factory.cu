#include "reconstructor_factory.hpp"


std::unique_ptr<Reconstructor> ReconstructorFactory::create(
    const nlohmann::json& configJSON, 
    const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
    MHDConstParameter& mHDConstParameter
)
{
    const std::string reconstructorType = getRequiredForJSON<std::string>(configJSON, "reconstructor");

    if (reconstructorType == "muscl") {
        return std::make_unique<MUSCL>(NX, NY, mHDConstParameter);
    } else if (reconstructorType == "weno5") {
        return std::make_unique<WENO5>(NX, NY, mHDConstParameter);
    } else {
        throw std::invalid_argument("Unknown reconstructor: " + reconstructorType);
    }
}

