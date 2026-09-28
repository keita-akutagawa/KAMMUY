#include "time_integrator_factory.hpp"


std::unique_ptr<TimeIntegrator> TimeIntegratorFactory::create(
    const nlohmann::json& configJSON, 
    const MHDInt level, 
    MHDConstParameter& mHDConstParameter, 
    const MHDGridParameter& mHDGridParameter
)
{
    const std::string pusherType = getRequiredForJSON<std::string>(configJSON, "pusher");
    
    if (pusherType == "ssprk3") {
        return std::make_unique<SSPRK3>(configJSON, level, mHDConstParameter, mHDGridParameter);
    } else {
        throw std::invalid_argument("Unknown pusher: " + pusherType);
    }
}

