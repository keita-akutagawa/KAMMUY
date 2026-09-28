#include "output_factory.hpp"

#include "binary/binary_output.hpp"


std::unique_ptr<Output> OutputFactory::create(
    const nlohmann::json& configJSON, 
    const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
    const MHDConstParameter& mHDConstParameter
)
{
    const std::string outputType = getRequiredForJSON<std::string>(configJSON, "output");

    if (outputType == "binary") {
        return std::make_unique<BinaryOutput>(NX, NY, mHDConstParameter);
    } else {
        throw std::invalid_argument("Unknown output format: " + outputType);
    }
}

