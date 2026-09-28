#ifndef MHD_OUTPUT_FACTORY_HPP
#define MHD_OUTPUT_FACTORY_HPP


#include <memory> 

#include "output.hpp"
#include "../../const_parameter.hpp"
#include "../../../utils/json_function.hpp"


class OutputFactory 
{
private: 

public: 
    static std::unique_ptr<Output> create(
        const nlohmann::json& configJSON, 
        const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
        const MHDConstParameter& mHDConstParameter
    );

private:

};

#endif
