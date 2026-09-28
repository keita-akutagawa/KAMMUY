#ifndef MHD_TIME_INTEGRATOR_FACTORY_HPP
#define MHD_TIME_INTEGRATOR_FACTORY_HPP


#include <memory> 

#include "time_integrator.hpp"
#include "../const_parameter.hpp"
#include "../grid_parameter.hpp"
#include "../../utils/json_function.hpp"

#include "ssprk3/ssprk3.hpp"


class TimeIntegratorFactory 
{
private: 

public: 
    static std::unique_ptr<TimeIntegrator> create(
        const nlohmann::json& config, 
        const MHDInt level, 
        MHDConstParameter& mHDConstParameter, 
        const MHDGridParameter& mHDGridParameter
    );

private:

};

#endif
