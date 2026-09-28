#ifndef OROCHI2D_HPP
#define OROCHI2D_HPP

#include <thrust/device_vector.h> 

#include "type.hpp"
#include "const_parameter.hpp"
#include "grid_parameter.hpp"
#include "mhd_value.hpp"
#include "global_function.hpp"
#include "push/time_integrator.hpp"
#include "push/time_integrator_factory.hpp"
#include "io/output/output.hpp"
#include "io/output/output_factory.hpp"
#include "mesh/smr/smr.hpp"
#include "../utils/get_index.hpp"

#include "../PIC2D/const_parameter.hpp"
#include "../PIC2D/grid_parameter.hpp"


class OROCHI2D 
{
private: 
    MHDConstParameter mHDConstParameter;
    MHDGridParameter mHDGridParameter;

    std::vector<std::unique_ptr<TimeIntegrator>> timeIntegrators; 
    std::vector<std::unique_ptr<Output>> outputs; 

    std::vector<std::unique_ptr<SMR>> smrs;

public: 
    OROCHI2D(
        nlohmann::json& configJSON, 
        nlohmann::json& constJSON, 
        nlohmann::json& gridJSON
    ); 

    virtual void initializeMHDValue();

    void oneStep(
        const MHDUnsignedInt step
    );

    void smrStep(
        std::vector<std::unique_ptr<TimeIntegrator>>& timeIntegrators, 
        std::vector<std::unique_ptr<SMR>>& smrs, 
        MHDUnsignedInt level, 
        const MHDFloat DT
    );

    void save();

    bool isCrashed();

    MHDGridParameter& getMHDGridParameterRef();
    MHDConstParameter& getMHDConstParameterRef();

    std::vector<std::unique_ptr<TimeIntegrator>>& getTimeIntegratorsRef();

private: 

};

#endif 
