#ifndef MHD_RECONSTRUCTOR_FACTORY_HPP
#define MHD_RECONSTRUCTOR_FACTORY_HPP


#include <memory> 

#include "reconstructor.hpp"
#include "../const_parameter.hpp"
#include "../../utils/json_function.hpp"
#include "muscl/muscl.hpp"
#include "weno5/weno5.hpp"


class ReconstructorFactory 
{
private: 

public: 
    static std::unique_ptr<Reconstructor> create(
        const nlohmann::json& config, 
        const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
        MHDConstParameter& mHDConstParameter
    );

private:

};

#endif
