#ifndef MHD_STATIC_MESH_REFINEMENT_HPP 
#define MHD_STATIC_MESH_REFINEMENT_HPP

#include <thrust/device_vector.h> 

#include "../../type.hpp"
#include "../../const_parameter.hpp"
#include "../../mhd_value.hpp"
#include "../../global_function.hpp"
#include "../../grid_parameter.hpp"
#include "../../push/time_integrator_factory.hpp"
#include "../../io/output/output_factory.hpp"
#include "../../../utils/get_index.hpp"


class SMR 
{
private: 

    MHDConstParameter& mHDConstParameter;
    const MHDGridParameter& mHDGridParameter; 

    const MHDInt level; 
    const MHDUnsignedInt COARSE_NX, COARSE_NY; 
    const MHDUnsignedInt SMR_NX, SMR_NY; 
    const MHDFloat SMR_DX, SMR_DY; 
    const MHDUnsignedInt START_INDEX_X, START_INDEX_Y; 
    const MHDUnsignedInt END_INDEX_X, END_INDEX_Y;

public: 
    SMR( 
        nlohmann::json& configJSON, 
        const MHDInt level, 
        MHDConstParameter& mHDConstParameter, 
        const MHDGridParameter& mHDGridParameter
    );

    void pushOneLayer(
        const thrust::device_vector<MHDValue>& UPast, 
        const thrust::device_vector<MHDValue>& UNext, 
        const MHDFloat DT, 
        const MHDInt substep, 
        std::unique_ptr<TimeIntegrator>& timeIntegrator
    );

    void synchronizeOneLayer(
        const thrust::device_vector<MHDValue>& U, 
        thrust::device_vector<MHDValue>& coarseUNext
    );

private:

};

#endif 
