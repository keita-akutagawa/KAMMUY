#ifndef MHD_SSPRK3_HPP 
#define MHD_SSPRK3_HPP 

#include "../time_integrator.hpp"
#include "../filtering_flux_value.hpp"
#include "../rhs_value.hpp"
#include "../heating_value.hpp"
#include "../../type.hpp"
#include "../../const_parameter.hpp"
#include "../../grid_parameter.hpp"
#include "../../mhd_value.hpp"
#include "../../global_function.hpp"
#include "../../reconstructor/reconstructor_factory.hpp"
#include "../../source/source_term_calculator.hpp"
#include "../../boundary/boundary.hpp"
#include "../../mesh/smr/smr_boundary/smr_boundary.hpp"
#include "../../../utils/get_index.hpp"


class SSPRK3 : public TimeIntegrator 
{
private:
    const MHDUnsignedInt NX, NY; 
    const MHDFloat DX, DY;    
    const MHDUnsignedInt level; 
    MHDConstParameter& mHDConstParameter;
    const MHDGridParameter& mHDGridParameter; 

    const MHDInt order; 

    thrust::device_vector<MHDFloat> dtVector; 

    thrust::device_vector<MHDValue> U;
    thrust::device_vector<MHDValue> U1;
    thrust::device_vector<MHDValue> U2; 
    thrust::device_vector<MHDValue> UPast;

    thrust::device_vector<FilteringFluxValue> filteringFluxX; 
    thrust::device_vector<FilteringFluxValue> filteringFluxY; 
    
    std::unique_ptr<Reconstructor> reconstructor;
    SourceTermCalculator sourceTermCalculator;
    std::unique_ptr<Boundary> boundary; 
    std::unique_ptr<SMRBoundary> smrBoundary; 

public:
    SSPRK3(
        const nlohmann::json& configJSON, 
        const MHDUnsignedInt level, 
        MHDConstParameter& mHDConstParameter, 
        const MHDGridParameter& mHDGridParameter
    ); 

    void setUPast() override;

    void push(
        const MHDFloat DT
    ) override;

    void push(
        const thrust::device_vector<MHDValue>& coarseUPast, 
        const thrust::device_vector<MHDValue>& coarseUNext, 
        const MHDFloat DT, 
        const MHDInt substep
    ) override;


    thrust::device_vector<MHDValue>& getUPastRef() override; 
    thrust::device_vector<MHDValue>& getURef() override; 

    Boundary& getBoundaryRef() override; 
    SMRBoundary& getSMRBoundaryRef() override; 

private: 
    void calculateFilteringFlux(
        const thrust::device_vector<MHDValue>& U, 
        const MHDFloat maxSpeed 
    ); 

    void calculateFilteringFluxForOneDirection(
        const thrust::device_vector<MHDValue>& U, 
        const MHDFloat maxSpeed, const MHDUnsignedInt shift, 
        thrust::device_vector<FilteringFluxValue>& filteringFlux
    ); 
};  

#endif
