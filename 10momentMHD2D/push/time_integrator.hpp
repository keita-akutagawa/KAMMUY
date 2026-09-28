#ifndef MHD_TIME_INTEGRATOR_HPP
#define MHD_TIME_INTEGRATOR_HPP

#include <thrust/device_vector.h>
#include "../mhd_value.hpp"
#include "../boundary/boundary.hpp"
#include "../mesh/smr/smr_boundary/smr_boundary.hpp"


class TimeIntegrator 
{
private: 

public: 
    
    virtual void setUPast() = 0;

    virtual void push(
        const MHDFloat DT
    ) = 0;

    virtual void push(
        const thrust::device_vector<MHDValue>& coarseUPast, 
        const thrust::device_vector<MHDValue>& coarseUNext, 
        const MHDFloat DT, 
        const MHDInt substep
    ) = 0;

    

    virtual thrust::device_vector<MHDValue>& getUPastRef() = 0; 
    virtual thrust::device_vector<MHDValue>& getURef() = 0;

    virtual Boundary& getBoundaryRef() = 0;
    virtual SMRBoundary& getSMRBoundaryRef() = 0;

    virtual ~TimeIntegrator() {}

private: 
};

#endif
