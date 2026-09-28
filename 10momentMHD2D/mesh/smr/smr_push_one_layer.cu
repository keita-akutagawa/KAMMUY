#include "smr.hpp"


void SMR::pushOneLayer(
    const thrust::device_vector<MHDValue>& coarseUPast, 
    const thrust::device_vector<MHDValue>& coarseUNext, 
    const MHDFloat DT, 
    const MHDInt substep, 
    std::unique_ptr<TimeIntegrator>& timeIntegrator
)
{
    timeIntegrator->setUPast();

    timeIntegrator->push(coarseUPast, coarseUNext, DT, substep);
}

