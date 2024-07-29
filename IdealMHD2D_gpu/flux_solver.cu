#include "flux_solver.hpp"


using namespace IdealMHD2DConst;

thrust::device_vector<Flux> FluxSolver::getFluxF(
    const thrust::device_vector<ConservationParameter>& U
)
{
    hLLD.calculateFluxF(U);
    flux = hLLD.getFlux();

    return flux;
}


thrust::device_vector<Flux> FluxSolver::getFluxG(
    const thrust::device_vector<ConservationParameter>& U
)
{
    hLLD.calculateFluxG(U);
    flux = hLLD.getFlux();

    return flux;
}


