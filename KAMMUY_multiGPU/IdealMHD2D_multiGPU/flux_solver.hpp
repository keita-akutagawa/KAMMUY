#include "const.hpp"
#include "hlld.hpp"
#include "mpi.hpp"


class FluxSolver
{
private:
    IdealMHD2DMPI::MPIInfo mPIInfo;

    HLLD hLLD;
    thrust::device_vector<Flux> flux;

public:
    FluxSolver(IdealMHD2DMPI::MPIInfo& mPIInfo);

    thrust::device_vector<Flux> getFluxF(
        const thrust::device_vector<ConservationParameter>& U
    );

    thrust::device_vector<Flux> getFluxG(
        const thrust::device_vector<ConservationParameter>& U
    );
};


