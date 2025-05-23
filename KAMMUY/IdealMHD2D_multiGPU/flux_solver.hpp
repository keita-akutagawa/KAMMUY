#include "const.hpp"
#include "hlld.hpp"
#include "basic_parameter_struct.hpp"
#include "mpi.hpp"


class FluxSolver
{
private:
    IdealMHD2DMPI::MPIInfo& mPIInfo;

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

    void addResistiveTermToFluxF(
        const thrust::device_vector<ConservationParameter>& U
    );

    void addResistiveTermToFluxG(
        const thrust::device_vector<ConservationParameter>& U
    );
};


