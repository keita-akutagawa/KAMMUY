#include "const.hpp"
#include "hlld.hpp"
#include "get_eta.hpp"
#include "basic_parameter_struct.hpp"


class FluxSolver
{
private:
    HLLD hLLD;
    thrust::device_vector<Flux> flux;

public:
    FluxSolver();

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


