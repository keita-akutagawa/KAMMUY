#include <thrust/device_vector.h>
#include "const.hpp"
#include "conservation_parameter_struct.hpp"


class Boundary
{
private:

public:

    void periodicBoundaryX2nd(
        thrust::device_vector<ConservationParameter>& U
    );

    void freeBoundaryX2nd(
        thrust::device_vector<ConservationParameter>& U
    );

    void periodicBoundaryY2nd(
        thrust::device_vector<ConservationParameter>& U
    );

    void freeBoundaryY2nd(
        thrust::device_vector<ConservationParameter>& U
    );

private:

};


