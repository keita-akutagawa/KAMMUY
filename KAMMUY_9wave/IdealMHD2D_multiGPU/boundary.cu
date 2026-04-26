#include "boundary.hpp"


void BoundaryMHD::boundaryU(
    thrust::device_vector<ConservationParameter>& U
)
{
    boundaryUXLeft(U);
    boundaryUXRight(U);
    boundaryUYDown(U);
    boundaryUYUp(U);
}

