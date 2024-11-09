#include "boundary.hpp"



//////////

void BoundaryPIC::periodicBoundaryB_x(
    thrust::device_vector<MagneticField>& B
)
{
    MPI_Barrier(MPI_COMM_WORLD);
}


void BoundaryPIC::periodicBoundaryB_y(
    thrust::device_vector<MagneticField>& B
)
{
    MPI_Barrier(MPI_COMM_WORLD);
}

//////////

void BoundaryPIC::periodicBoundaryE_x(
    thrust::device_vector<ElectricField>& E
)
{
    MPI_Barrier(MPI_COMM_WORLD);
}


void BoundaryPIC::periodicBoundaryE_y(
    thrust::device_vector<ElectricField>& E
)
{
    MPI_Barrier(MPI_COMM_WORLD);
}

//////////

void BoundaryPIC::periodicBoundaryCurrent_x(
    thrust::device_vector<CurrentField>& current
)
{
    MPI_Barrier(MPI_COMM_WORLD);
}


void BoundaryPIC::periodicBoundaryCurrent_y(
    thrust::device_vector<CurrentField>& current
)
{
    MPI_Barrier(MPI_COMM_WORLD);
}



