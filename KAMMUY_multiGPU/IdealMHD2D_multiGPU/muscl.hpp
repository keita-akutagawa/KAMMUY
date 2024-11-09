#include <thrust/device_vector.h>
#include "const.hpp"
#include "minmod.hpp"
#include "basic_parameter_struct.hpp"
#include "mpi.hpp"


class MUSCL
{
private:
    IdealMHD2DMPI::MPIInfo mPIInfo;

public:
    MUSCL(IdealMHD2DMPI::MPIInfo& mPIInfo);

    void getLeftQX(
        const thrust::device_vector<BasicParameter>& dQ, 
        thrust::device_vector<BasicParameter>& dQLeft
    );

    void getLeftQY(
        const thrust::device_vector<BasicParameter>& dQ, 
        thrust::device_vector<BasicParameter>& dQLeft
    );

    void getRightQX(
        const thrust::device_vector<BasicParameter>& dQ, 
        thrust::device_vector<BasicParameter>& dQRight
    );

    void getRightQY(
        const thrust::device_vector<BasicParameter>& dQ, 
        thrust::device_vector<BasicParameter>& dQRight
    );
};

