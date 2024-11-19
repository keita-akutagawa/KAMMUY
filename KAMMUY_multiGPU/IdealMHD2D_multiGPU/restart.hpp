#include <thrust/device_vector.h>
#include <string>
#include <fstream>
#include "const.hpp"
#include "conservation_parameter_struct.hpp"
#include "mpi.hpp"


class RestartMHD 
{
private:
    IdealMHD2DMPI::MPIInfo& mPIInfo; 

public:
    RestartMHD(
        IdealMHD2DMPI::MPIInfo& mPIInfo
    ); 

    void loadU(
        thrust::host_vector<ConservationParameter>& hU, 
        thrust::device_vector<ConservationParameter>& U, 
        std::string directoryname, 
        std::string filenameWithoutStep, 
        int step
    ); 

private:

};


