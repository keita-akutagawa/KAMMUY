#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <string>
#include "flux_solver.hpp"
#include "ct.hpp"
#include "boundary.hpp"
#include "mpi.hpp"


class IdealMHD2D
{
private:
    IdealMHD2DMPI::MPIInfo mPIInfo; 
    IdealMHD2DMPI::MPIInfo* device_mPIInfo; 

    FluxSolver fluxSolver;
    
    thrust::device_vector<Flux> fluxF;
    thrust::device_vector<Flux> fluxG;
    thrust::device_vector<ConservationParameter> U;
    thrust::device_vector<ConservationParameter> UBar;
    thrust::device_vector<ConservationParameter> UPast;
    thrust::device_vector<double> dtVector;
    thrust::device_vector<double> bXOld;
    thrust::device_vector<double> bYOld;
    thrust::device_vector<double> tmpVector;
    thrust::host_vector<ConservationParameter> hU;

    BoundaryMHD boundaryMHD;
    CT ct;

public:
    IdealMHD2D(IdealMHD2DMPI::MPIInfo& mPIInfo);

    virtual void initializeU(); 

    void setPastU();

    void oneStepRK2();

    void oneStepRK2_periodicXWallY();

    void oneStepRK2_periodicXSymmetricY();

    void save(
        std::string directoryname, 
        std::string filenameWithoutStep, 
        int step
    );

    void calculateDt();

    bool checkCalculationIsCrashed();

    thrust::device_vector<ConservationParameter>& getURef();

    thrust::device_vector<ConservationParameter>& getUPastRef();

private:
    void shiftUToCenterForCT(
        thrust::device_vector<ConservationParameter>& U
    );

    void backUToCenterHalfForCT(
        thrust::device_vector<ConservationParameter>& U
    );
};



