#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <string>
#include "flux_solver.hpp"
#include "boundary.hpp"
#include "const.hpp"
#include "projection.hpp"
#include "mpi.hpp"


class IdealMHD2D
{
private:
    IdealMHD2DMPI::MPIInfo& mPIInfo; 
    IdealMHD2DMPI::MPIInfo* device_mPIInfo; 

    FluxSolver fluxSolver;
    
    thrust::device_vector<Flux> fluxF;
    thrust::device_vector<Flux> fluxG;
    thrust::device_vector<ConservationParameter> U;
    thrust::device_vector<ConservationParameter> UBar;
    thrust::device_vector<ConservationParameter> UPast;
    thrust::device_vector<double> dtVector;
    thrust::device_vector<double> tmpVector;
    thrust::host_vector<ConservationParameter> host_U;

    BoundaryMHD boundaryMHD;
    Projection projection; 

public:
    IdealMHD2D(IdealMHD2DMPI::MPIInfo& mPIInfo);

    virtual void initializeU(); 

    void setPastU();

    void oneStepRK2_periodicXY_predictor();

    void oneStepRK2_periodicXY_corrector(
        thrust::device_vector<ConservationParameter>& UHalf
    );

    void oneStepRK2_periodicXSymmetricY_predictor();

    void oneStepRK2_periodicXSymmetricY_corrector(
        thrust::device_vector<ConservationParameter>& UHalf
    );

    void save(
        std::string directoryname, 
        std::string filenameWithoutStep, 
        int step
    );

    void calculateDt();

    void checkAndResetExtremeValues(
        thrust::device_vector<ConservationParameter>& U
    );

    bool checkCalculationIsCrashed();

    thrust::host_vector<ConservationParameter>& getHostURef();

    thrust::device_vector<ConservationParameter>& getURef();

    thrust::device_vector<ConservationParameter>& getUPastRef();

    BoundaryMHD& getBoundaryMHDRef(); 

    Projection& getProjectionRef(); 

private:

};



