#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <string>
#include "const.hpp"
#include "flux_solver.hpp"
#include "ct.hpp"
#include "boundary.hpp"


class IdealMHD2D
{
private:
    FluxSolver fluxSolver;
    thrust::device_vector<Flux> fluxF;
    thrust::device_vector<Flux> fluxG;
    thrust::device_vector<ConservationParameter> U;
    thrust::device_vector<ConservationParameter> UBar;
    thrust::device_vector<ConservationParameter> tmpU;
    thrust::device_vector<ConservationParameter> UPast;
    thrust::device_vector<double> dtVector;
    BoundaryMHD boundaryMHD;
    CT ct;
    thrust::device_vector<double> bXOld;
    thrust::device_vector<double> bYOld;
    thrust::device_vector<double> tmpVector;
    thrust::host_vector<ConservationParameter> hU;

public:
    IdealMHD2D();

    virtual void initializeU(); 

    void setPastU();

    void oneStepRK2PeriodicXSymmetricY_predictor();

    void oneStepRK2PeriodicXSymmetricY_corrector(
        thrust::device_vector<ConservationParameter>& UHalf
    );

    void oneStepRK2SymmetricXSymmetricY_predictor();

    void oneStepRK2SymmetricXSymmetricY_corrector(
        thrust::device_vector<ConservationParameter>& UHalf
    );

    void save(
        std::string directoryname, 
        std::string filenameWithoutStep, 
        int step
    );

    thrust::device_vector<ConservationParameter> getU();

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



