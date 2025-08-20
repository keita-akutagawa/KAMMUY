#include <thrust/device_vector.h>
#include "const.hpp"
#include "field_parameter_struct.hpp"
#include "particle_struct.hpp"
#include "moment_calculator.hpp"
#include "mpi.hpp"


class Filter
{
private:
    PIC2DMPI::MPIInfo& mPIInfo;

    thrust::device_vector<RhoField> rho;
    thrust::device_vector<FilterField> F;

    MomentCalculator momentCalculator; 

public:
    Filter(PIC2DMPI::MPIInfo& mPIInfo);

    void calculateRho(
        thrust::device_vector<ZerothMoment>& zerothMomentIon, 
        thrust::device_vector<ZerothMoment>& zerothMomentElectron, 
        const thrust::device_vector<Particle>& particlesIon, 
        const thrust::device_vector<Particle>& particlesElectron
    );

    void langdonMarderTypeCorrection(
        thrust::device_vector<ElectricField>& E, 
        const double dt
    );

private:

};




