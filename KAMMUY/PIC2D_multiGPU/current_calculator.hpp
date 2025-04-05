#include <thrust/device_vector.h>
#include "const.hpp"
#include "particle_struct.hpp"
#include "field_parameter_struct.hpp"
#include "moment_calculator.hpp"
#include "mpi.hpp"


class CurrentCalculator
{
private: 
    PIC2DMPI::MPIInfo& mPIInfo; 
    
    MomentCalculator momentCalculator; 

public: 
    CurrentCalculator(PIC2DMPI::MPIInfo& mPIInfo);

    //void resetCurrent(
    //    thrust::device_vector<CurrentField>& current
    //);

    void calculateCurrent(
        thrust::device_vector<CurrentField>& current, 
        thrust::device_vector<FirstMoment>& firstMomentIon, 
        thrust::device_vector<FirstMoment>& firstMomentElectron, 
        const thrust::device_vector<Particle>& particlesIon, 
        const thrust::device_vector<Particle>& particlesEleectron
    );

private:

};

