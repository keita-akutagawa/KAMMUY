#include <thrust/device_vector.h>
#include "const.hpp"
#include "particle_struct.hpp"
#include "field_parameter_struct.hpp"
#include "moment_calculator.hpp"


class CurrentCalculator
{
private: 
    
    MomentCalculator momentCalculator; 

public: 
    CurrentCalculator();

    void calculateCurrent(
        thrust::device_vector<CurrentField>& current, 
        thrust::device_vector<FirstMoment>& firstMomentIon, 
        thrust::device_vector<FirstMoment>& firstMomentElectron, 
        const thrust::device_vector<Particle>& particlesIon, 
        const thrust::device_vector<Particle>& particlesEleectron
    );

private:

};

