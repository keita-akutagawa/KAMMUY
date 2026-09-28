#ifndef PIC_CURRENT_CALCULATOR_HPP 
#define PIC_CURRENT_CALCULATOR_HPP

#include <thrust/device_vector.h>
#include "particle_struct.hpp"
#include "field_parameter_struct.hpp"
#include "moment_calculator.hpp"
#include "type.hpp"
#include "const_parameter.hpp"
#include "grid_parameter.hpp"
#include "../utils/get_index.hpp"


class CurrentCalculator
{
private: 
    const PICUnsignedInt NX, NY; 
    const PICFloat DX, DY; 
    PICConstParameter& pICConstParameter; 
    const PICGridParameter& pICGridParameter; 
    
    MomentCalculator momentCalculator; 

public: 
    CurrentCalculator(
        PICConstParameter& pICConstParameter, 
        const PICGridParameter& pICGridParameter
    );

    void calculateCurrent(
        const thrust::device_vector<Particle>& particlesIon, 
        const thrust::device_vector<Particle>& particlesElectron, 
        thrust::device_vector<FirstMoment>& firstMomentIon, 
        thrust::device_vector<FirstMoment>& firstMomentElectron, 
        thrust::device_vector<CurrentField>& current
    );

private:

};

#endif 

