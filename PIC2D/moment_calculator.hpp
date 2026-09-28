#ifndef PIC_MOMENT_CALCULATOR_HPP
#define PIC_MOMENT_CALCULATOR_HPP

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include "moment_struct.hpp"
#include "particle_struct.hpp"
#include "const_parameter.hpp"
#include "grid_parameter.hpp"
#include "../utils/get_index.hpp"


class MomentCalculator
{
private:
    const PICUnsignedInt NX, NY; 
    const PICFloat DX, DY; 
    PICConstParameter& pICConstParameter; 
    const PICGridParameter& pICGridParameter; 

public:
    MomentCalculator(
        PICConstParameter& pICConstParameter, 
        const PICGridParameter& pICGridParameter
    );

    void calculateZerothMoment(
        const thrust::device_vector<Particle>& particles, 
        const PICUnsignedLongLong existNumSpecies, 
        thrust::device_vector<ZerothMoment>& zerothMoment
    );

    void calculateFirstMoment(
        const thrust::device_vector<Particle>& particles, 
        const PICUnsignedLongLong existNumSpecies, 
        thrust::device_vector<FirstMoment>& firstMoment
    );

    void calculateSecondMoment(
        const thrust::device_vector<Particle>& particles, 
        const PICUnsignedLongLong existNumSpecies, 
        thrust::device_vector<SecondMoment>& secondMoment
    );

private:
    void resetZerothMoment(
        thrust::device_vector<ZerothMoment>& zerothMoment
    );

    void resetFirstMoment(
        thrust::device_vector<FirstMoment>& firstMoment
    );

    void resetSecondMoment(
        thrust::device_vector<SecondMoment>& secondMoment
    );
};

#endif


