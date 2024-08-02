#include <thrust/device_vector.h>
#include "moment_struct.hpp"
#include "particle_struct.hpp"
#include "const.hpp"


#ifndef MOMENT_CALCULATER_STRUCT_H
#define MOMENT_CALCULATER_STRUCT_H

class MomentCalculater
{
private:

public:

    void calculateZerothMomentOfOneSpecies(
        thrust::device_vector<ZerothMoment>& zerothMomentOfOneSpecies, 
        const thrust::device_vector<Particle>& particlesSpecies, 
        unsigned long long existNumSpecies
    );

    void calculateFirstMomentOfOneSpecies(
        thrust::device_vector<FirstMoment>& firstMomentOfOneSpecies, 
        const thrust::device_vector<Particle>& particlesSpecies, 
        unsigned long long existNumSpecies
    );

    void calculateSecondMomentOfOneSpecies(
        thrust::device_vector<SecondMoment>& secondMomentOfOneSpecies, 
        const thrust::device_vector<Particle>& particlesSpecies, 
        unsigned long long existNumSpecies
    );

private:
    void resetZerothMomentOfOneSpecies(
        thrust::device_vector<ZerothMoment>& zerothMomentOfOneSpecies
    );

    void resetFirstMomentOfOneSpecies(
        thrust::device_vector<FirstMoment>& firstMomentOfOneSpecies
    );

    void resetSecondMomentOfOneSpecies(
        thrust::device_vector<SecondMoment>& secondMomentOfOneSpecies
    );
};

#endif

