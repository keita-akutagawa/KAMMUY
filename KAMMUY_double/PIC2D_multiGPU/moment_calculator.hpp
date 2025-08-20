#ifndef MOMENT_CALCULATOR_H
#define MOMENT_CALCULATOR_H

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include "moment_struct.hpp"
#include "particle_struct.hpp"
#include "const.hpp"
#include "mpi.hpp"


class MomentCalculator
{
private:
    PIC2DMPI::MPIInfo& mPIInfo;

public:
    MomentCalculator(PIC2DMPI::MPIInfo& mPIInfo);

    void calculateZerothMomentOfOneSpecies(
        thrust::device_vector<ZerothMoment>& zerothMomentOfOneSpecies, 
        const thrust::device_vector<Particle>& particlesSpecies, 
        const unsigned long long existNumSpecies
    );

    void calculateFirstMomentOfOneSpecies(
        thrust::device_vector<FirstMoment>& firstMomentOfOneSpecies, 
        const thrust::device_vector<Particle>& particlesSpecies, 
        const unsigned long long existNumSpecies
    );

    void calculateSecondMomentOfOneSpecies(
        thrust::device_vector<SecondMoment>& secondMomentOfOneSpecies, 
        const thrust::device_vector<Particle>& particlesSpecies, 
        const unsigned long long existNumSpecies
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


