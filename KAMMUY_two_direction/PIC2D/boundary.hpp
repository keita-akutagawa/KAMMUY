#ifndef BOUNDARY_PIC_H
#define BOUNDARY_PIC_H

#include <thrust/device_vector.h>
#include <thrust/partition.h>
#include <thrust/transform_reduce.h>
#include "const.hpp"
#include "particle_struct.hpp"
#include "field_parameter_struct.hpp"
#include "moment_struct.hpp"


class BoundaryPIC
{
private:

    thrust::device_vector<Particle> bufferParticlesSpeciesX; 
    thrust::device_vector<Particle> bufferParticlesSpeciesY; 

public:
    BoundaryPIC();

    void freeBoundaryParticle_x(
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron
    );
    void freeBoundaryParticleOfOneSpecies_x(
        thrust::device_vector<Particle>& particlesSpecies, 
        unsigned long long& existNumSpecies
    );
    void freeBoundaryParticle_y(
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron
    );
    void freeBoundaryParticleOfOneSpecies_y(
        thrust::device_vector<Particle>& particlesSpecies, 
        unsigned long long& existNumSpecies
    );

    void freeBoundaryB_x(
        thrust::device_vector<MagneticField>& B
    );
    void freeBoundaryB_y(
        thrust::device_vector<MagneticField>& B
    );

    void freeBoundaryE_x(
        thrust::device_vector<ElectricField>& E
    );
    void freeBoundaryE_y(
        thrust::device_vector<ElectricField>& E
    );

    void freeBoundaryCurrent_x(
        thrust::device_vector<CurrentField>& current
    );
    void freeBoundaryCurrent_y(
        thrust::device_vector<CurrentField>& current
    );

    void freeBoundaryZerothMoment_x(
        thrust::device_vector<ZerothMoment>& zerothMoment
    );
    void freeBoundaryZerothMoment_y(
        thrust::device_vector<ZerothMoment>& zerothMoment
    );

    void freeBoundaryFirstMoment_x(
        thrust::device_vector<FirstMoment>& firstMoment
    );
    void freeBoundaryFirstMoment_y(
        thrust::device_vector<FirstMoment>& firstMoment
    );

    void freeBoundarySecondMoment_x(
        thrust::device_vector<SecondMoment>& secondMoment
    );
    void freeBoundarySecondMoment_y(
        thrust::device_vector<SecondMoment>& secondMoment
    );

private:

};

#endif


