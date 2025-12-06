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

    thrust::device_vector<Particle> sendParticlesSpeciesYDown; 
    thrust::device_vector<Particle> sendParticlesSpeciesYUp; 
    thrust::device_vector<Particle> recvParticlesSpeciesYDown; 
    thrust::device_vector<Particle> recvParticlesSpeciesYUp;

public:
    BoundaryPIC();

    void periodicBoundaryParticle_x(
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron
    );
    void periodicBoundaryParticleOfOneSpecies_x(
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


    void periodicBoundaryB_x(
        thrust::device_vector<MagneticField>& B
    );

    void freeBoundaryB_x(
        thrust::device_vector<MagneticField>& B
    );
    void freeBoundaryB_y(
        thrust::device_vector<MagneticField>& B
    );


    void periodicBoundaryE_x(
        thrust::device_vector<ElectricField>& E
    );
    
    void freeBoundaryE_x(
        thrust::device_vector<ElectricField>& E
    );
    void freeBoundaryE_y(
        thrust::device_vector<ElectricField>& E
    );


    void periodicBoundaryCurrent_x(
        thrust::device_vector<CurrentField>& current
    );

    void freeBoundaryCurrent_x(
        thrust::device_vector<CurrentField>& current
    );
    void freeBoundaryCurrent_y(
        thrust::device_vector<CurrentField>& current
    );


    void periodicBoundaryZerothMoment_x(
        thrust::device_vector<ZerothMoment>& zerothMoment
    );

    void freeBoundaryZerothMoment_x(
        thrust::device_vector<ZerothMoment>& zerothMoment
    );
    void freeBoundaryZerothMoment_y(
        thrust::device_vector<ZerothMoment>& zerothMoment
    );

    
    void periodicBoundaryFirstMoment_x(
        thrust::device_vector<FirstMoment>& firstMoment
    );

    void freeBoundaryFirstMoment_x(
        thrust::device_vector<FirstMoment>& firstMoment
    );
    void freeBoundaryFirstMoment_y(
        thrust::device_vector<FirstMoment>& firstMoment
    );

    void periodicBoundarySecondMoment_x(
        thrust::device_vector<SecondMoment>& secondMoment
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


