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

    thrust::device_vector<Particle> bufferParticlesSpecies; 

public:
    BoundaryPIC();

    void boundaryParticle(
        thrust::device_vector<Particle>& particlesIon, 
        unsigned long long& existNumIon,
        thrust::device_vector<Particle>& particlesElectron,
        unsigned long long& existNumElectron
    );

    virtual void boundaryParticleSpecies(
        thrust::device_vector<Particle>& particlesSpecies, 
        unsigned long long& existNumSpecies
    );

    virtual void boundaryB(
        thrust::device_vector<MagneticField>& B  
    ); 

    virtual void boundaryE(
        thrust::device_vector<ElectricField>& E  
    );

    virtual void boundaryCurrent(
        thrust::device_vector<CurrentField>& current
    );

    virtual void boundaryZerothMoment(
        thrust::device_vector<ZerothMoment>& zerothMoment
    );

    virtual void boundaryFirstMoment(
        thrust::device_vector<FirstMoment>& firstMoment
    );

    virtual void boundarySecondMoment(
        thrust::device_vector<SecondMoment>& secondMoment
    );

private:

};

#endif


