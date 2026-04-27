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

    void boundaryB(
        thrust::device_vector<MagneticField>& B
    );

    void boundaryE(
        thrust::device_vector<ElectricField>& E
    );

    void boundaryCurrent(
        thrust::device_vector<CurrentField>& current
    );

    void boundaryZerothMoment(
        thrust::device_vector<ZerothMoment>& zerothMoment
    );

    void boundaryFirstMoment(
        thrust::device_vector<FirstMoment>& firstMoment
    );

    void boundarySecondMoment(
        thrust::device_vector<SecondMoment>& secondMoment
    );
    

    virtual void boundaryParticleXLeft(
        thrust::device_vector<Particle>& particlesSpecies, 
        unsigned long long& existNumSpecies
    );
    
    virtual void boundaryParticleXRight(
        thrust::device_vector<Particle>& particlesSpecies, 
        unsigned long long& existNumSpecies
    );

    virtual void boundaryParticleYDown(
        thrust::device_vector<Particle>& particlesSpecies, 
        unsigned long long& existNumSpecies
    );

    virtual void boundaryParticleYUp(
        thrust::device_vector<Particle>& particlesSpecies, 
        unsigned long long& existNumSpecies
    );

    template<typename T> 
    void boundaryFieldXLeft(
        thrust::device_vector<T>& field
    );

    template<typename T> 
    void boundaryFieldXRight(
        thrust::device_vector<T>& field
    );

    template<typename T> 
    void boundaryFieldYDown(
        thrust::device_vector<T>& field
    );

    template<typename T> 
    void boundaryFieldYUp(
        thrust::device_vector<T>& field
    );

private:

};

#endif


