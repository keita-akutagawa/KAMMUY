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

    void boundaryParticle(
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron
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
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron
    );
    
    virtual void boundaryParticleXRight(
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron
    );

    virtual void boundaryParticleYDown(
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron
    );

    virtual void boundaryParticleYUp(
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron
    );


    virtual void boundaryBXLeft(
        thrust::device_vector<MagneticField>& B
    );

    virtual void boundaryBXRight(
        thrust::device_vector<MagneticField>& B
    );

    virtual void boundaryBYDown(
        thrust::device_vector<MagneticField>& B
    );

    virtual void boundaryBYUp(
        thrust::device_vector<MagneticField>& B
    );


    virtual void boundaryEXLeft(
        thrust::device_vector<ElectricField>& E
    );

    virtual void boundaryEXRight(
        thrust::device_vector<ElectricField>& E
    );

    virtual void boundaryEYDown(
        thrust::device_vector<ElectricField>& E
    );

    virtual void boundaryEYUp(
        thrust::device_vector<ElectricField>& E
    );


    virtual void boundaryCurrentXLeft(
        thrust::device_vector<CurrentField>& current
    );

    virtual void boundaryCurrentXRight(
        thrust::device_vector<CurrentField>& current
    );

    virtual void boundaryCurrentYDown(
        thrust::device_vector<CurrentField>& current
    );

    virtual void boundaryCurrentYUp(
        thrust::device_vector<CurrentField>& current
    );


    virtual void boundaryZerothMomentXLeft(
        thrust::device_vector<ZerothMoment>& zerothMoment
    );

    virtual void boundaryZerothMomentXRight(
        thrust::device_vector<ZerothMoment>& zerothMoment
    );

    virtual void boundaryZerothMomentYDown(
        thrust::device_vector<ZerothMoment>& zerothMoment
    );

    virtual void boundaryZerothMomentYUp(
        thrust::device_vector<ZerothMoment>& zerothMoment
    );

    
    virtual void boundaryFirstMomentXLeft(
        thrust::device_vector<FirstMoment>& firstMoment
    );

    virtual void boundaryFirstMomentXRight(
        thrust::device_vector<FirstMoment>& firstMoment
    );

    virtual void boundaryFirstMomentYDown(
        thrust::device_vector<FirstMoment>& firstMoment
    );

    virtual void boundaryFirstMomentYUp(
        thrust::device_vector<FirstMoment>& firstMoment
    );


    virtual void boundarySecondMomentXLeft(
        thrust::device_vector<SecondMoment>& secondMoment
    );

    virtual void boundarySecondMomentXRight(
        thrust::device_vector<SecondMoment>& secondMoment
    );

    virtual void boundarySecondMomentYDown(
        thrust::device_vector<SecondMoment>& secondMoment
    );

    virtual void boundarySecondMomentYUp(
        thrust::device_vector<SecondMoment>& secondMoment
    );

private:

};

#endif


