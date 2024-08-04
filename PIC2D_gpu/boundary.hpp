#ifndef CONST_PIC_BOUNDARY_H
#define CONST_PIC_BOUNDARY_H


#include <thrust/device_vector.h>
#include "const.hpp"
#include "particle_struct.hpp"
#include "field_parameter_struct.hpp"


class BoundaryPIC
{
private:

public:

    void periodicBoundaryParticleX(
        thrust::device_vector<Particle>& particlesIon,
        thrust::device_vector<Particle>& particlesElectron
    );
    void conductingWallBoundaryParticleX(
        thrust::device_vector<Particle>& particlesIon,
        thrust::device_vector<Particle>& particlesElectron
    );

    void conductingWallBoundaryParticleY(
        thrust::device_vector<Particle>& particlesIon,
        thrust::device_vector<Particle>& particlesElectron
    );
    void openBoundaryParticleY(
        thrust::device_vector<Particle>& particlesIon,
        thrust::device_vector<Particle>& particlesElectron
    );


    void conductingWallBoundaryBX(
        thrust::device_vector<MagneticField>& B
    );
    void symmetricBoundaryBX(
        thrust::device_vector<MagneticField>& B
    );

    void conductingWallBoundaryBY(
        thrust::device_vector<MagneticField>& B
    );
    void symmetricBoundaryBY(
        thrust::device_vector<MagneticField>& B
    );
    void freeBoundaryBY(
        thrust::device_vector<MagneticField>& B
    );


    void conductingWallBoundaryEX(
        thrust::device_vector<ElectricField>& E
    );
    void symmetricBoundaryEX(
        thrust::device_vector<ElectricField>& E
    );

    void conductingWallBoundaryEY(
        thrust::device_vector<ElectricField>& E
    );
    void symmetricBoundaryEY(
        thrust::device_vector<ElectricField>& E
    );
    void freeBoundaryEY(
        thrust::device_vector<ElectricField>& E
    );


    void conductingWallBoundaryCurrentX(
        thrust::device_vector<CurrentField>& current
    );
    void symmetricBoundaryCurrentX(
        thrust::device_vector<CurrentField>& current
    );

    void conductingWallBoundaryCurrentY(
        thrust::device_vector<CurrentField>& current
    );
    void symmetricBoundaryCurrentY(
        thrust::device_vector<CurrentField>& current
    );
    void freeBoundaryCurrentY(
        thrust::device_vector<CurrentField>& current
    );

private:

};


#endif

