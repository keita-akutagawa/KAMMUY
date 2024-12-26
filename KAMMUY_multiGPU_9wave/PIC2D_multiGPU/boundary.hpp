#ifndef BOUNDARY_PIC_H
#define BOUNDARY_PIC_H

#include <thrust/device_vector.h>
#include <thrust/partition.h>
#include <thrust/transform_reduce.h>
#include "const.hpp"
#include "particle_struct.hpp"
#include "field_parameter_struct.hpp"
#include "moment_struct.hpp"
#include "mpi.hpp"


class BoundaryPIC
{
private:
    PIC2DMPI::MPIInfo& mPIInfo; 
    PIC2DMPI::MPIInfo* device_mPIInfo; 

public:
    BoundaryPIC(PIC2DMPI::MPIInfo& mPIInfo);

    void periodicBoundaryForInitializeParticle_x(
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron
    );
    void periodicBoundaryForInitializeParticle_y(
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron
    );
    void periodicBoundaryForInitializeParticleOfOneSpecies_x(
        thrust::device_vector<Particle>& particlesSpecies, 
        unsigned long long& existNumSpecies, 
        unsigned int& numForSendParticlesSpeciesLeftToRight, 
        unsigned int& numForSendParticlesSpeciesRightToLeft, 
        unsigned int& numForRecvParticlesSpeciesLeftToRight, 
        unsigned int& numForRecvParticlesSpeciesRightToLeft
    );
    void periodicBoundaryForInitializeParticleOfOneSpecies_y(
        thrust::device_vector<Particle>& particlesSpecies, 
        unsigned long long& existNumSpecies, 
        unsigned int& numForSendParticlesSpeciesDownToUp, 
        unsigned int& numForSendParticlesSpeciesUpToDown, 
        unsigned int& numForRecvParticlesSpeciesDownToUp, 
        unsigned int& numForRecvParticlesSpeciesUpToDown
    );

    void freeBoundaryForInitializeParticle_x(
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron
    );
    void freeBoundaryForInitializeParticle_y(
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron
    );
    void freeBoundaryForInitializeParticleOfOneSpecies_x(
        thrust::device_vector<Particle>& particlesSpecies, 
        unsigned long long& existNumSpecies
    );
    void freeBoundaryForInitializeParticleOfOneSpecies_y(
        thrust::device_vector<Particle>& particlesSpecies, 
        unsigned long long& existNumSpecies
    );


    void periodicBoundaryParticle_x(
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron
    );
    void periodicBoundaryParticle_y(
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron
    );
    void periodicBoundaryParticleOfOneSpecies_x(
        thrust::device_vector<Particle>& particlesSpecies, 
        unsigned long long& existNumSpecies, 
        unsigned int& numForSendParticlesSpeciesLeft, 
        unsigned int& numForSendParticlesSpeciesRight, 
        unsigned int& numForRecvParticlesSpeciesLeft, 
        unsigned int& numForRecvParticlesSpeciesRight
    );
    void periodicBoundaryParticleOfOneSpecies_y(
        thrust::device_vector<Particle>& particlesSpecies, 
        unsigned long long& existNumSpecies, 
        unsigned int& numForSendParticlesSpeciesDown, 
        unsigned int& numForSendParticlesSpeciesUp, 
        unsigned int& numForRecvParticlesSpeciesDown, 
        unsigned int& numForRecvParticlesSpeciesUp
    );

    void freeBoundaryParticle_x(
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron
    );
    void freeBoundaryParticle_y(
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron
    );
    void freeBoundaryParticleOfOneSpecies_x(
        thrust::device_vector<Particle>& particlesSpecies, 
        unsigned long long& existNumSpecies, 
        unsigned int& numForSendParticlesSpeciesLeft, 
        unsigned int& numForSendParticlesSpeciesRight, 
        unsigned int& numForRecvParticlesSpeciesLeft, 
        unsigned int& numForRecvParticlesSpeciesRight
    );
    void freeBoundaryParticleOfOneSpecies_y(
        thrust::device_vector<Particle>& particlesSpecies, 
        unsigned long long& existNumSpecies, 
        unsigned int& numForSendParticlesSpeciesDown, 
        unsigned int& numForSendParticlesSpeciesUp, 
        unsigned int& numForRecvParticlesSpeciesDown, 
        unsigned int& numForRecvParticlesSpeciesUp
    );

    void modifySendNumParticles();

    void modifySendNumParticlesSpecies(
        const unsigned int& numForSendParticlesSpeciesCornerLeftDown, 
        const unsigned int& numForSendParticlesSpeciesCornerRightDown, 
        const unsigned int& numForSendParticlesSpeciesCornerLeftUp, 
        const unsigned int& numForSendParticlesSpeciesCornerRightUp, 
        unsigned int& numForRecvParticlesSpeciesCornerLeftDown, 
        unsigned int& numForRecvParticlesSpeciesCornerRightDown, 
        unsigned int& numForRecvParticlesSpeciesCornerLeftUp, 
        unsigned int& numForRecvParticlesSpeciesCornerRightUp, 
        unsigned int& numForSendParticlesSpeciesDown, 
        unsigned int& numForSendParticlesSpeciesUp
    );
    


    void periodicBoundaryB_x(
        thrust::device_vector<MagneticField>& B
    );
    void periodicBoundaryB_y(
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
    void periodicBoundaryE_y(
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
    void periodicBoundaryCurrent_y(
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
    void periodicBoundaryZerothMoment_y(
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
    void periodicBoundaryFirstMoment_y(
        thrust::device_vector<FirstMoment>& firstMoment
    );

    void freeBoundaryFirstMoment_x(
        thrust::device_vector<FirstMoment>& firstMoment
    );
    void freeBoundaryFirstMoment_y(
        thrust::device_vector<FirstMoment>& firstMoment
    );

private:

};

#endif


