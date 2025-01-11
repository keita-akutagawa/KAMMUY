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

    thrust::device_vector<Particle> sendParticlesSpeciesLeft; 
    thrust::device_vector<Particle> sendParticlesSpeciesRight; 
    thrust::device_vector<Particle> sendParticlesSpeciesDown; 
    thrust::device_vector<Particle> sendParticlesSpeciesUp; 
    thrust::device_vector<Particle> recvParticlesSpeciesLeft; 
    thrust::device_vector<Particle> recvParticlesSpeciesRight; 
    thrust::device_vector<Particle> recvParticlesSpeciesDown; 
    thrust::device_vector<Particle> recvParticlesSpeciesUp; 

    thrust::device_vector<MagneticField> sendMagneticFieldLeft; 
    thrust::device_vector<MagneticField> sendMagneticFieldRight; 
    thrust::device_vector<MagneticField> recvMagneticFieldLeft; 
    thrust::device_vector<MagneticField> recvMagneticFieldRight;

    thrust::device_vector<MagneticField> sendMagneticFieldDown; 
    thrust::device_vector<MagneticField> sendMagneticFieldUp; 
    thrust::device_vector<MagneticField> recvMagneticFieldDown; 
    thrust::device_vector<MagneticField> recvMagneticFieldUp; 

    thrust::device_vector<ElectricField> sendElectricFieldLeft; 
    thrust::device_vector<ElectricField> sendElectricFieldRight; 
    thrust::device_vector<ElectricField> recvElectricFieldLeft; 
    thrust::device_vector<ElectricField> recvElectricFieldRight; 

    thrust::device_vector<ElectricField> sendElectricFieldDown; 
    thrust::device_vector<ElectricField> sendElectricFieldUp; 
    thrust::device_vector<ElectricField> recvElectricFieldDown; 
    thrust::device_vector<ElectricField> recvElectricFieldUp; 

    thrust::device_vector<CurrentField> sendCurrentFieldLeft; 
    thrust::device_vector<CurrentField> sendCurrentFieldRight; 
    thrust::device_vector<CurrentField> recvCurrentFieldLeft; 
    thrust::device_vector<CurrentField> recvCurrentFieldRight; 

    thrust::device_vector<CurrentField> sendCurrentFieldDown; 
    thrust::device_vector<CurrentField> sendCurrentFieldUp; 
    thrust::device_vector<CurrentField> recvCurrentFieldDown; 
    thrust::device_vector<CurrentField> recvCurrentFieldUp; 

    thrust::device_vector<ZerothMoment> sendZerothMomentLeft; 
    thrust::device_vector<ZerothMoment> sendZerothMomentRight; 
    thrust::device_vector<ZerothMoment> recvZerothMomentLeft; 
    thrust::device_vector<ZerothMoment> recvZerothMomentRight; 

    thrust::device_vector<ZerothMoment> sendZerothMomentDown; 
    thrust::device_vector<ZerothMoment> sendZerothMomentUp; 
    thrust::device_vector<ZerothMoment> recvZerothMomentDown; 
    thrust::device_vector<ZerothMoment> recvZerothMomentUp; 

    thrust::device_vector<FirstMoment> sendFirstMomentLeft; 
    thrust::device_vector<FirstMoment> sendFirstMomentRight; 
    thrust::device_vector<FirstMoment> recvFirstMomentLeft; 
    thrust::device_vector<FirstMoment> recvFirstMomentRight; 

    thrust::device_vector<FirstMoment> sendFirstMomentDown; 
    thrust::device_vector<FirstMoment> sendFirstMomentUp; 
    thrust::device_vector<FirstMoment> recvFirstMomentDown; 
    thrust::device_vector<FirstMoment> recvFirstMomentUp; 
    

public:
    BoundaryPIC(PIC2DMPI::MPIInfo& mPIInfo);

    void periodicBoundaryForInitializeParticle_x(
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
    void periodicBoundaryParticleOfOneSpecies_x(
        thrust::device_vector<Particle>& particlesSpecies, 
        unsigned long long& existNumSpecies, 
        unsigned int& numForSendParticlesSpeciesLeft, 
        unsigned int& numForSendParticlesSpeciesRight, 
        unsigned int& numForRecvParticlesSpeciesLeft, 
        unsigned int& numForRecvParticlesSpeciesRight
    );

    void freeBoundaryParticle_y(
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron
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


