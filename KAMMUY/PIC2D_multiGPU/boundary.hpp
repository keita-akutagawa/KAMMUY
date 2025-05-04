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

    thrust::device_vector<Particle> sendParticlesSpeciesLeft; 
    thrust::device_vector<Particle> sendParticlesSpeciesRight; 
    thrust::device_vector<Particle> recvParticlesSpeciesLeft; 
    thrust::device_vector<Particle> recvParticlesSpeciesRight; 

    thrust::device_vector<MagneticField> sendMagneticFieldLeft; 
    thrust::device_vector<MagneticField> sendMagneticFieldRight; 
    thrust::device_vector<MagneticField> recvMagneticFieldLeft; 
    thrust::device_vector<MagneticField> recvMagneticFieldRight;

    thrust::device_vector<ElectricField> sendElectricFieldLeft; 
    thrust::device_vector<ElectricField> sendElectricFieldRight; 
    thrust::device_vector<ElectricField> recvElectricFieldLeft; 
    thrust::device_vector<ElectricField> recvElectricFieldRight; 

    thrust::device_vector<CurrentField> sendCurrentFieldLeft; 
    thrust::device_vector<CurrentField> sendCurrentFieldRight; 
    thrust::device_vector<CurrentField> recvCurrentFieldLeft; 
    thrust::device_vector<CurrentField> recvCurrentFieldRight; 

    thrust::device_vector<ZerothMoment> sendZerothMomentLeft; 
    thrust::device_vector<ZerothMoment> sendZerothMomentRight; 
    thrust::device_vector<ZerothMoment> recvZerothMomentLeft; 
    thrust::device_vector<ZerothMoment> recvZerothMomentRight; 

    thrust::device_vector<FirstMoment> sendFirstMomentLeft; 
    thrust::device_vector<FirstMoment> sendFirstMomentRight; 
    thrust::device_vector<FirstMoment> recvFirstMomentLeft; 
    thrust::device_vector<FirstMoment> recvFirstMomentRight; 

    thrust::device_vector<SecondMoment> sendSecondMomentLeft; 
    thrust::device_vector<SecondMoment> sendSecondMomentRight; 
    thrust::device_vector<SecondMoment> recvSecondMomentLeft; 
    thrust::device_vector<SecondMoment> recvSecondMomentRight; 

public:
    BoundaryPIC(PIC2DMPI::MPIInfo& mPIInfo);

    void periodicBoundaryParticle_x(
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron
    );
    void periodicBoundaryParticleOfOneSpecies_x(
        thrust::device_vector<Particle>& particlesSpecies, 
        unsigned long long& existNumSpecies, 
        unsigned long long& numForSendParticlesSpeciesLeft, 
        unsigned long long& numForSendParticlesSpeciesRight, 
        unsigned long long& numForRecvParticlesSpeciesLeft, 
        unsigned long long& numForRecvParticlesSpeciesRight
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


