#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/partition.h>
#include <cmath>
#include "particle_struct.hpp"
#include "field_parameter_struct.hpp"
#include "const.hpp" 
#include "is_exist_transform.hpp"
#include "mpi.hpp"


class ParticlePush
{
private: 
    PIC2DMPI::MPIInfo& mPIInfo; 

public:
    ParticlePush(PIC2DMPI::MPIInfo& mPIInfo); 

    void pushVelocity(
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron, 
        const thrust::device_vector<MagneticField>& B, 
        const thrust::device_vector<ElectricField>& E, 
        const float dt
    );
    void pushPosition(
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron, 
        const float dt
    );

    void pushVelocityOfOneSpecies(
        thrust::device_vector<Particle>& particlesSpecies, 
        const thrust::device_vector<MagneticField>& B,
        const thrust::device_vector<ElectricField>& E, 
        const float q, const float m, const unsigned long long existNumSpecies, 
        const float dt
    );

    void pushPositionOfOneSpecies(
        thrust::device_vector<Particle>& particlesSpecies, 
        unsigned long long& existNumSpecies, 
        const float dt
    );

private:

};


