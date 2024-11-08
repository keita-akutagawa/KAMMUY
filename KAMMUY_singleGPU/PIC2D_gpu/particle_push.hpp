#include <thrust/device_vector.h>
#include "particle_struct.hpp"
#include "field_parameter_struct.hpp"
#include "const.hpp"


class ParticlePush
{
private: 

public:

    void pushVelocity(
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron, 
        const thrust::device_vector<MagneticField>& B, 
        const thrust::device_vector<ElectricField>& E, 
        double dt
    );
    void pushPosition(
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron, 
        double dt
    );

private:

    void pushVelocityOfOneSpecies(
        thrust::device_vector<Particle>& particlesSpecies, 
        const thrust::device_vector<MagneticField>& B,
        const thrust::device_vector<ElectricField>& E, 
        double q, double m, unsigned long long existNumSpecies, 
        double dt
    );

    void pushPositionOfOneSpecies(
        thrust::device_vector<Particle>& particlesSpecies, 
        unsigned long long existNumSpecies, 
        double dt
    );
};


