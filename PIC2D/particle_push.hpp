#ifndef PIC_PARTICLE_PUSH_HPP 
#define PIC_PARTICLE_PUSH_HPP

#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/partition.h>
#include <cmath>
#include "particle_struct.hpp"
#include "field_parameter_struct.hpp"
#include "is_exist_transform.hpp"
#include "const_parameter.hpp"
#include "grid_parameter.hpp"
#include "../utils/get_index.hpp"


class ParticlePush
{
private: 
    const PICUnsignedInt NX, NY; 
    const PICFloat DX, DY; 
    PICConstParameter& pICConstParameter; 
    const PICGridParameter& pICGridParameter; 

public:
    ParticlePush(
        PICConstParameter& pICConstParameter, 
        const PICGridParameter& pICGridParameter
    ); 

    void pushVelocity(
        const thrust::device_vector<MagneticField>& B, 
        const thrust::device_vector<ElectricField>& E, 
        const PICFloat DT, 
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron
    );

    void pushPosition(
        const PICFloat DT, 
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron
    );

private:

    void pushVelocityOfOneSpecies(
        const thrust::device_vector<MagneticField>& B,
        const thrust::device_vector<ElectricField>& E, 
        const PICFloat Q, const PICFloat M, const PICUnsignedLongLong EXIST_NUM, 
        const PICFloat DT, 
        thrust::device_vector<Particle>& particles
    );

    void pushPositionOfOneSpecies(
        const PICUnsignedLongLong EXIST_NUM, 
        const PICFloat DT, 
        thrust::device_vector<Particle>& particles
    );
};

#endif
