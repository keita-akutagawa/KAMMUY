#ifndef PIC_BOUNDARY_HPP
#define PIC_BOUNDARY_HPP

#include <thrust/device_vector.h>
#include <thrust/partition.h>
#include <thrust/transform_reduce.h>
#include "const_parameter.hpp"
#include "grid_parameter.hpp"
#include "particle_struct.hpp"
#include "field_parameter_struct.hpp"
#include "moment_struct.hpp"
#include "type.hpp"
#include "../utils/get_index.hpp"


class PICBoundary
{
private:
    const PICUnsignedInt NX, NY; 
    const PICFloat DX, DY; 
    PICConstParameter& pICConstParameter; 
    const PICGridParameter& pICGridParameter; 

    thrust::device_vector<Particle> bufferParticlesSpecies; 

public:
    PICBoundary(
        PICConstParameter& pICConstParameter, 
        const PICGridParameter& pICGridParameter
    );

    void freeBoundaryParticleX(
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron
    );

    void freeBoundaryParticleY(
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron
    );

    void freeBoundaryParticleSpeciesX(
        thrust::device_vector<Particle>& particlesSpecies, 
        PICUnsignedLongLong& EXIST_NUM
    );

    void freeBoundaryParticleSpeciesY(
        thrust::device_vector<Particle>& particlesSpecies, 
        PICUnsignedLongLong& EXIST_NUM
    );

    template <typename FieldType>
    void freeBoundaryFieldX(
        thrust::device_vector<FieldType>& field
    );

    template <typename FieldType>
    void freeBoundaryFieldY(
        thrust::device_vector<FieldType>& field
    );


private:

};

#endif


