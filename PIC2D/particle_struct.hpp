#ifndef PIC_PARTICLE_STRUCT_HPP
#define PIC_PARTICLE_STRUCT_HPP

#include "type.hpp"


struct Particle
{
    PICFloat x;
    PICFloat y;
    PICFloat z;
    PICFloat ux;
    PICFloat uy; 
    PICFloat uz;
    PICFloat gamma;
    bool isExist;

    __host__ __device__
    Particle() : 
        x(0.0), 
        y(0.0), 
        z(0.0), 
        ux(0.0), 
        uy(0.0), 
        uz(0.0), 
        gamma(0.0), 
        isExist(false)
        {}
    
    __host__ __device__
    Particle& operator=(const Particle& other)
    {
        if (this != &other) {
            x = other.x;
            y = other.y;
            z = other.z;
            ux = other.ux;
            uy = other.uy;
            uz = other.uz;
            gamma = other.gamma;
            isExist = other.isExist;
        }
        return *this;
    }
};


struct ParticleField
{
    PICFloat bX;
    PICFloat bY;
    PICFloat bZ;
    PICFloat eX;
    PICFloat eY; 
    PICFloat eZ;

    __host__ __device__
    ParticleField() : 
        bX(0.0), 
        bY(0.0), 
        bZ(0.0), 
        eX(0.0), 
        eY(0.0), 
        eZ(0.0)
        {}
    
    __host__ __device__
    ParticleField& operator=(const ParticleField& other)
    {
        if (this != &other) {
            bX = other.bX;
            bY = other.bY;
            bZ = other.bZ;
            eX = other.eX;
            eY = other.eY;
            eZ = other.eZ;
        }
        return *this;
    }
};

#endif

