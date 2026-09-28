#ifndef INTERFACE_RELOAD_PARTICLES_DATA_STRUCT_HPP
#define INTERFACE_RELOAD_PARTICLES_DATA_STRUCT_HPP

#include "type.hpp"

struct ReloadParticlesData
{
    InterfaceUnsignedInt number; 
    InterfaceFloat u;
    InterfaceFloat v;
    InterfaceFloat w;
    InterfaceFloat L11;
    InterfaceFloat L21;
    InterfaceFloat L22;
    InterfaceFloat L31;
    InterfaceFloat L32;
    InterfaceFloat L33;

    __host__ __device__
    ReloadParticlesData() : 
        number(0), 
        u(0.0), 
        v(0.0), 
        w(0.0),
        L11(0.0),
        L21(0.0),
        L22(0.0),
        L31(0.0),
        L32(0.0),
        L33(0.0)
        {}
    
    __host__ __device__
    ReloadParticlesData& operator=(const ReloadParticlesData& other)
    {
        if (this != &other) {
            number = other.number;
            u      = other.u;
            v      = other.v;
            w      = other.w;
            L11 = other.L11;
            L21 = other.L21;
            L22 = other.L22;
            L31 = other.L31;
            L32 = other.L32;
            L33 = other.L33;
        }
        return *this;
    }
};


#endif

