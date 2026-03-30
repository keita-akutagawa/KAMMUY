#ifndef CONST_RELOAD_PARTICLES_DATA_STRUCT_H
#define CONST_RELOAD_PARTICLES_DATA_STRUCT_H


struct ReloadParticlesData
{
    int number; 
    double u;
    double v;
    double w;
    double vth;

    __host__ __device__
    ReloadParticlesData() : 
        number(0), 
        u(0.0), 
        v(0.0), 
        w(0.0),
        vth(0.0)
        {}
    
    __host__ __device__
    ReloadParticlesData& operator=(const ReloadParticlesData& other)
    {
        if (this != &other) {
            number = other.number;
            u      = other.u;
            v      = other.v;
            w      = other.w;
            vth    = other.vth;
        }
        return *this;
    }
};


#endif

