#ifndef CONST_RELOAD_PARTICLES_DATA_STRUCT_H
#define CONST_RELOAD_PARTICLES_DATA_STRUCT_H


struct ReloadParticlesData
{
    unsigned long long numAndIndex; 
    float u;
    float v;
    float w;
    float vth;

    __host__ __device__
    ReloadParticlesData() : 
        numAndIndex(0), 
        u(0.0f), 
        v(0.0f), 
        w(0.0f),
        vth(0.0f)
        {}
    
    __host__ __device__
    ReloadParticlesData& operator=(const ReloadParticlesData& other)
    {
        if (this != &other) {
            numAndIndex = other.numAndIndex;
            u           = other.u;
            v           = other.v;
            w           = other.w;
            vth         = other.vth;
        }
        return *this;
    }
};


#endif

