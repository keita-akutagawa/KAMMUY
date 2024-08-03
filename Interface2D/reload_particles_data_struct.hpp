#ifndef CONST_INTERFACE_STRUCT_H
#define CONST_INTERFACE_STRUCT_H


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
    
};


#endif

