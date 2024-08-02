#ifndef CONST_Particle_STRUCT_H
#define CONST_Particle_STRUCT_H


struct Particle
{
    double x;
    double y;
    double z;
    double vx;
    double vy; 
    double vz;
    double gamma;
    bool isExist;

    __host__ __device__
    Particle() : 
        x(-1.0e10), 
        y(-1.0e10), 
        z(-1.0e10), 
        vx(-1.0e10), 
        vy(-1.0e10), 
        vz(-1.0e10), 
        gamma(-1.0e10), 
        isExist(false)
        {}
    
    __host__ __device__
    bool operator<(const Particle& other) const
    {
        return y < other.y;
    }
};


struct IsExistTransform
{
    __host__ __device__
    double operator()(const Particle& p) const {
        return p.isExist ? 1 : 0;
    }
};


struct ParticleField
{
    double bX;
    double bY;
    double bZ;
    double eX;
    double eY; 
    double eZ;

    __host__ __device__
    ParticleField() : 
        bX(0.0), 
        bY(0.0), 
        bZ(0.0), 
        eX(0.0), 
        eY(0.0), 
        eZ(0.0)
        {}
};


#endif
