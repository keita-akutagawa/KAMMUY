#ifndef CONST_Particle_STRUCT_H
#define CONST_Particle_STRUCT_H


struct Particle
{
    float x;
    float y;
    float z;
    float vx;
    float vy; 
    float vz;
    float gamma;
    bool isExist;

    __host__ __device__
    Particle() : 
        x(-1.0f), 
        y(-1.0f), 
        z(-1.0f), 
        vx(-1.0f), 
        vy(-1.0f), 
        vz(-1.0f), 
        gamma(-1.0f), 
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
    float operator()(const Particle& p) const {
        return p.isExist ? 1 : 0;
    }
};


struct ParticleField
{
    float bX;
    float bY;
    float bZ;
    float eX;
    float eY; 
    float eZ;

    __host__ __device__
    ParticleField() : 
        bX(0.0f), 
        bY(0.0f), 
        bZ(0.0f), 
        eX(0.0f), 
        eY(0.0f), 
        eZ(0.0f)
        {}
};


#endif
