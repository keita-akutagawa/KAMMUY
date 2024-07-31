#ifndef FIELD_STRUCT_H
#define FIELD_STRUCT_H

struct MagneticField
{
    float bX; 
    float bY; 
    float bZ; 

    __host__ __device__
    MagneticField() : 
        bX(0.0f),
        bY(0.0f),
        bZ(0.0f)
        {}
    
    __host__ __device__
    MagneticField(float x, float y, float z) :
        bX(x),
        bY(y),
        bZ(z)
    {}

    __host__ __device__
    MagneticField operator+(const MagneticField& other) const
    {
        return MagneticField(bX + other.bX, bY + other.bY, bZ + other.bZ);
    }
};


struct ElectricField
{
    float eX; 
    float eY; 
    float eZ; 

    __host__ __device__
    ElectricField() : 
        eX(0.0f),
        eY(0.0f),
        eZ(0.0f)
        {}
    
    __host__ __device__
    ElectricField(float x, float y, float z) :
        eX(x),
        eY(y),
        eZ(z)
    {}

    __host__ __device__
    ElectricField operator+(const ElectricField& other) const
    {
        return ElectricField(eX + other.eX, eY + other.eY, eZ + other.eZ);
    }
};


struct CurrentField
{
    float jX; 
    float jY; 
    float jZ; 

    __host__ __device__
    CurrentField() : 
        jX(0.0f),
        jY(0.0f),
        jZ(0.0f)
        {}
    
    __host__ __device__
    CurrentField(float x, float y, float z) :
        jX(x),
        jY(y),
        jZ(z)
    {}

    __host__ __device__
    CurrentField operator+(const CurrentField& other) const
    {
        return CurrentField(jX + other.jX, jY + other.jY, jZ + other.jZ);
    }
};


struct RhoField
{
    float rho; 

    __host__ __device__
    RhoField() : 
        rho(0.0f)
        {}
    
    __host__ __device__
    RhoField(float x) :
        rho(x)
    {}

    __host__ __device__
    RhoField operator+(const RhoField& other) const
    {
        return RhoField(rho + other.rho);
    }
};


struct FilterField
{
    float F;

    __host__ __device__
    FilterField() : 
        F(0.0f)
        {}
};

#endif