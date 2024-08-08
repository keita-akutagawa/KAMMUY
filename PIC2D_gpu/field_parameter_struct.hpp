#ifndef FIELD_STRUCT_H
#define FIELD_STRUCT_H

struct MagneticField
{
    double bX; 
    double bY; 
    double bZ; 

    __host__ __device__
    MagneticField() : 
        bX(0.0),
        bY(0.0),
        bZ(0.0)
        {}
    
    __host__ __device__
    MagneticField(double x, double y, double z) :
        bX(x),
        bY(y),
        bZ(z)
    {}

    __host__ __device__
    MagneticField operator+(const MagneticField& other) const
    {
        return MagneticField(bX + other.bX, bY + other.bY, bZ + other.bZ);
    }

    __host__ __device__
    MagneticField operator*(double scalar) const
    {
        return MagneticField(scalar * bX, scalar * bY, scalar * bZ);
    }

    __host__ __device__
    friend MagneticField operator*(double scalar, const MagneticField& field) 
    {
        return MagneticField(scalar * field.bX, scalar * field.bY, scalar * field.bZ);
    }
};


struct ElectricField
{
    double eX; 
    double eY; 
    double eZ; 

    __host__ __device__
    ElectricField() : 
        eX(0.0),
        eY(0.0),
        eZ(0.0)
        {}
    
    __host__ __device__
    ElectricField(double x, double y, double z) :
        eX(x),
        eY(y),
        eZ(z)
    {}

    __host__ __device__
    ElectricField operator+(const ElectricField& other) const
    {
        return ElectricField(eX + other.eX, eY + other.eY, eZ + other.eZ);
    }

    __host__ __device__
    ElectricField operator*(double scalar) const
    {
        return ElectricField(scalar * eX, scalar * eY, scalar * eZ);
    }

    __host__ __device__
    friend ElectricField operator*(double scalar, const ElectricField& field) 
    {
        return ElectricField(scalar * field.eX, scalar * field.eY, scalar * field.eZ);
    }
};


struct CurrentField
{
    double jX; 
    double jY; 
    double jZ; 

    __host__ __device__
    CurrentField() : 
        jX(0.0),
        jY(0.0),
        jZ(0.0)
        {}
    
    __host__ __device__
    CurrentField(double x, double y, double z) :
        jX(x),
        jY(y),
        jZ(z)
    {}

    __host__ __device__
    CurrentField operator+(const CurrentField& other) const
    {
        return CurrentField(jX + other.jX, jY + other.jY, jZ + other.jZ);
    }

    __host__ __device__
    CurrentField operator*(double scalar) const
    {
        return CurrentField(scalar * jX, scalar * jY, scalar * jZ);
    }

    __host__ __device__
    friend CurrentField operator*(double scalar, const CurrentField& field) 
    {
        return CurrentField(scalar * field.jX, scalar * field.jY, scalar * field.jZ);
    }
};


struct RhoField
{
    double rho; 

    __host__ __device__
    RhoField() : 
        rho(0.0)
        {}
    
    __host__ __device__
    RhoField(double x) :
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
    double F;

    __host__ __device__
    FilterField() : 
        F(0.0)
        {}
};

#endif