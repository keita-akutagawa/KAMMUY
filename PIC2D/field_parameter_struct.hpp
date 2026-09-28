#ifndef PIC_FIELD_STRUCT_HPP
#define PIC_FIELD_STRUCT_HPP

#include "type.hpp"


struct MagneticField
{
    PICFloat bX; 
    PICFloat bY; 
    PICFloat bZ; 

    __host__ __device__
    MagneticField() : 
        bX(0.0),
        bY(0.0),
        bZ(0.0)
        {}
    
    __host__ __device__
    MagneticField(PICFloat bX, PICFloat bY, PICFloat bZ) :
        bX(bX),
        bY(bY),
        bZ(bZ)
    {}

    __host__ __device__
    MagneticField& operator=(const MagneticField& other)
    {
        if (this != &other) {
            bX = other.bX;
            bY = other.bY;
            bZ = other.bZ;
        }
        return *this;
    }

    __host__ __device__
    MagneticField operator+(const MagneticField& other) const
    {
        return MagneticField(bX + other.bX, bY + other.bY, bZ + other.bZ);
    }

    __host__ __device__
    MagneticField& operator+=(const MagneticField& other)
    {
        bX += other.bX;
        bY += other.bY;
        bZ += other.bZ;
        
        return *this;
    }

    __host__ __device__
    MagneticField operator*(PICFloat scalar) const
    {
        return MagneticField(scalar * bX, scalar * bY, scalar * bZ);
    }

    __host__ __device__
    friend MagneticField operator*(PICFloat scalar, const MagneticField& other) 
    {
        return MagneticField(scalar * other.bX, scalar * other.bY, scalar * other.bZ);
    }

    __host__ __device__
    MagneticField operator/(PICFloat scalar) const
    {
        return MagneticField(bX / scalar, bY / scalar, bZ / scalar);
    }
};


struct ElectricField
{
    PICFloat eX; 
    PICFloat eY; 
    PICFloat eZ; 

    __host__ __device__
    ElectricField() : 
        eX(0.0),
        eY(0.0),
        eZ(0.0)
        {}
    
    __host__ __device__
    ElectricField(PICFloat eX, PICFloat eY, PICFloat eZ) :
        eX(eX),
        eY(eY),
        eZ(eZ)
    {}
    
    __host__ __device__
    ElectricField& operator=(const ElectricField& other)
    {
        if (this != &other) {
            eX = other.eX;
            eY = other.eY;
            eZ = other.eZ;
        }
        return *this;
    }

    __host__ __device__
    ElectricField operator+(const ElectricField& other) const
    {
        return ElectricField(eX + other.eX, eY + other.eY, eZ + other.eZ);
    }

    __host__ __device__
    ElectricField& operator+=(const ElectricField& other)
    {
        eX += other.eX;
        eY += other.eY;
        eZ += other.eZ;
        
        return *this;
    }

    __host__ __device__
    ElectricField operator*(PICFloat scalar) const
    {
        return ElectricField(scalar * eX, scalar * eY, scalar * eZ);
    }

    __host__ __device__
    friend ElectricField operator*(PICFloat scalar, const ElectricField& other) 
    {
        return ElectricField(scalar * other.eX, scalar * other.eY, scalar * other.eZ);
    }

    __host__ __device__
    ElectricField operator/(PICFloat scalar) const
    {
        return ElectricField(eX / scalar, eY / scalar, eZ / scalar);
    }
};


struct CurrentField
{
    PICFloat jX; 
    PICFloat jY; 
    PICFloat jZ; 

    __host__ __device__
    CurrentField() : 
        jX(0.0),
        jY(0.0),
        jZ(0.0)
        {}
    
    __host__ __device__
    CurrentField(PICFloat jx, PICFloat jy, PICFloat jz) :
        jX(jx),
        jY(jy),
        jZ(jz)
    {}
    
    __host__ __device__
    CurrentField& operator=(const CurrentField& other)
    {
        if (this != &other) {
            jX = other.jX;
            jY = other.jY;
            jZ = other.jZ;
        }
        return *this;
    }

    __host__ __device__
    CurrentField operator+(const CurrentField& other) const
    {
        return CurrentField(jX + other.jX, jY + other.jY, jZ + other.jZ);
    }

    __host__ __device__
    CurrentField& operator+=(const CurrentField& other)
    {
        jX += other.jX;
        jY += other.jY;
        jZ += other.jZ;
        
        return *this;
    }

    __host__ __device__
    CurrentField operator*(PICFloat scalar) const
    {
        return CurrentField(scalar * jX, scalar * jY, scalar * jZ);
    }

    __host__ __device__
    friend CurrentField operator*(PICFloat scalar, const CurrentField& other) 
    {
        return CurrentField(scalar * other.jX, scalar * other.jY, scalar * other.jZ);
    }

    __host__ __device__
    CurrentField operator/(PICFloat scalar) const
    {
        return CurrentField(jX / scalar, jY / scalar, jZ / scalar);
    }
};


struct RhoField
{
    PICFloat rho; 

    __host__ __device__
    RhoField() : 
        rho(0.0)
        {}
    
    __host__ __device__
    RhoField(PICFloat rho) :
        rho(rho)
    {}
    
    __host__ __device__
    RhoField& operator=(const RhoField& other)
    {
        if (this != &other) {
            rho = other.rho;
        }
        return *this;
    }

    __host__ __device__
    RhoField operator+(const RhoField& other) const
    {
        return RhoField(rho + other.rho);
    }

    __host__ __device__
    RhoField& operator+=(const RhoField& other)
    {
        rho += other.rho;
        
        return *this;
    }

    __host__ __device__
    RhoField operator*(PICFloat scalar) const
    {
        return RhoField(scalar * rho);
    }

    __host__ __device__
    friend RhoField operator*(PICFloat scalar, const RhoField& other) 
    {
        return RhoField(scalar * other.rho);
    }
};


struct FilterField
{
    PICFloat F;

    __host__ __device__
    FilterField() : 
        F(0.0)
        {}
    
    __host__ __device__
    FilterField(PICFloat F) :
        F(F)
    {}
    
    __host__ __device__
    FilterField& operator=(const FilterField& other)
    {
        if (this != &other) {
            F = other.F;
        }
        return *this;
    }

    __host__ __device__
    FilterField operator+(const FilterField& other) const
    {
        return FilterField(F + other.F);
    }

    __host__ __device__
    FilterField& operator+=(const FilterField& other)
    {
        F += other.F;
        
        return *this;
    }

    __host__ __device__
    FilterField operator*(PICFloat scalar) const
    {
        return FilterField(scalar * F);
    }

    __host__ __device__
    friend FilterField operator*(PICFloat scalar, const FilterField& other) 
    {
        return FilterField(scalar * other.F);
    }
};

#endif
