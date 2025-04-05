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
    MagneticField(float bX, float bY, float bZ) :
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
    MagneticField operator*(float scalar) const
    {
        return MagneticField(scalar * bX, scalar * bY, scalar * bZ);
    }

    __host__ __device__
    friend MagneticField operator*(float scalar, const MagneticField& other) 
    {
        return MagneticField(scalar * other.bX, scalar * other.bY, scalar * other.bZ);
    }

    __host__ __device__
    MagneticField operator/(double scalar) const
    {
        return MagneticField(bX / scalar, bY / scalar, bZ / scalar);
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
    ElectricField(float eX, float eY, float eZ) :
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
    ElectricField operator*(float scalar) const
    {
        return ElectricField(scalar * eX, scalar * eY, scalar * eZ);
    }

    __host__ __device__
    friend ElectricField operator*(float scalar, const ElectricField& other) 
    {
        return ElectricField(scalar * other.eX, scalar * other.eY, scalar * other.eZ);
    }

    __host__ __device__
    ElectricField operator/(double scalar) const
    {
        return ElectricField(eX / scalar, eY / scalar, eZ / scalar);
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
    CurrentField(float jx, float jy, float jz) :
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
    CurrentField operator*(float scalar) const
    {
        return CurrentField(scalar * jX, scalar * jY, scalar * jZ);
    }

    __host__ __device__
    friend CurrentField operator*(float scalar, const CurrentField& other) 
    {
        return CurrentField(scalar * other.jX, scalar * other.jY, scalar * other.jZ);
    }

    __host__ __device__
    CurrentField operator/(double scalar) const
    {
        return CurrentField(jX / scalar, jY / scalar, jZ / scalar);
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
    RhoField(float rho) :
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
    RhoField operator*(float scalar) const
    {
        return RhoField(scalar * rho);
    }

    __host__ __device__
    friend RhoField operator*(float scalar, const RhoField& other) 
    {
        return RhoField(scalar * other.rho);
    }
};


struct FilterField
{
    float F;

    __host__ __device__
    FilterField() : 
        F(0.0f)
        {}
    
    __host__ __device__
    FilterField(float F) :
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
    FilterField operator*(float scalar) const
    {
        return FilterField(scalar * F);
    }

    __host__ __device__
    friend FilterField operator*(float scalar, const FilterField& other) 
    {
        return FilterField(scalar * other.F);
    }
};

#endif
