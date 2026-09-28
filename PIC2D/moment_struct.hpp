#ifndef PIC_MOMENT_STRUCT_HPP
#define PIC_MOMENT_STRUCT_HPP

#include "type.hpp"


struct ZerothMoment
{
    PICFloat n;

    __host__ __device__
    ZerothMoment() : 
        n(0.0f)
        {}
    
    __host__ __device__
    ZerothMoment(PICFloat n) :
        n(n)
    {}
    
    __host__ __device__
    ZerothMoment& operator=(const ZerothMoment& other)
    {
        if (this != &other) {
            n = other.n;
        }
        return *this;
    }

     __host__ __device__
    ZerothMoment operator+(const ZerothMoment& other) const
    {
        return ZerothMoment(n + other.n);
    }

    __host__ __device__
    ZerothMoment& operator+=(const ZerothMoment& other)
    {
        n += other.n;
        
        return *this;
    }

    __host__ __device__
    ZerothMoment operator*(PICFloat scalar) const
    {
        return ZerothMoment(scalar * n);
    }

    __host__ __device__
    friend ZerothMoment operator*(PICFloat scalar, const ZerothMoment& other) 
    {
        return ZerothMoment(scalar * other.n);
    }

    __host__ __device__
    ZerothMoment operator/(PICFloat scalar) const
    {
        return ZerothMoment(n / scalar);
    }
};


struct FirstMoment
{
    PICFloat x;
    PICFloat y;
    PICFloat z;

    __host__ __device__
    FirstMoment() : 
        x(0.0f), 
        y(0.0f), 
        z(0.0f)
        {}
    
    __host__ __device__
    FirstMoment(PICFloat x, PICFloat y, PICFloat z) :
        x(x),
        y(y),
        z(z)
    {}
    
    __host__ __device__
    FirstMoment& operator=(const FirstMoment& other)
    {
        if (this != &other) {
            x = other.x;
            y = other.y;
            z = other.z;
        }
        return *this;
    }

    __host__ __device__
    FirstMoment operator+(const FirstMoment& other) const
    {
        return FirstMoment(x + other.x, y + other.y, z + other.z);
    }

    __host__ __device__
    FirstMoment& operator+=(const FirstMoment& other)
    {
        x += other.x;
        y += other.y;
        z += other.z;
        
        return *this;
    }

    __host__ __device__
    FirstMoment operator*(PICFloat scalar) const
    {
        return FirstMoment(scalar * x, scalar * y, scalar * z);
    }

    __host__ __device__
    friend FirstMoment operator*(PICFloat scalar, const FirstMoment& other) 
    {
        return FirstMoment(scalar * other.x, scalar * other.y, scalar * other.z);
    }

    __host__ __device__
    FirstMoment operator/(PICFloat scalar) const
    {
        return FirstMoment(x / scalar, y / scalar, z / scalar);
    }
};


struct SecondMoment
{
    PICFloat xx;
    PICFloat yy;
    PICFloat zz;
    PICFloat xy;
    PICFloat xz;
    PICFloat yz;

    __host__ __device__
    SecondMoment() : 
        xx(0.0f), 
        yy(0.0f), 
        zz(0.0f), 
        xy(0.0f), 
        xz(0.0f), 
        yz(0.0f)
        {}
    
    __host__ __device__
    SecondMoment(PICFloat xx, PICFloat yy, PICFloat zz, PICFloat xy, PICFloat xz, PICFloat yz) :
        xx(xx), 
        yy(yy), 
        zz(zz), 
        xy(xy), 
        xz(xz), 
        yz(yz)
    {}
    
    __host__ __device__
    SecondMoment& operator=(const SecondMoment& other)
    {
        if (this != &other) {
            xx = other.xx;
            yy = other.yy;
            zz = other.zz;
            xy = other.xy;
            xz = other.xz;
            yz = other.yz;
        }
        return *this;
    }

    __host__ __device__
    SecondMoment operator+(const SecondMoment& other) const
    {
        return SecondMoment(
            xx + other.xx, yy + other.yy, zz + other.zz, 
            xy + other.xy, xz + other.xz, yz + other.yz
        );
    }

    __host__ __device__
    SecondMoment& operator+=(const SecondMoment& other)
    {
        xx += other.xx;
        yy += other.yy;
        zz += other.zz;
        xy += other.xy; 
        xz += other.xz; 
        yz += other.yz; 
        
        return *this;
    }

    __host__ __device__
    SecondMoment operator*(PICFloat scalar) const
    {
        return SecondMoment(scalar * xx, scalar * yy, scalar * zz, 
                            scalar * xy, scalar * xz, scalar * yz);
    }

    __host__ __device__
    friend SecondMoment operator*(PICFloat scalar, const SecondMoment& other) 
    {
        return SecondMoment(scalar * other.xx, scalar * other.yy, scalar * other.zz, 
                            scalar * other.xy, scalar * other.xz, scalar * other.yz);
    }

    __host__ __device__
    SecondMoment operator/(PICFloat scalar) const
    {
        return SecondMoment(xx / scalar, yy / scalar, zz / scalar, xy / scalar, xz / scalar, yz / scalar);
    }
};

#endif
