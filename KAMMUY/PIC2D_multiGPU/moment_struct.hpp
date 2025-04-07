#ifndef MOMENT_STRUCT_H
#define MOMENT_STRUCT_H


struct ZerothMoment
{
    float n;

    __host__ __device__
    ZerothMoment() : 
        n(0.0f)
        {}
    
    __host__ __device__
    ZerothMoment(float n) :
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
    ZerothMoment operator*(float scalar) const
    {
        return ZerothMoment(scalar * n);
    }

    __host__ __device__
    friend ZerothMoment operator*(float scalar, const ZerothMoment& other) 
    {
        return ZerothMoment(scalar * other.n);
    }

    __host__ __device__
    ZerothMoment operator/(float scalar) const
    {
        return ZerothMoment(n / scalar);
    }
};


struct FirstMoment
{
    float x;
    float y;
    float z;

    __host__ __device__
    FirstMoment() : 
        x(0.0f), 
        y(0.0f), 
        z(0.0f)
        {}
    
    __host__ __device__
    FirstMoment(float x, float y, float z) :
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
    FirstMoment operator*(float scalar) const
    {
        return FirstMoment(scalar * x, scalar * y, scalar * z);
    }

    __host__ __device__
    friend FirstMoment operator*(float scalar, const FirstMoment& other) 
    {
        return FirstMoment(scalar * other.x, scalar * other.y, scalar * other.z);
    }

    __host__ __device__
    FirstMoment operator/(float scalar) const
    {
        return FirstMoment(x / scalar, y / scalar, z / scalar);
    }
};


struct SecondMoment
{
    float xx;
    float yy;
    float zz;
    float xy;
    float xz;
    float yz;

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
    SecondMoment(float xx, float yy, float zz, float xy, float xz, float yz) :
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
    SecondMoment operator*(float scalar) const
    {
        return SecondMoment(scalar * xx, scalar * yy, scalar * zz, 
                            scalar * xy, scalar * xz, scalar * yz);
    }

    __host__ __device__
    friend SecondMoment operator*(float scalar, const SecondMoment& other) 
    {
        return SecondMoment(scalar * other.xx, scalar * other.yy, scalar * other.zz, 
                            scalar * other.xy, scalar * other.xz, scalar * other.yz);
    }

    __host__ __device__
    SecondMoment operator/(float scalar) const
    {
        return SecondMoment(xx / scalar, yy / scalar, zz / scalar, xy / scalar, xz / scalar, yz / scalar);
    }
};

#endif
