#ifndef MOMENT_STRUCT_H
#define MOMENT_STRUCT_H


struct ZerothMoment
{
    double n;

    __host__ __device__
    ZerothMoment() : 
        n(0.0)
        {}

    __host__ __device__
    ZerothMoment(double x) :
        n(x)
    {}

    __host__ __device__
    ZerothMoment operator+(const ZerothMoment& other) const
    {
        return ZerothMoment(n + other.n);
    }

    __host__ __device__
    ZerothMoment operator*(double scalar) const
    {
        return ZerothMoment(scalar * n);
    }

    __host__ __device__
    friend ZerothMoment operator*(double scalar, const ZerothMoment& moment) 
    {
        return ZerothMoment(scalar * moment.n);
    }
};


struct FirstMoment
{
    double x;
    double y;
    double z;

    __host__ __device__
    FirstMoment() : 
        x(0.0), 
        y(0.0), 
        z(0.0)
        {}
    
    __host__ __device__
    FirstMoment(double x, double y, double z) :
        x(x),
        y(y),
        z(z)
    {}

    __host__ __device__
    FirstMoment operator+(const FirstMoment& other) const
    {
        return FirstMoment(x + other.x, y + other.y, z + other.z);
    }

    __host__ __device__
    FirstMoment operator*(double scalar) const
    {
        return FirstMoment(scalar * x, scalar * y, scalar * z);
    }

    __host__ __device__
    friend FirstMoment operator*(double scalar, const FirstMoment& moment) 
    {
        return FirstMoment(scalar * moment.x, scalar * moment.y, scalar * moment.z);
    }
};


struct SecondMoment
{
    double xx;
    double yy;
    double zz;
    double xy;
    double xz;
    double yz;

    __host__ __device__
    SecondMoment() : 
        xx(0.0), 
        yy(0.0), 
        zz(0.0), 
        xy(0.0), 
        xz(0.0), 
        yz(0.0)
        {}
    
    __host__ __device__
    SecondMoment(double xx, double yy, double zz, double xy, double xz, double yz) :
        xx(xx), 
        yy(yy), 
        zz(zz), 
        xy(xy), 
        xz(xz), 
        yz(yz)
    {}

    __host__ __device__
    SecondMoment operator+(const SecondMoment& other) const
    {
        return SecondMoment(
            xx + other.xx, yy + other.yy, zz + other.zz, 
            xy + other.xy, xz + other.xz, yz + other.yz
        );
    }

    __host__ __device__
    SecondMoment operator*(double scalar) const
    {
        return SecondMoment(scalar * xx, scalar * yy, scalar * zz, 
                            scalar * xy, scalar * xz, scalar * yz);
    }

    __host__ __device__
    friend SecondMoment operator*(double scalar, const SecondMoment& moment) 
    {
        return SecondMoment(scalar * moment.xx, scalar * moment.yy, scalar * moment.zz, 
                            scalar * moment.xy, scalar * moment.xz, scalar * moment.yz);
    }
};

#endif