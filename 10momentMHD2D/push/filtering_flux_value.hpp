#ifndef MHD_FILTERING_FLUX_VALUE_HPP
#define MHD_FILTERING_FLUX_VALUE_HPP

#include "../type.hpp"


struct FilteringFluxValue
{
    MHDFloat f0; //rho
    MHDFloat f1; //u
    MHDFloat f2; //v
    MHDFloat f3; //w
    MHDFloat f4; //bX
    MHDFloat f5; //bY
    MHDFloat f6; //bZ
    MHDFloat f7; //pXX
    MHDFloat f8; //pYY
    MHDFloat f9; //pZZ
    MHDFloat f10; //pXY
    MHDFloat f11; //pXZ
    MHDFloat f12; //pYZ
    MHDFloat f13; //psi


    __host__ __device__
    FilteringFluxValue() : 
        f0(0.0), 
        f1(0.0),
        f2(0.0),
        f3(0.0),
        f4(0.0),
        f5(0.0),
        f6(0.0),
        f7(0.0), 
        f8(0.0), 
        f9(0.0), 
        f10(0.0), 
        f11(0.0), 
        f12(0.0), 
        f13(0.0) 
        {}
    
    __host__ __device__
    FilteringFluxValue(MHDFloat f0, MHDFloat f1, MHDFloat f2, MHDFloat f3, 
                MHDFloat f4, MHDFloat f5, MHDFloat f6, MHDFloat f7, 
                MHDFloat f8, MHDFloat f9, MHDFloat f10, MHDFloat f11, 
                MHDFloat f12, MHDFloat f13):
        f0(f0), 
        f1(f1),
        f2(f2), 
        f3(f3), 
        f4(f4), 
        f5(f5), 
        f6(f6), 
        f7(f7), 
        f8(f8), 
        f9(f9), 
        f10(f10), 
        f11(f11), 
        f12(f12), 
        f13(f13) 
    {}
    
    __host__ __device__
    FilteringFluxValue operator+(const FilteringFluxValue& other) const
    {
        return FilteringFluxValue(
            f0 + other.f0, 
            f1 + other.f1, 
            f2 + other.f2, 
            f3 + other.f3, 
            f4 + other.f4, 
            f5 + other.f5, 
            f6 + other.f6, 
            f7 + other.f7, 
            f8 + other.f8, 
            f9 + other.f9, 
            f10 + other.f10, 
            f11 + other.f11, 
            f12 + other.f12, 
            f13 + other.f13
        );
    }

    __host__ __device__
    FilteringFluxValue operator-(const FilteringFluxValue& other) const
    {
        return FilteringFluxValue(
            f0 - other.f0, 
            f1 - other.f1, 
            f2 - other.f2, 
            f3 - other.f3, 
            f4 - other.f4, 
            f5 - other.f5, 
            f6 - other.f6, 
            f7 - other.f7, 
            f8 - other.f8, 
            f9 - other.f9, 
            f10 - other.f10, 
            f11 - other.f11, 
            f12 - other.f12, 
            f13 - other.f13
        );
    }

    __host__ __device__
    FilteringFluxValue operator+=(const FilteringFluxValue& other)
    {
        f0 += other.f0; 
        f1 += other.f1; 
        f2 += other.f2; 
        f3 += other.f3; 
        f4 += other.f4; 
        f5 += other.f5; 
        f6 += other.f6; 
        f7 += other.f7;
        f8 += other.f8;
        f9 += other.f9;
        f10 += other.f10;
        f11 += other.f11;
        f12 += other.f12;
        f13 += other.f13;

        return *this;
    }

    __host__ __device__
    FilteringFluxValue operator-=(const FilteringFluxValue& other)
    {
        f0 -= other.f0; 
        f1 -= other.f1; 
        f2 -= other.f2; 
        f3 -= other.f3; 
        f4 -= other.f4; 
        f5 -= other.f5; 
        f6 -= other.f6; 
        f7 -= other.f7;
        f8 -= other.f8; 
        f9 -= other.f9;
        f10 -= other.f10;
        f11 -= other.f11;
        f12 -= other.f12;
        f13 -= other.f13;

        return *this;
    }

    __host__ __device__
    FilteringFluxValue operator*(MHDFloat scalar) const
    {
        return FilteringFluxValue(
            scalar * f0,
            scalar * f1, 
            scalar * f2, 
            scalar * f3, 
            scalar * f4,
            scalar * f5,
            scalar * f6,
            scalar * f7, 
            scalar * f8, 
            scalar * f9, 
            scalar * f10, 
            scalar * f11, 
            scalar * f12, 
            scalar * f13
        );
    }

    __host__ __device__
    friend FilteringFluxValue operator*(MHDFloat scalar, const FilteringFluxValue& other) 
    {
        return other * scalar; 
    }

    __host__ __device__
    FilteringFluxValue operator/(MHDFloat scalar) const
    {
        return FilteringFluxValue(
            f0 / scalar,
            f1 / scalar, 
            f2 / scalar, 
            f3 / scalar, 
            f4 / scalar,
            f5 / scalar,
            f6 / scalar,
            f7 / scalar, 
            f8 / scalar, 
            f9 / scalar, 
            f10 / scalar, 
            f11 / scalar, 
            f12 / scalar, 
            f13 / scalar
        );
    }
};

#endif
