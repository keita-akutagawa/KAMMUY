#ifndef MHD_SOURCE_VALUE_HPP
#define MHD_SOURCE_VALUE_HPP

#include "../type.hpp"


struct SourceValue
{
    MHDFloat s0; //rho
    MHDFloat s1; //rho * u (注意)
    MHDFloat s2; //rho * v (注意)
    MHDFloat s3; //rho * w (注意)
    MHDFloat s4; //bX
    MHDFloat s5; //bY
    MHDFloat s6; //bZ
    MHDFloat s7; //pXX
    MHDFloat s8; //pYY
    MHDFloat s9; //pZZ
    MHDFloat s10; //pXY
    MHDFloat s11; //pXZ
    MHDFloat s12; //pYZ
    MHDFloat s13; //psi


    __host__ __device__
    SourceValue() : 
        s0(0.0), 
        s1(0.0),
        s2(0.0),
        s3(0.0),
        s4(0.0),
        s5(0.0),
        s6(0.0),
        s7(0.0), 
        s8(0.0), 
        s9(0.0), 
        s10(0.0), 
        s11(0.0), 
        s12(0.0), 
        s13(0.0) 
        {}
    
    __host__ __device__
    SourceValue(MHDFloat s0, MHDFloat s1, MHDFloat s2, MHDFloat s3, 
                MHDFloat s4, MHDFloat s5, MHDFloat s6, MHDFloat s7, 
                MHDFloat s8, MHDFloat s9, MHDFloat s10, MHDFloat s11, 
                MHDFloat s12, MHDFloat s13):
        s0(s0), 
        s1(s1),
        s2(s2), 
        s3(s3), 
        s4(s4), 
        s5(s5), 
        s6(s6), 
        s7(s7), 
        s8(s8), 
        s9(s9), 
        s10(s10), 
        s11(s11), 
        s12(s12), 
        s13(s13) 
    {}
    
    __host__ __device__
    SourceValue operator+(const SourceValue& other) const
    {
        return SourceValue(
            s0 + other.s0, 
            s1 + other.s1, 
            s2 + other.s2, 
            s3 + other.s3, 
            s4 + other.s4, 
            s5 + other.s5, 
            s6 + other.s6, 
            s7 + other.s7, 
            s8 + other.s8, 
            s9 + other.s9, 
            s10 + other.s10, 
            s11 + other.s11, 
            s12 + other.s12, 
            s13 + other.s13
        );
    }

    __host__ __device__
    SourceValue operator-(const SourceValue& other) const
    {
        return SourceValue(
            s0 - other.s0, 
            s1 - other.s1, 
            s2 - other.s2, 
            s3 - other.s3, 
            s4 - other.s4, 
            s5 - other.s5, 
            s6 - other.s6, 
            s7 - other.s7, 
            s8 - other.s8, 
            s9 - other.s9, 
            s10 - other.s10, 
            s11 - other.s11, 
            s12 - other.s12, 
            s13 - other.s13
        );
    }

    __host__ __device__
    SourceValue operator+=(const SourceValue& other)
    {
        s0 += other.s0; 
        s1 += other.s1; 
        s2 += other.s2; 
        s3 += other.s3; 
        s4 += other.s4; 
        s5 += other.s5; 
        s6 += other.s6; 
        s7 += other.s7;
        s8 += other.s8;
        s9 += other.s9;
        s10 += other.s10;
        s11 += other.s11;
        s12 += other.s12;
        s13 += other.s13;

        return *this;
    }

    __host__ __device__
    SourceValue operator-=(const SourceValue& other)
    {
        s0 -= other.s0; 
        s1 -= other.s1; 
        s2 -= other.s2; 
        s3 -= other.s3; 
        s4 -= other.s4; 
        s5 -= other.s5; 
        s6 -= other.s6; 
        s7 -= other.s7;
        s8 -= other.s8; 
        s9 -= other.s9;
        s10 -= other.s10;
        s11 -= other.s11;
        s12 -= other.s12;
        s13 -= other.s13;

        return *this;
    }

    __host__ __device__
    SourceValue operator*(MHDFloat scalar) const
    {
        return SourceValue(
            scalar * s0,
            scalar * s1, 
            scalar * s2, 
            scalar * s3, 
            scalar * s4,
            scalar * s5,
            scalar * s6,
            scalar * s7, 
            scalar * s8, 
            scalar * s9, 
            scalar * s10, 
            scalar * s11, 
            scalar * s12, 
            scalar * s13
        );
    }

    __host__ __device__
    friend SourceValue operator*(MHDFloat scalar, const SourceValue& other) 
    {
        return other * scalar; 
    }

    __host__ __device__
    SourceValue operator/(MHDFloat scalar) const
    {
        return SourceValue(
            s0 / scalar,
            s1 / scalar, 
            s2 / scalar, 
            s3 / scalar, 
            s4 / scalar,
            s5 / scalar,
            s6 / scalar,
            s7 / scalar, 
            s8 / scalar, 
            s9 / scalar, 
            s10 / scalar, 
            s11 / scalar, 
            s12 / scalar, 
            s13 / scalar
        );
    }
};

#endif
