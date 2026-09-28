#ifndef MHD_VALUE_STRUCT_HPP
#define MHD_VALUE_STRUCT_HPP

#include "type.hpp"


struct MHDValue
{
    MHDFloat rho;
    MHDFloat u;
    MHDFloat v;
    MHDFloat w;
    MHDFloat bX; 
    MHDFloat bY;
    MHDFloat bZ;
    MHDFloat pXX;
    MHDFloat pYY;
    MHDFloat pZZ;
    MHDFloat pXY;
    MHDFloat pXZ;
    MHDFloat pYZ;
    MHDFloat psi; 

    __host__ __device__
    MHDValue() : 
        rho(0.0), 
        u(0.0), 
        v(0.0), 
        w(0.0), 
        bX(0.0), 
        bY(0.0), 
        bZ(0.0), 
        pXX(0.0), 
        pYY(0.0), 
        pZZ(0.0), 
        pXY(0.0), 
        pXZ(0.0), 
        pYZ(0.0), 
        psi(0.0)
        {}
    
    __host__ __device__
    MHDValue(MHDFloat rho, MHDFloat u, MHDFloat v, MHDFloat w, 
                   MHDFloat bX, MHDFloat bY, MHDFloat bZ, 
                   MHDFloat pXX, MHDFloat pYY, MHDFloat pZZ, 
                   MHDFloat pXY, MHDFloat pXZ, MHDFloat pYZ, 
                   MHDFloat psi) :
        rho(rho), 
        u(u),
        v(v), 
        w(w), 
        bX(bX), 
        bY(bY), 
        bZ(bZ), 
        pXX(pXX), 
        pYY(pYY), 
        pZZ(pZZ), 
        pXY(pXY), 
        pXZ(pXZ), 
        pYZ(pYZ), 
        psi(psi)
    {}
    
    __host__ __device__
    MHDValue operator+(const MHDValue& other) const
    {
        return MHDValue(
            rho + other.rho, 
            u   + other.u, 
            v   + other.v, 
            w   + other.w, 
            bX  + other.bX, 
            bY  + other.bY, 
            bZ  + other.bZ, 
            pXX + other.pXX, 
            pYY + other.pYY, 
            pZZ + other.pZZ, 
            pXY + other.pXY, 
            pXZ + other.pXZ, 
            pYZ + other.pYZ, 
            psi + other.psi
        );
    }

    __host__ __device__
    MHDValue operator-(const MHDValue& other) const
    {
        return MHDValue(
            rho - other.rho, 
            u   - other.u, 
            v   - other.v, 
            w   - other.w, 
            bX  - other.bX, 
            bY  - other.bY, 
            bZ  - other.bZ, 
            pXX - other.pXX, 
            pYY - other.pYY, 
            pZZ - other.pZZ, 
            pXY - other.pXY, 
            pXZ - other.pXZ, 
            pYZ - other.pYZ, 
            psi - other.psi
        );
    }
    
    __host__ __device__
    MHDValue& operator+=(const MHDValue& other)
    {
        rho += other.rho; 
        u   += other.u; 
        v   += other.v; 
        w   += other.w; 
        bX  += other.bX; 
        bY  += other.bY; 
        bZ  += other.bZ; 
        pXX += other.pXX; 
        pYY += other.pYY; 
        pZZ += other.pZZ; 
        pXY += other.pXY; 
        pXZ += other.pXZ; 
        pYZ += other.pYZ; 
        psi += other.psi;
        
        return *this;
    }

    __host__ __device__
    MHDValue& operator-=(const MHDValue& other)
    {
        rho -= other.rho; 
        u   -= other.u; 
        v   -= other.v; 
        w   -= other.w; 
        bX  -= other.bX; 
        bY  -= other.bY; 
        bZ  -= other.bZ; 
        pXX -= other.pXX; 
        pYY -= other.pYY; 
        pZZ -= other.pZZ; 
        pXY -= other.pXY; 
        pXZ -= other.pXZ; 
        pYZ -= other.pYZ; 
        psi -= other.psi;
        
        return *this;
    }

    __host__ __device__
    MHDValue operator*(MHDFloat scalar) const
    {
        return MHDValue(
            scalar * rho, 
            scalar * u, 
            scalar * v, 
            scalar * w, 
            scalar * bX, 
            scalar * bY, 
            scalar * bZ, 
            scalar * pXX, 
            scalar * pYY, 
            scalar * pZZ, 
            scalar * pXY, 
            scalar * pXZ, 
            scalar * pYZ, 
            scalar * psi
        );
    }

    __host__ __device__
    friend MHDValue operator*(MHDFloat scalar, const MHDValue& other) 
    {
        return other * scalar; 
    }

    __host__ __device__
    MHDValue operator/(MHDFloat scalar) const
    {
        return MHDValue(
            rho / scalar, 
            u / scalar, 
            v / scalar, 
            w / scalar, 
            bX / scalar, 
            bY / scalar, 
            bZ / scalar, 
            pXX / scalar, 
            pYY / scalar, 
            pZZ / scalar, 
            pXY / scalar, 
            pXZ / scalar, 
            pYZ / scalar, 
            psi / scalar
        );
    }
};

#endif
