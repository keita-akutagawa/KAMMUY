#ifndef MHD_HEATING_VALUE_HPP
#define MHD_HEATING_VALUE_HPP

#include "../type.hpp"


struct HeatingValue
{
    MHDFloat XX;
    MHDFloat YY;
    MHDFloat ZZ;
    MHDFloat XY;
    MHDFloat XZ;
    MHDFloat YZ;

    __host__ __device__
    HeatingValue() : 
        XX(0.0), 
        YY(0.0),
        ZZ(0.0),
        XY(0.0),
        XZ(0.0),
        YZ(0.0)
        {}
    
    __host__ __device__
    HeatingValue(MHDFloat XX, MHDFloat YY, MHDFloat ZZ, 
                 MHDFloat XY, MHDFloat XZ, MHDFloat YZ) :
        XX(XX), 
        YY(YY),
        ZZ(ZZ), 
        XY(XY), 
        XZ(XZ), 
        YZ(YZ)
    {}
};

#endif
