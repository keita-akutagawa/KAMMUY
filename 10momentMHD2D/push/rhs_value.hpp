#ifndef MHD_RHS_VALUE_HPP
#define MHD_RHS_VALUE_HPP

#include "../type.hpp"


struct RHSValue
{
    MHDFloat v0;
    MHDFloat v1;
    MHDFloat v2;
    MHDFloat v3;
    MHDFloat v4;
    MHDFloat v5;
    MHDFloat v6;
    MHDFloat v7;
    MHDFloat v8;
    MHDFloat v9;
    MHDFloat v10;
    MHDFloat v11;
    MHDFloat v12;
    MHDFloat v13;

    __host__ __device__
    RHSValue() : 
        v0(0.0), 
        v1(0.0),
        v2(0.0),
        v3(0.0),
        v4(0.0),
        v5(0.0),
        v6(0.0),
        v7(0.0), 
        v8(0.0), 
        v9(0.0), 
        v10(0.0), 
        v11(0.0), 
        v12(0.0), 
        v13(0.0) 
        {}
    
    __host__ __device__
    RHSValue(MHDFloat v0, MHDFloat v1, MHDFloat v2, MHDFloat v3, 
             MHDFloat v4, MHDFloat v5, MHDFloat v6, MHDFloat v7, 
             MHDFloat v8, MHDFloat v9, MHDFloat v10, MHDFloat v11, 
             MHDFloat v12, MHDFloat v13) :
        v0(v0), 
        v1(v1),
        v2(v2), 
        v3(v3), 
        v4(v4), 
        v5(v5), 
        v6(v6), 
        v7(v7), 
        v8(v8), 
        v9(v9), 
        v10(v10), 
        v11(v11), 
        v12(v12), 
        v13(v13)  
    {}
};

#endif
