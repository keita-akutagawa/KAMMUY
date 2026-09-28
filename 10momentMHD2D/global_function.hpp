#ifndef MHD_GLOBAL_FUNCTION_HPP
#define MHD_GLOBAL_FUNCTION_HPP 

#include <thrust/extrema.h>

#include "mhd_value.hpp"
#include "type.hpp"


template <typename T>
__host__ __device__ 
inline T sign(const T& x)
{
    return static_cast<T>((x > 0.0) - (x < 0.0));
}


template <typename T> 
__host__ __device__ 
inline T minmod(const T& x, const T& y)
{
    return sign(x) * thrust::max(static_cast<T>(0.0), thrust::min(abs(x), sign(x) * y));
}


template <>
__host__ __device__ 
inline MHDValue minmod(const MHDValue& a, const MHDValue& b) 
{
    return MHDValue(
        minmod(a.rho, b.rho),
        minmod(a.u,   b.u),
        minmod(a.v,   b.v),
        minmod(a.w,   b.w),
        minmod(a.bX,  b.bX),
        minmod(a.bY,  b.bY),
        minmod(a.bZ,  b.bZ),
        minmod(a.pXX, b.pXX), 
        minmod(a.pYY, b.pYY), 
        minmod(a.pZZ, b.pZZ), 
        minmod(a.pXY, b.pXY), 
        minmod(a.pXZ, b.pXZ), 
        minmod(a.pYZ, b.pYZ), 
        minmod(a.psi, b.psi)
    );
}


__host__ __device__ 
inline MHDFloat secondOrderDifferenceOperator(
    const MHDFloat& phiLeft, const MHDFloat& phiRight, const MHDFloat& D, const MHDInt& stencil
)
{   
    return (phiRight - phiLeft) / (stencil * D); 
}


__host__ __device__ 
inline MHDFloat differenceOperator(
    const MHDFloat& phiLeft3, const MHDFloat& phiLeft2, const MHDFloat& phiLeft1, 
    const MHDFloat& phi, 
    const MHDFloat& phiRight1, const MHDFloat& phiRight2, const MHDFloat& phiRight3, 
    const MHDFloat& D, const MHDInt& order
)
{
    if (order == 2) {
        return secondOrderDifferenceOperator(phiLeft1, phiRight1, D, 2); 
    } else if (order == 4) {
        return 4.0 / 3.0 * secondOrderDifferenceOperator(phiLeft1, phiRight1, D, 2)
             - 1.0 / 3.0 * secondOrderDifferenceOperator(phiLeft2, phiRight2, D, 4); 
    } else if (order == 6) {
        return 3.0 / 2.0 * secondOrderDifferenceOperator(phiLeft1, phiRight1, D, 2)
             - 3.0 / 5.0 * secondOrderDifferenceOperator(phiLeft2, phiRight2, D, 4) 
             + 1.0 / 10.0 * secondOrderDifferenceOperator(phiLeft3, phiRight3, D, 6);
    } else {
        printf("Not supported order! Please select order = 2, 4, or 6!\n");
        return 1e100;
    }
}


#endif
