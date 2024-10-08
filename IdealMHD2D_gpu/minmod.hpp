#include <thrust/device_vector.h>
#include <cmath>
#include "const.hpp"


using namespace IdealMHD2DConst;

struct MinMod
{
    __device__
    double operator()(const double& x, const double& y) const
    {
        int sign_x = (x > 0) - (x < 0);
        double abs_x = std::abs(x);

        return sign_x * thrust::max(thrust::min(abs_x, sign_x * y), device_EPS_MHD);
    }
};


