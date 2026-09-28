#ifndef MHD_NOISE_REMOVER_HPP
#define MHD_NOISE_REMOVER_HPP

#include <thrust/device_vector.h>
#include "const_parameter.hpp"
#include "grid_parameter.hpp"
#include "type.hpp"
#include "../utils/get_index.hpp"
#include "const_parameter.hpp"
#include "grid_parameter.hpp"
#include "mhd_value.hpp"


class NoiseRemover2D
{
private:
    MHDConstParameter& mHDConstParameter; 
    const MHDGridParameter& mHDGridParameter; 

    MHDUnsignedInt NX_MHD, NY_MHD; 

    thrust::device_vector<MHDValue> tmpU;

public:
    NoiseRemover2D(
        const MHDUnsignedInt level, 
        MHDConstParameter& mHDConstParameter, 
        const MHDGridParameter& mHDGridParameter
    );

    void convolutionU(
        thrust::device_vector<MHDValue>& U
    );

private:

};


#endif

