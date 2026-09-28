#ifndef MHD_BINARY_OUTPUT_HPP 
#define MHD_BINARY_OUTPUT_HPP 

#include <thrust/host_vector.h>
#include <thrust/device_vector.h> 
#include <fstream>

#include "../../../const_parameter.hpp"
#include "../../../global_function.hpp"
#include "../../../mhd_value.hpp"
#include "../output.hpp"
#include "../../../../utils/get_index.hpp"


class BinaryOutput : public Output 
{
private:
    const MHDUnsignedInt NX, NY;
    const MHDConstParameter& mHDConstParameter; 
    thrust::host_vector<MHDValue> host_U; 

public: 
    BinaryOutput(
        const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
        const MHDConstParameter& mHDConstParameter
    );

    void save(
        const thrust::device_vector<MHDValue>& U, 
        std::string addName  
    ) override;

private:
};

#endif
