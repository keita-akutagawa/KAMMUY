#ifndef MHD_OUTPUT_HPP 
#define MHD_OUTPUT_HPP 

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "../../mhd_value.hpp"


class Output
{
private: 

public: 
    virtual void save(
        const thrust::device_vector<MHDValue>& U, 
        std::string addName
    ) = 0;

    virtual ~Output() {}

private:

};

#endif
