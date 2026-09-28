#ifndef SOURCE_TERM_CALCULATOR_HPP
#define SOURCE_TERM_CALCULATOR_HPP

#include <thrust/device_vector.h>
#include "../type.hpp"
#include "../mhd_value.hpp"
#include "../global_function.hpp"
#include "../const_parameter.hpp"
#include "../grid_parameter.hpp"
#include "source_value.hpp"
#include "../../utils/get_index.hpp"


class SourceTermCalculator 
{
private: 
    MHDInt level; 
    MHDConstParameter& mHDConstParameter;
    const MHDGridParameter& mHDGridParameter; 

    thrust::device_vector<SourceValue> source; 

public:
    SourceTermCalculator(
        const MHDInt level, 
        MHDConstParameter& mHDConstParameter, 
        const MHDGridParameter& mHDGridParameter
    );
    
    virtual void calculateSourceTerm(
        const thrust::device_vector<MHDValue>& U
    ); 

    thrust::device_vector<SourceValue>& getSourceRef();

private: 

};

#endif 
