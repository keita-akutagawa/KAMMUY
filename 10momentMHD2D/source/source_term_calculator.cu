#include "source_term_calculator.hpp"


SourceTermCalculator::SourceTermCalculator(
    const MHDInt level, 
    MHDConstParameter& mHDConstParameter, 
    const MHDGridParameter& mHDGridParameter
)
  : level(level),  
    mHDConstParameter(mHDConstParameter), 
    mHDGridParameter(mHDGridParameter), 
    source(mHDGridParameter.NX[level] * mHDGridParameter.NY[level])
{
}


thrust::device_vector<SourceValue>& SourceTermCalculator::getSourceRef()
{
    return source; 
}
