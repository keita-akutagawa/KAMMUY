#ifndef MHD_RECONSTRUCTOR_HPP 
#define MHD_RECONSTRUCTOR_HPP

#include <thrust/device_vector.h>
#include "../mhd_value.hpp"


class Reconstructor 
{
private: 

public: 

    virtual void calculateReconstructedMHDValue(
        const thrust::device_vector<MHDValue>& U, 
        const MHDUnsignedInt& shift
    ) = 0;

    virtual const thrust::device_vector<MHDValue>& getCenterMHDValueRef() const = 0; 

    virtual const thrust::device_vector<MHDValue>& getLeftMHDValueRef() const = 0; 

    virtual const thrust::device_vector<MHDValue>& getRightMHDValueRef() const = 0;

    virtual ~Reconstructor() {}

private:

    virtual void calculateCenterMHDValue(
        const thrust::device_vector<MHDValue>& U
    ) = 0;

    virtual void calculateLeftMHDValue(
        const MHDUnsignedInt& shift
    ) = 0; 

    virtual void calculateRightMHDValue(
        const MHDUnsignedInt& shift
    ) = 0;
};

#endif 
