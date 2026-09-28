#ifndef MHD_WENO5_HPP 
#define MHD_WENO5_HPP

#include <thrust/device_vector.h>

#include "../reconstructor.hpp"
#include "../../type.hpp"
#include "../../mhd_value.hpp"
#include "../../global_function.hpp"
#include "../../const_parameter.hpp"
#include "../../../utils/get_index.hpp"


class WENO5 : public Reconstructor
{
private: 
    const MHDUnsignedInt NX, NY;
    MHDConstParameter& mHDConstParameter; 

    thrust::device_vector<MHDValue> centerMHDValue; 
    thrust::device_vector<MHDValue> leftMHDValue;
    thrust::device_vector<MHDValue> rightMHDValue;

public: 
    WENO5(
        const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
        MHDConstParameter& mHDConstParameter
    ); 

    void calculateReconstructedMHDValue(
        const thrust::device_vector<MHDValue>& U, 
        const MHDUnsignedInt& shift
    ) override;

    const thrust::device_vector<MHDValue>& getCenterMHDValueRef() const override; 

    const thrust::device_vector<MHDValue>& getLeftMHDValueRef() const override; 

    const thrust::device_vector<MHDValue>& getRightMHDValueRef() const override;

private:

    void calculateCenterMHDValue(
        const thrust::device_vector<MHDValue>& U
    ) override; 

    void calculateLeftMHDValue(
        const MHDUnsignedInt& shift
    ) override; 

    void calculateRightMHDValue(
        const MHDUnsignedInt& shift
    ) override;

};

#endif
