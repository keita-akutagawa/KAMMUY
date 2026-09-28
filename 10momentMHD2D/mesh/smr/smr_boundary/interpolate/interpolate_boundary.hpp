#ifndef MHD_SMR_INTERPOLATE_BOUNDARY_HPP 
#define MHD_SMR_INTERPOLATE_BOUNDARY_HPP

#include "../smr_boundary_for_one_side.hpp"
#include "../../../../const_parameter.hpp"
#include "../../../../grid_parameter.hpp" 


class SMRInterpolateBoundaryXLeft : public SMRBoundaryForOneSide 
{
private:
    const MHDInt level; 
    const MHDUnsignedInt NX, NY;
    const MHDUnsignedInt SMR_NX, SMR_NY;
    const MHDUnsignedInt START_INDEX_X, START_INDEX_Y;
    MHDConstParameter& mHDConstParameter;
    const MHDGridParameter& mHDGridParameter; 

public: 
    SMRInterpolateBoundaryXLeft(
        const MHDInt level, 
        MHDConstParameter& mHDConstParameter, 
        const MHDGridParameter& mHDGridParameter
    );

    void apply(
        const thrust::device_vector<MHDValue>& coarseUPast, 
        const thrust::device_vector<MHDValue>& coarseUNext, 
        const MHDFloat timeRatio, 
        thrust::device_vector<MHDValue>& smrU
    ) override; 

private:
};


class SMRInterpolateBoundaryXRight : public SMRBoundaryForOneSide  
{
private:
    const MHDInt level; 
    const MHDUnsignedInt NX, NY;
    const MHDUnsignedInt SMR_NX, SMR_NY;
    const MHDUnsignedInt START_INDEX_X, START_INDEX_Y;
    MHDConstParameter& mHDConstParameter;
    const MHDGridParameter& mHDGridParameter;

public: 
    SMRInterpolateBoundaryXRight(
        const MHDInt level, 
        MHDConstParameter& mHDConstParameter, 
        const MHDGridParameter& mHDGridParameter
    );

    void apply(
        const thrust::device_vector<MHDValue>& coarseUPast, 
        const thrust::device_vector<MHDValue>& coarseUNext, 
        const MHDFloat timeRatio, 
        thrust::device_vector<MHDValue>& smrU
    ) override; 

private:
};


class SMRInterpolateBoundaryYDown : public SMRBoundaryForOneSide  
{
private:
    const MHDInt level; 
    const MHDUnsignedInt NX, NY;
    const MHDUnsignedInt SMR_NX, SMR_NY;
    const MHDUnsignedInt START_INDEX_X, START_INDEX_Y;
    MHDConstParameter& mHDConstParameter;
    const MHDGridParameter& mHDGridParameter;

public: 
    SMRInterpolateBoundaryYDown(
        const MHDInt level, 
        MHDConstParameter& mHDConstParameter, 
        const MHDGridParameter& mHDGridParameter
    );

    void apply(
        const thrust::device_vector<MHDValue>& coarseUPast, 
        const thrust::device_vector<MHDValue>& coarseUNext, 
        const MHDFloat timeRatio, 
        thrust::device_vector<MHDValue>& smrU
    ) override; 

private:
};


class SMRInterpolateBoundaryYUp : public SMRBoundaryForOneSide  
{
private:
    const MHDInt level; 
    const MHDUnsignedInt NX, NY;
    const MHDUnsignedInt SMR_NX, SMR_NY;
    const MHDUnsignedInt START_INDEX_X, START_INDEX_Y;
    MHDConstParameter& mHDConstParameter;
    const MHDGridParameter& mHDGridParameter;

public: 
    SMRInterpolateBoundaryYUp(
        const MHDInt level, 
        MHDConstParameter& mHDConstParameter, 
        const MHDGridParameter& mHDGridParameter
    );

    void apply(
        const thrust::device_vector<MHDValue>& coarseUPast, 
        const thrust::device_vector<MHDValue>& coarseUNext, 
        const MHDFloat timeRatio, 
        thrust::device_vector<MHDValue>& smrU
    ) override; 

private:
};

#endif
