#ifndef MHD_SMR_CUSTOM_BOUNDARY_HPP 
#define MHD_SMR_CUSTOM_BOUNDARY_HPP

#include "../smr_boundary_for_one_side.hpp"
#include "../../../../const_parameter.hpp"
#include "../../../../grid_parameter.hpp" 


class SMRCustomBoundaryXLeft : public SMRBoundaryForOneSide 
{
private:
    const MHDInt level; 
    const MHDUnsignedInt NX, NY;
    const MHDUnsignedInt SMR_NX, SMR_NY;
    const MHDUnsignedInt START_INDEX_X, START_INDEX_Y;
    MHDConstParameter& mHDConstParameter;
    const MHDGridParameter& mHDGridParameter; 

public: 
    SMRCustomBoundaryXLeft(
        const MHDInt level, 
        MHDConstParameter& mHDConstParameter, 
        const MHDGridParameter& mHDGridParameter
    );

    virtual void apply(
        const thrust::device_vector<MHDValue>& coarseUPast, 
        const thrust::device_vector<MHDValue>& coarseUNext, 
        const MHDFloat timeRatio, 
        thrust::device_vector<MHDValue>& smrU
    ) override; 

private:
};


class SMRCustomBoundaryXRight : public SMRBoundaryForOneSide  
{
private:
    const MHDInt level; 
    const MHDUnsignedInt NX, NY;
    const MHDUnsignedInt SMR_NX, SMR_NY;
    const MHDUnsignedInt START_INDEX_X, START_INDEX_Y;
    MHDConstParameter& mHDConstParameter;
    const MHDGridParameter& mHDGridParameter;

public: 
    SMRCustomBoundaryXRight(
        const MHDInt level, 
        MHDConstParameter& mHDConstParameter, 
        const MHDGridParameter& mHDGridParameter
    );

    virtual void apply(
        const thrust::device_vector<MHDValue>& coarseUPast, 
        const thrust::device_vector<MHDValue>& coarseUNext, 
        const MHDFloat timeRatio, 
        thrust::device_vector<MHDValue>& smrU
    ) override; 

private:
};


class SMRCustomBoundaryYDown : public SMRBoundaryForOneSide  
{
private:
    const MHDInt level; 
    const MHDUnsignedInt NX, NY;
    const MHDUnsignedInt SMR_NX, SMR_NY;
    const MHDUnsignedInt START_INDEX_X, START_INDEX_Y;
    MHDConstParameter& mHDConstParameter;
    const MHDGridParameter& mHDGridParameter;

public: 
    SMRCustomBoundaryYDown(
        const MHDInt level, 
        MHDConstParameter& mHDConstParameter, 
        const MHDGridParameter& mHDGridParameter
    );

    virtual void apply(
        const thrust::device_vector<MHDValue>& coarseUPast, 
        const thrust::device_vector<MHDValue>& coarseUNext, 
        const MHDFloat timeRatio, 
        thrust::device_vector<MHDValue>& smrU
    ) override; 

private:
};


class SMRCustomBoundaryYUp : public SMRBoundaryForOneSide  
{
private:
    const MHDInt level; 
    const MHDUnsignedInt NX, NY;
    const MHDUnsignedInt SMR_NX, SMR_NY;
    const MHDUnsignedInt START_INDEX_X, START_INDEX_Y;
    MHDConstParameter& mHDConstParameter;
    const MHDGridParameter& mHDGridParameter;

public: 
    SMRCustomBoundaryYUp(
        const MHDInt level, 
        MHDConstParameter& mHDConstParameter, 
        const MHDGridParameter& mHDGridParameter
    );

    virtual void apply(
        const thrust::device_vector<MHDValue>& coarseUPast, 
        const thrust::device_vector<MHDValue>& coarseUNext, 
        const MHDFloat timeRatio, 
        thrust::device_vector<MHDValue>& smrU
    ) override; 

private:
};

#endif
