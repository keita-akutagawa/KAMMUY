#ifndef MHD_CUSTOM_BOUNDARY_HPP 
#define MHD_CUSTOM_BOUNDARY_HPP

#include "../boundary_for_one_side.hpp"
#include "../../const_parameter.hpp"
#include "../../grid_parameter.hpp" 


class CustomBoundaryXLeft : public BoundaryForOneSide 
{
private: 
    const MHDUnsignedInt NX, NY;
    MHDConstParameter& mHDConstParameter;
    const MHDGridParameter& mHDGridParameter; 

public: 
    CustomBoundaryXLeft(
        const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
        MHDConstParameter& mHDConstParameter, 
        const MHDGridParameter& mHDGridParameter
    );

    virtual void apply(
        thrust::device_vector<MHDValue>& U
    ) override; 

private:
};


class CustomBoundaryXRight : public BoundaryForOneSide  
{
private: 
    const MHDUnsignedInt NX, NY;
    MHDConstParameter& mHDConstParameter;
    const MHDGridParameter& mHDGridParameter;

public: 
    CustomBoundaryXRight(
        const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
        MHDConstParameter& mHDConstParameter, 
        const MHDGridParameter& mHDGridParameter
    );

    virtual void apply(
        thrust::device_vector<MHDValue>& U
    ) override; 

private:
};


class CustomBoundaryYDown : public BoundaryForOneSide  
{
private: 
    const MHDUnsignedInt NX, NY;
    MHDConstParameter& mHDConstParameter;
    const MHDGridParameter& mHDGridParameter;

public: 
    CustomBoundaryYDown(
        const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
        MHDConstParameter& mHDConstParameter, 
        const MHDGridParameter& mHDGridParameter
    );

    virtual void apply(
        thrust::device_vector<MHDValue>& U
    ) override; 

private:
};


class CustomBoundaryYUp : public BoundaryForOneSide  
{
private: 
    const MHDUnsignedInt NX, NY;
    MHDConstParameter& mHDConstParameter;
    const MHDGridParameter& mHDGridParameter;

public: 
    CustomBoundaryYUp(
        const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
        MHDConstParameter& mHDConstParameter, 
        const MHDGridParameter& mHDGridParameter
    );

    virtual void apply(
        thrust::device_vector<MHDValue>& U
    ) override; 

private:
};

#endif
