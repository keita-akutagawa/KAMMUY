#ifndef MHD_PERIODIC_BOUNDARY_HPP 
#define MHD_PERIODIC_BOUNDARY_HPP

#include "../boundary_for_one_side.hpp"
#include "../../const_parameter.hpp"
#include "../../grid_parameter.hpp"


class PeriodicBoundaryXLeft : public BoundaryForOneSide 
{
private: 
    const MHDUnsignedInt NX, NY;
    MHDConstParameter& mHDConstParameter;
    const MHDGridParameter& mHDGridParameter; 

public: 
    PeriodicBoundaryXLeft(
        const MHDUnsignedInt NX, const MHDUnsignedInt NY,  
        MHDConstParameter& mHDConstParameter, 
        const MHDGridParameter& mHDGridParameter
    );

    void apply(
        thrust::device_vector<MHDValue>& U
    ) override; 

private:
};


class PeriodicBoundaryXRight : public BoundaryForOneSide  
{
private: 
    const MHDUnsignedInt NX, NY;
    MHDConstParameter& mHDConstParameter;
    const MHDGridParameter& mHDGridParameter; 

public: 
    PeriodicBoundaryXRight(
        const MHDUnsignedInt NX, const MHDUnsignedInt NY,  
        MHDConstParameter& mHDConstParameter, 
        const MHDGridParameter& mHDGridParameter
    );

    void apply(
        thrust::device_vector<MHDValue>& U
    ) override; 

private:
};


class PeriodicBoundaryYDown : public BoundaryForOneSide  
{
private: 
    const MHDUnsignedInt NX, NY;
    MHDConstParameter& mHDConstParameter;
    const MHDGridParameter& mHDGridParameter; 

public: 
    PeriodicBoundaryYDown(
        const MHDUnsignedInt NX, const MHDUnsignedInt NY,  
        MHDConstParameter& mHDConstParameter, 
        const MHDGridParameter& mHDGridParameter
    );

    void apply(
        thrust::device_vector<MHDValue>& U
    ) override; 

private:
};


class PeriodicBoundaryYUp : public BoundaryForOneSide  
{
private: 
    const MHDUnsignedInt NX, NY;
    MHDConstParameter& mHDConstParameter;
    const MHDGridParameter& mHDGridParameter; 

public: 
    PeriodicBoundaryYUp(
        const MHDUnsignedInt NX, const MHDUnsignedInt NY,  
        MHDConstParameter& mHDConstParameter, 
        const MHDGridParameter& mHDGridParameter
    );

    void apply(
        thrust::device_vector<MHDValue>& U
    ) override; 

private:
};

#endif
