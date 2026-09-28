#ifndef MHD_FREE_BOUNDARY_HPP 
#define MHD_FREE_BOUNDARY_HPP

#include "../boundary_for_one_side.hpp"
#include "../../const_parameter.hpp"
#include "../../grid_parameter.hpp"


class FreeBoundaryXLeft : public BoundaryForOneSide 
{
private: 
    const MHDUnsignedInt NX, NY;
    MHDConstParameter& mHDConstParameter;
    const MHDGridParameter& mHDGridParameter;

public: 
    FreeBoundaryXLeft(
        const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
        MHDConstParameter& mHDConstParameter, 
        const MHDGridParameter& mHDGridParameter
    );

    void apply(
        thrust::device_vector<MHDValue>& U
    ) override; 

private:
};


class FreeBoundaryXRight : public BoundaryForOneSide  
{
private: 
    const MHDUnsignedInt NX, NY;
    MHDConstParameter& mHDConstParameter;
    const MHDGridParameter& mHDGridParameter;

public: 
    FreeBoundaryXRight(
        const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
        MHDConstParameter& mHDConstParameter, 
        const MHDGridParameter& mHDGridParameter
    );

    void apply(
        thrust::device_vector<MHDValue>& U
    ) override; 

private:
};


class FreeBoundaryYDown : public BoundaryForOneSide  
{
private: 
    const MHDUnsignedInt NX, NY;
    MHDConstParameter& mHDConstParameter;
    const MHDGridParameter& mHDGridParameter;

public: 
    FreeBoundaryYDown(
        const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
        MHDConstParameter& mHDConstParameter, 
        const MHDGridParameter& mHDGridParameter
    );

    void apply(
        thrust::device_vector<MHDValue>& U
    ) override; 

private:
};


class FreeBoundaryYUp : public BoundaryForOneSide  
{
private: 
    const MHDUnsignedInt NX, NY;
    MHDConstParameter& mHDConstParameter;
    const MHDGridParameter& mHDGridParameter;

public: 
    FreeBoundaryYUp(
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
