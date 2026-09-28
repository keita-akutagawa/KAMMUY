#ifndef PIC_INITIALIZE_PARTICLE_HPP 
#define PIC_INITIALIZE_PARTICLE_HPP

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "particle_struct.hpp"
#include "const_parameter.hpp"
#include "grid_parameter.hpp"


class InitializeParticle
{
private:

    PICConstParameter& pICConstParameter; 
    const PICGridParameter& pICGridParameter; 

public:
    InitializeParticle(
        PICConstParameter& pICConstParameter, 
        const PICGridParameter& pICGridParameter
    );

    void uniformPosition_maxwellDistributionVelocity_eachCell(
        const PICFloat xmin, const PICFloat xmax, const PICFloat ymin, const PICFloat ymax, 
        const PICFloat bulkVx, const PICFloat bulkVy, const PICFloat bulkVz, 
        const PICFloat vxTh, const PICFloat vyTh, const PICFloat vzTh, 
        const PICUnsignedLongLong nStart, const PICUnsignedLongLong nEnd, 
        const PICUnsignedLongLong seed, 
        thrust::device_vector<Particle>& particles
    );

private:

};

#endif
