#ifndef PIC_FILTER_HPP 
#define PIC_FILTER_HPP

#include <thrust/device_vector.h>
#include "field_parameter_struct.hpp"
#include "particle_struct.hpp"
#include "moment_calculator.hpp"
#include "const_parameter.hpp"
#include "grid_parameter.hpp"
#include "../utils/get_index.hpp"


class Filter
{
private:
    const PICUnsignedInt NX, NY; 
    const PICFloat DX, DY; 
    PICConstParameter& pICConstParameter; 
    const PICGridParameter& pICGridParameter; 

    thrust::device_vector<RhoField> rho;
    thrust::device_vector<FilterField> F_E;
    thrust::device_vector<FilterField> F_B;

    MomentCalculator momentCalculator; 

public:
    Filter(
        PICConstParameter& pICConstParameter, 
        const PICGridParameter& pICGridParameter
    );

    void calculateRho(
        const thrust::device_vector<Particle>& particlesIon, 
        const thrust::device_vector<Particle>& particlesElectron, 
        thrust::device_vector<ZerothMoment>& zerothMomentIon, 
        thrust::device_vector<ZerothMoment>& zerothMomentElectron
    );

    void langdonMarderTypeCorrectionE(
        const PICFloat DT, 
        thrust::device_vector<ElectricField>& E
    );

    void langdonMarderTypeCorrectionB(
        const PICFloat DT, 
        thrust::device_vector<MagneticField>& B
    );

private:

};

#endif


