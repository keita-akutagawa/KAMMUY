#ifndef PIC_FIELD_SOLVER_HPP 
#define PIC_FIELD_SOLVER_HPP

#include <thrust/device_vector.h>
#include "field_parameter_struct.hpp"
#include "const_parameter.hpp"
#include "grid_parameter.hpp"
#include "../utils/get_index.hpp"


class FieldSolver
{
private:
    const PICUnsignedInt NX, NY; 
    const PICFloat DX, DY; 
    PICConstParameter& pICConstParameter; 
    const PICGridParameter& pICGridParameter; 

public:
    FieldSolver(
        PICConstParameter& pICConstParameter, 
        const PICGridParameter& pICGridParameter
    );

    void timeEvolutionB(
        const thrust::device_vector<ElectricField>& E, 
        const PICFloat DT,  
        thrust::device_vector<MagneticField>& B
    );

    void timeEvolutionE(
        const thrust::device_vector<MagneticField>& B, 
        const thrust::device_vector<CurrentField>& current, 
        const PICFloat DT, 
        thrust::device_vector<ElectricField>& E
    );

private:

};

#endif 
