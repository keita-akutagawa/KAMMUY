#ifndef PIC2D_HPP 
#define PIC2D_HPP

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "initialize_particle.hpp"
#include "particle_push.hpp"
#include "field_solver.hpp"
#include "current_calculator.hpp"
#include "moment_calculator.hpp"
#include "filter.hpp"
#include "particle_struct.hpp"
#include "field_parameter_struct.hpp"
#include "moment_struct.hpp"
#include "const_parameter.hpp"
#include "../utils/get_index.hpp"
#include "boundary.hpp"

#include "../10momentMHD2D/mhd_value.hpp"
#include "../Interface2D/interface2D.hpp"
#include "../10momentMHD2D/const_parameter.hpp"
#include "../10momentMHD2D/grid_parameter.hpp"


class PIC2D
{
private:
    PICConstParameter pICConstParameter; 
    PICGridParameter pICGridParameter; 

    const PICUnsignedInt NX, NY; 
    const PICFloat DX, DY; 

    thrust::device_vector<Particle> particlesIon;
    thrust::device_vector<Particle> particlesElectron;
    thrust::device_vector<ElectricField> E;
    thrust::device_vector<ElectricField> tmpE;
    thrust::device_vector<MagneticField> B;
    thrust::device_vector<MagneticField> tmpB;
    thrust::device_vector<CurrentField> current;
    thrust::device_vector<CurrentField> tmpCurrent;
    thrust::device_vector<ZerothMoment> zerothMomentIon;
    thrust::device_vector<ZerothMoment> zerothMomentElectron;
    thrust::device_vector<FirstMoment> firstMomentIon;
    thrust::device_vector<FirstMoment> firstMomentElectron;
    thrust::device_vector<SecondMoment> secondMomentIon;
    thrust::device_vector<SecondMoment> secondMomentElectron;

    thrust::host_vector<Particle> host_particlesIon;
    thrust::host_vector<Particle> host_particlesElectron;
    thrust::host_vector<ElectricField> host_E;
    thrust::host_vector<MagneticField> host_B; 
    thrust::host_vector<CurrentField> host_current;
    thrust::host_vector<ZerothMoment> host_zerothMomentIon;
    thrust::host_vector<ZerothMoment> host_zerothMomentElectron;
    thrust::host_vector<FirstMoment> host_firstMomentIon;
    thrust::host_vector<FirstMoment> host_firstMomentElectron;
    thrust::host_vector<SecondMoment> host_secondMomentIon;
    thrust::host_vector<SecondMoment> host_secondMomentElectron;

    InitializeParticle initializeParticle;
    ParticlePush particlePush;
    FieldSolver fieldSolver;
    CurrentCalculator currentCalculator;
    MomentCalculator momentCalculator;
    Filter filter;
    PICBoundary pICBoundary; 

public:
    PIC2D(
        nlohmann::json& constJSON, 
        nlohmann::json& gridJSON
    );
    
    virtual void initialize();

    void oneStep(
        Interface2D& interface2D, 
        const thrust::device_vector<MHDValue>& timeInterpolatedU, 
        const PICUnsignedLongLong& seedForReload
    );

    void saveFields();

    void saveFullMoments();

    void saveZerothMoments();

    void saveFirstMoments();

    void saveSecondMoments();

    void saveParticle();

    thrust::host_vector<MagneticField>& getHostBRef();

    thrust::device_vector<MagneticField>& getBRef();

    thrust::device_vector<MagneticField>& getTmpBRef();

    thrust::host_vector<ElectricField>& getHostERef(); 

    thrust::device_vector<ElectricField>& getERef(); 

    thrust::host_vector<Particle>& getHostParticlesIonRef();

    thrust::device_vector<Particle>& getParticlesIonRef();

    thrust::host_vector<Particle>& getHostParticlesElectronRef();

    thrust::device_vector<Particle>& getParticlesElectronRef();

    thrust::device_vector<ZerothMoment>& getZerothMomentIonRef();

    thrust::device_vector<ZerothMoment>& getZerothMomentElectronRef();

    thrust::device_vector<FirstMoment>& getFirstMomentIonRef();

    thrust::device_vector<FirstMoment>& getFirstMomentElectronRef();

    thrust::device_vector<SecondMoment>& getSecondMomentIonRef();

    thrust::device_vector<SecondMoment>& getSecondMomentElectronRef();

    void calculateFullMoments();

    void calculateZerothMoments();

    void calculateFirstMoments();

    void calculateSecondMoments();

    PICGridParameter& getPICGridParameterRef();
    PICConstParameter& getPICConstParameterRef();

private:

};

#endif
