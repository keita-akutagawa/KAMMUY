#ifndef INTERFACE2D_HPP
#define INTERFACE2D_HPP

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cmath>
#include <curand_kernel.h>
#include <random>
#include <algorithm>
#include <thrust/fill.h>
#include <thrust/partition.h>
#include <thrust/transform.h>

#include "const_parameter.hpp"
#include "grid_parameter.hpp"
#include "reload_particles_data_struct.hpp"
#include "../10momentMHD2D/const_parameter.hpp"
#include "../10momentMHD2D/grid_parameter.hpp"
#include "../10momentMHD2D/mhd_value.hpp"
#include "../PIC2D/const_parameter.hpp"
#include "../PIC2D/grid_parameter.hpp"
#include "../PIC2D/field_parameter_struct.hpp"
#include "../PIC2D/moment_struct.hpp"
#include "../PIC2D/particle_struct.hpp"
#include "../PIC2D/moment_calculator.hpp"
#include "../PIC2D/is_exist_transform.hpp"
#include "../utils/get_index.hpp"


class Interface2D
{
private:
    InterfaceConstParameter interfaceConstParameter; 
    InterfaceGridParameter interfaceGridParameter; 
    MHDConstParameter& mHDConstParameter; 
    const MHDGridParameter& mHDGridParameter; 
    PICConstParameter& pICConstParameter; 
    const PICGridParameter& pICGridParameter; 

    InterfaceUnsignedInt GRID_SIZE_RATIO;
    InterfaceUnsignedInt START_INDEX_IN_MHD_X, START_INDEX_IN_MHD_Y;
    MHDUnsignedInt NX_MHD, NY_MHD; 
    MHDFloat DX_MHD, DY_MHD; 
    PICUnsignedInt NX_PIC, NY_PIC; 
    PICFloat DX_PIC, DY_PIC; 

    InterfaceUnsignedLongLong restartParticlesIndexIon;
    InterfaceUnsignedLongLong restartParticlesIndexElectron;

    thrust::device_vector<ReloadParticlesData> reloadParticlesDataIon;
    thrust::device_vector<ReloadParticlesData> reloadParticlesDataElectron;

    thrust::device_vector<MagneticField> B_PICtoMHD;
    thrust::device_vector<ZerothMoment> zerothMomentIon_PICtoMHD;
    thrust::device_vector<ZerothMoment> zerothMomentElectron_PICtoMHD;
    thrust::device_vector<FirstMoment> firstMomentIon_PICtoMHD;
    thrust::device_vector<FirstMoment> firstMomentElectron_PICtoMHD;
    thrust::device_vector<SecondMoment> secondMomentIon_PICtoMHD;
    thrust::device_vector<SecondMoment> secondMomentElectron_PICtoMHD;

    thrust::device_vector<MHDValue> timeInterpolatedU; 

    thrust::device_vector<InterfaceFloat> interlockingFunction; 

public:
    Interface2D(
        nlohmann::json& constJSON, 
        nlohmann::json& gridJSON, 
        const MHDUnsignedInt level, 
        MHDConstParameter& mHDConstParameter, 
        const MHDGridParameter& mHDGridParameter, 
        PICConstParameter& pICConstParameter, 
        const PICGridParameter& pICGridParameter
    );

    //MHD -> PIC

    void calculateTimeInterpolatedU(
        const thrust::device_vector<MHDValue>& UPast, 
        const thrust::device_vector<MHDValue>& UNext, 
        const PICUnsignedInt substep, 
        const PICUnsignedInt totalSubstep
    );

    thrust::device_vector<MHDValue>& getTimeInterpolatedURef();

    void sendMHDtoPIC_B(
        const thrust::device_vector<MHDValue>& U, 
        thrust::device_vector<MagneticField>& B
    );

    void sendMHDtoPIC_E(
        const thrust::device_vector<MHDValue>& U, 
        thrust::device_vector<ElectricField>& E
    );

    void sendMHDtoPIC_current(
        const thrust::device_vector<MHDValue>& U, 
        thrust::device_vector<CurrentField>& current
    );

    void deleteParticles(
        const InterfaceUnsignedLongLong seed, 
        PICUnsignedLongLong& EXIST_NUM, 
        thrust::device_vector<Particle>& particles 
    );

    void reloadParticles(
        const thrust::device_vector<ReloadParticlesData>& reloadParticlesData, 
        const InterfaceUnsignedLongLong seed, 
        thrust::device_vector<Particle>& particles, 
        PICUnsignedLongLong& EXIST_NUM
    );

    void sendMHDtoPIC_particle(
        const thrust::device_vector<MHDValue>& U, 
        const thrust::device_vector<ZerothMoment>& zerothMomentIon, 
        const thrust::device_vector<ZerothMoment>& zerothMomentElectron, 
        const thrust::device_vector<FirstMoment>& firstMomentIon, 
        const thrust::device_vector<FirstMoment>& firstMomentElectron, 
        const thrust::device_vector<SecondMoment>& secondMomentIon, 
        const thrust::device_vector<SecondMoment>& secondMomentElectron, 
        const InterfaceUnsignedLongLong seed, 
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron
    );

    // PIC -> MHD

    void resetPICtoMHDParameters();

    void calculateSpaceAveragedPICtoMHDParameters(
        const thrust::device_vector<MagneticField>& B, 
        const thrust::device_vector<ZerothMoment>& zerothMomentIon, 
        const thrust::device_vector<ZerothMoment>& zerothMomentElectron, 
        const thrust::device_vector<FirstMoment>& firstMomentIon, 
        const thrust::device_vector<FirstMoment>& firstMomentElectron, 
        const thrust::device_vector<SecondMoment>& secondMomentIon, 
        const thrust::device_vector<SecondMoment>& secondMomentElectron
    );

    void sendPICtoMHD(
        thrust::device_vector<MHDValue>& U
    );

    InterfaceGridParameter& getInterfaceGridParameterRef();
    InterfaceConstParameter& getInterfaceConstParameterRef();

private:

};

#endif

