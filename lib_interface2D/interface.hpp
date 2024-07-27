#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "const.hpp"
#include "../lib_IdealMHD2D_gpu/const.hpp"
#include "../lib_IdealMHD2D_gpu/conservation_parameter_struct.hpp"
#include "../lib_PIC2D_gpu_single/const.hpp"
#include "../lib_PIC2D_gpu_single/field_parameter_struct.hpp"
#include "../lib_pic2D_gpu_single/moment_struct.hpp"
#include "../lib_PIC2D_gpu_single/particle_struct.hpp"
#include "../lib_PIC2D_gpu_single/moment_calculater.hpp"


class Interface2D
{
private:
    int indexOfInterfaceStartInMHD;
    int indexOfInterfaceStartInPIC;
    int indexOfInterfaceEndInMHD;
    int indexOfInterfaceEndInPIC;

    thrust::device_vector<float> interlockingFunctionY;
    thrust::device_vector<float> interlockingFunctionYHalf;

    thrust::host_vector<float> host_interlockingFunctionY;
    thrust::host_vector<float> host_interlockingFunctionYHalf;

    thrust::device_vector<MagneticField> interfacePIC_B;
    thrust::device_vector<ElectricField> interfacePIC_E;
    thrust::device_vector<CurrentField> interfacePIC_current;
    thrust::device_vector<ZerothMoment> interfacePIC_zerothMomentIon;
    thrust::device_vector<ZerothMoment> interfacePIC_zerothMomentElectron;
    thrust::device_vector<FirstMoment> interfacePIC_firstMomentIon;
    thrust::device_vector<FirstMoment> interfacePIC_firstMomentElectron;
    thrust::device_vector<SecondMoment> interfacePIC_secondMomentIon;
    thrust::device_vector<SecondMoment> interfacePIC_secondMomentElectron;
    thrust::device_vector<ConservationParameter> interfaceMHD_U;

    thrust::device_vector<ZerothMoment> tmp_interfacePIC_zerothMomentIon;
    thrust::device_vector<ZerothMoment> tmp_interfacePIC_zerothMomentElectron;
    thrust::device_vector<FirstMoment> tmp_interfacePIC_firstMomentIon;
    thrust::device_vector<FirstMoment> tmp_interfacePIC_firstMomentElectron;
    thrust::device_vector<SecondMoment> tmp_interfacePIC_secondMomentIon;
    thrust::device_vector<SecondMoment> tmp_interfacePIC_secondMomentElectron;

    MomentCalculater momentCalculater;

    
public:
    Interface2D(
        int indexOfInterfaceStartInMHD, 
        int indexOfInterfaceStartInPIC
    );

    void sendMHDtoPIC_magneticField_yDirection(
        const thrust::device_vector<ConservationParameter>& U, 
        thrust::device_vector<MagneticField>& B
    );

    void sendMHDtoPIC_electricField_yDirection(
        const thrust::device_vector<ConservationParameter>& U, 
        thrust::device_vector<ElectricField>& E
    );

    void sendMHDtoPIC_currentField_yDirection(
        const thrust::device_vector<ConservationParameter>& U, 
        thrust::device_vector<CurrentField>& Current
    );

    void sendMHDtoPIC_particle(
        const thrust::device_vector<ConservationParameter>& U, 
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron
    );

    void reloadParticles(
        thrust::device_vector<Particle>& particlesSpecies
    );

    void sendPICtoMHD();

private:

};

