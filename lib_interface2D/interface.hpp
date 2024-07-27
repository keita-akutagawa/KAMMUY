#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "const.hpp"
#include "reload_particles_data_struct.hpp"
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

    thrust::device_vector<ZerothMoment> zerothMomentIon;
    thrust::device_vector<ZerothMoment> zerothMomentElectron;
    thrust::device_vector<FirstMoment> firstMomentIon;
    thrust::device_vector<FirstMoment> firstMomentElectron;

    int reloadParticlesNumIon;
    int reloadParticlesNumElectron;
    thrust::device_vector<ReloadParticlesData> reloadParticlesDataIon;
    thrust::device_vector<ReloadParticlesData> reloadParticlesDataElectron;
    thrust::device_vector<Particle> reloadParticlesIon;
    thrust::device_vector<Particle> reloadParticlesElectron;

    MomentCalculater momentCalculater;

    
public:
    Interface2D(
        int indexStartMHD, 
        int indexStartPIC
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

    void sendPICtoMHD();

private:

    void setMoments(
        const thrust::device_vector<Particle>& particlesIon, 
        const thrust::device_vector<Particle>& particlesElectron
    );
};

