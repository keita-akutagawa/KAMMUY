#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "const.hpp"
#include "reload_particles_data_struct.hpp"
#include "../IdealMHD2D_gpu/const.hpp"
#include "../IdealMHD2D_gpu/conservation_parameter_struct.hpp"
#include "../PIC2D_gpu_single/const.hpp"
#include "../PIC2D_gpu_single/field_parameter_struct.hpp"
#include "../pic2D_gpu_single/moment_struct.hpp"
#include "../PIC2D_gpu_single/particle_struct.hpp"
#include "../PIC2D_gpu_single/moment_calculater.hpp"


class Interface2D
{
private:
    int indexOfInterfaceStartInMHD;
    int indexOfInterfaceStartInPIC;
    int interfaceLength; 
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
    int restartParticlesIndexIon;
    int restartParticlesIndexElectron;
    thrust::device_vector<ReloadParticlesData> reloadParticlesDataIon;
    thrust::device_vector<ReloadParticlesData> reloadParticlesDataElectron;
    thrust::device_vector<Particle> reloadParticlesSourceIon;
    thrust::device_vector<Particle> reloadParticlesSourceElectron;

    thrust::device_vector<int> reloadParticlesIndexIon;
    thrust::device_vector<int> reloadParticlesIndexElectron;
    thrust::host_vector<int> host_reloadParticlesIndexIon;
    thrust::host_vector<int> host_reloadParticlesIndexElectron;

    MomentCalculater momentCalculater;

    thrust::device_vector<MagneticField> B_timeAve;
    thrust::device_vector<ZerothMoment> zerothMomentIon_timeAve;
    thrust::device_vector<ZerothMoment> zerothMomentElectron_timeAve;
    thrust::device_vector<FirstMoment> firstMomentIon_timeAve;
    thrust::device_vector<FirstMoment> firstMomentElectron_timeAve;

    
public:
    Interface2D(
        int indexStartMHD, 
        int indexStartPIC, 
        int interfaceLength
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
        thrust::device_vector<Particle>& particlesElectron, 
        int step
    );


    void sendPICtoMHD(
        const thrust::device_vector<MagneticField>& B, 
        thrust::device_vector<ConservationParameter>& U
    );


    void resetTimeAveParameters();

    void calculateTimeAveParameters(
        const thrust::device_vector<MagneticField>& B, 
        const thrust::device_vector<Particle>& particlesIon, 
        const thrust::device_vector<Particle>& particlesElectron
    );

private:

    void setMoments(
        const thrust::device_vector<Particle>& particlesIon, 
        const thrust::device_vector<Particle>& particlesElectron
    );

    void deleteParticles(
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron, 
        int step
    );
};

