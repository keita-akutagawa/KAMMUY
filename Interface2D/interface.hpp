#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "const.hpp"
#include "reload_particles_data_struct.hpp"
#include "../IdealMHD2D_gpu/const.hpp"
#include "../IdealMHD2D_gpu/conservation_parameter_struct.hpp"
#include "../PIC2D_gpu/const.hpp"
#include "../PIC2D_gpu/field_parameter_struct.hpp"
#include "../PIC2D_gpu/moment_struct.hpp"
#include "../PIC2D_gpu/particle_struct.hpp"
#include "../PIC2D_gpu/moment_calculater.hpp"


class Interface2D
{
private:
    int indexOfInterfaceStartInMHD;
    int indexOfInterfaceStartInPIC;
    int interfaceLength; 
    int indexOfInterfaceEndInMHD;
    int indexOfInterfaceEndInPIC;

    thrust::device_vector<double> interlockingFunctionY;
    thrust::device_vector<double> interlockingFunctionYHalf;

    thrust::host_vector<double> host_interlockingFunctionY;
    thrust::host_vector<double> host_interlockingFunctionYHalf;

    thrust::device_vector<ZerothMoment> zerothMomentIon;
    thrust::device_vector<ZerothMoment> zerothMomentElectron;
    thrust::device_vector<FirstMoment> firstMomentIon;
    thrust::device_vector<FirstMoment> firstMomentElectron;

    unsigned long long restartParticlesIndexIon;
    unsigned long long restartParticlesIndexElectron;

    thrust::device_vector<Particle> reloadParticlesSourceIon;
    thrust::device_vector<Particle> reloadParticlesSourceElectron;

    thrust::device_vector<ReloadParticlesData> reloadParticlesDataIon;
    thrust::device_vector<ReloadParticlesData> reloadParticlesDataElectron;
    thrust::host_vector<ReloadParticlesData> host_reloadParticlesDataIon;
    thrust::host_vector<ReloadParticlesData> host_reloadParticlesDataElectron;

    thrust::device_vector<unsigned long long> reloadParticlesIndexIon;
    thrust::device_vector<unsigned long long> reloadParticlesIndexElectron;
    thrust::host_vector<unsigned long long> host_reloadParticlesIndexIon;
    thrust::host_vector<unsigned long long> host_reloadParticlesIndexElectron;

    thrust::device_vector<MagneticField> B_timeAve;
    thrust::device_vector<ZerothMoment> zerothMomentIon_timeAve;
    thrust::device_vector<ZerothMoment> zerothMomentElectron_timeAve;
    thrust::device_vector<FirstMoment> firstMomentIon_timeAve;
    thrust::device_vector<FirstMoment> firstMomentElectron_timeAve;

    thrust::device_vector<ConservationParameter> USub;
    thrust::device_vector<ConservationParameter> UHalf;

    MomentCalculater momentCalculater;


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

    void deleteParticles(
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron, 
        int step
    );

    void sendMHDtoPIC_particle(
        const thrust::device_vector<ConservationParameter>& U, 
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron, 
        int step
    );


    void sendPICtoMHD(
        const thrust::device_vector<ConservationParameter>& UPast, 
        const thrust::device_vector<ConservationParameter>& UNext
    );


    thrust::device_vector<ConservationParameter>& calculateAndGetSubU(
        const thrust::device_vector<ConservationParameter>& UPast, 
        const thrust::device_vector<ConservationParameter>& UNext, 
        double mixingRatio
    );

    void resetTimeAveParameters();

    void sumUpTimeAveParameters(
        const thrust::device_vector<MagneticField>& B, 
        const thrust::device_vector<Particle>& particlesIon, 
        const thrust::device_vector<Particle>& particlesElectron
    );

    void calculateTimeAveParameters(int substeps);

    thrust::device_vector<ConservationParameter>& getUHalfRef();

private:

    void setMoments(
        const thrust::device_vector<Particle>& particlesIon, 
        const thrust::device_vector<Particle>& particlesElectron
    );

};

