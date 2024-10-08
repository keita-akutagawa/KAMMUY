#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>
#include <string>
#include "const.hpp"
#include "initialize_particle.hpp"
#include "particle_push.hpp"
#include "field_solver.hpp"
#include "current_calculator.hpp"
#include "boundary.hpp"
#include "sort_particle.hpp"
#include "moment_calculater.hpp"
#include "filter.hpp"
#include "particle_struct.hpp"
#include "field_parameter_struct.hpp"
#include "moment_struct.hpp"
#include "../IdealMHD2D_gpu/conservation_parameter_struct.hpp"
#include "../Interface2D/interface.hpp"
#include "../Interface2D/remove_noise.hpp"


class PIC2D
{
private:
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

    InitializeParticle initializeParticle;
    ParticlePush particlePush;
    FieldSolver fieldSolver;
    CurrentCalculator currentCalculator;
    BoundaryPIC boundaryPIC;
    ParticleSorter particleSorter;
    MomentCalculater momentCalculater;
    Filter filter;

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


public:
    PIC2D();
    
    virtual void initialize();

    void oneStepPeriodicXFreeY(
        thrust::device_vector<ConservationParameter>& UPast_Lower, 
        thrust::device_vector<ConservationParameter>& UPast_Upper, 
        thrust::device_vector<ConservationParameter>& UNext_Lower, 
        thrust::device_vector<ConservationParameter>& UNext_Upper, 
        Interface2D& interface2D_Lower, 
        Interface2D& interface2D_Upper, 
        InterfaceNoiseRemover2D& interfaceNoiseRemover2D_Lower, 
        InterfaceNoiseRemover2D& interfaceNoiseRemover2D_Upper, 
        int step, int substep, int totalSubstep
    );

    void oneStepFreeXFreeY(
        thrust::device_vector<ConservationParameter>& UPast_Lower, 
        thrust::device_vector<ConservationParameter>& UPast_Upper, 
        thrust::device_vector<ConservationParameter>& UNext_Lower, 
        thrust::device_vector<ConservationParameter>& UNext_Upper, 
        Interface2D& interface2D_Lower, 
        Interface2D& interface2D_Upper, 
        InterfaceNoiseRemover2D& interfaceNoiseRemover2D_Lower, 
        InterfaceNoiseRemover2D& interfaceNoiseRemover2D_Upper, 
        int step, int substep, int totalSubstep
    );

    void sortParticle();

    void saveFields(
        std::string directoryname, 
        std::string filenameWithoutStep, 
        int step
    );

    void saveFullMoments(
        std::string directoryname, 
        std::string filenameWithoutStep, 
        int step
    );

    void saveZerothMoments(
        std::string directoryname, 
        std::string filenameWithoutStep, 
        int step
    );

    void saveFirstMoments(
        std::string directoryname, 
        std::string filenameWithoutStep, 
        int step
    );

    void saveSecondMoments(
        std::string directoryname, 
        std::string filenameWithoutStep, 
        int step
    );

    void saveParticle(
        std::string directoryname, 
        std::string filenameWithoutStep, 
        int step
    );

    thrust::device_vector<MagneticField>& getBRef();

    thrust::device_vector<ElectricField>& getERef();

    thrust::device_vector<CurrentField>& getCurrentRef();

    thrust::device_vector<Particle>& getParticlesIonRef();

    thrust::device_vector<Particle>& getParticlesElectronRef();

private:
    void calculateFullMoments();

    void calculateZerothMoments();

    void calculateFirstMoments();

    void calculateSecondMoments();

};


