#ifndef INTERFACE_H
#define INTERFACE_H

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cmath>
#include <curand_kernel.h>
#include <random>
#include <algorithm>
#include <thrust/fill.h>
#include <thrust/partition.h>
#include <thrust/transform.h>

#include "const.hpp"
#include "reload_particles_data_struct.hpp"
#include "remove_noise.hpp"
#include "mpi.hpp" 
#include "../IdealMHD2D_multiGPU/const.hpp"
#include "../IdealMHD2D_multiGPU/conservation_parameter_struct.hpp"
#include "../IdealMHD2D_multiGPU/basic_parameter_struct.hpp"
#include "../PIC2D_multiGPU/const.hpp"
#include "../PIC2D_multiGPU/field_parameter_struct.hpp"
#include "../PIC2D_multiGPU/moment_struct.hpp"
#include "../PIC2D_multiGPU/particle_struct.hpp"
#include "../PIC2D_multiGPU/moment_calculator.hpp"
#include "../PIC2D_multiGPU/is_exist_transform.hpp"
#include "../PIC2D_multiGPU/boundary.hpp"
#include "../IdealMHD2D_multiGPU/mpi.hpp"
#include "../PIC2D_multiGPU/mpi.hpp"


class Interface2D
{
private:
    IdealMHD2DMPI::MPIInfo& mPIInfoMHD;
    PIC2DMPI::MPIInfo& mPIInfoPIC;
    Interface2DMPI::MPIInfo& mPIInfoInterface; 
    IdealMHD2DMPI::MPIInfo* device_mPIInfoMHD; 
    PIC2DMPI::MPIInfo* device_mPIInfoPIC; 
    Interface2DMPI::MPIInfo* device_mPIInfoInterface; 

    int indexOfInterfaceStartInMHD;

    thrust::device_vector<double> interlockingFunctionY;

    thrust::device_vector<MagneticField> B_timeAve; 
    thrust::device_vector<ZerothMoment> zerothMomentIon_timeAve;
    thrust::device_vector<ZerothMoment> zerothMomentElectron_timeAve;
    thrust::device_vector<FirstMoment> firstMomentIon_timeAve;
    thrust::device_vector<FirstMoment> firstMomentElectron_timeAve;
    thrust::device_vector<SecondMoment> secondMomentIon_timeAve;
    thrust::device_vector<SecondMoment> secondMomentElectron_timeAve;

    unsigned long long restartParticlesIndexIon;
    unsigned long long restartParticlesIndexElectron;

    thrust::device_vector<Particle> reloadParticlesSourceIon;
    thrust::device_vector<Particle> reloadParticlesSourceElectron;

    thrust::device_vector<ReloadParticlesData> reloadParticlesDataIon;
    thrust::device_vector<ReloadParticlesData> reloadParticlesDataElectron;

    thrust::device_vector<MagneticField> B_PICtoMHD;
    thrust::device_vector<ZerothMoment> zerothMomentIon_PICtoMHD;
    thrust::device_vector<ZerothMoment> zerothMomentElectron_PICtoMHD;
    thrust::device_vector<FirstMoment> firstMomentIon_PICtoMHD;
    thrust::device_vector<FirstMoment> firstMomentElectron_PICtoMHD;
    thrust::device_vector<SecondMoment> secondMomentIon_PICtoMHD;
    thrust::device_vector<SecondMoment> secondMomentElectron_PICtoMHD;

    thrust::device_vector<ConservationParameter> USub;
    thrust::device_vector<ConservationParameter> UHalf;

    MomentCalculator momentCalculator;
    BoundaryPIC boundaryPIC; 
    InterfaceNoiseRemover2D& interfaceNoiseRemover2D;


public:
    Interface2D(
        IdealMHD2DMPI::MPIInfo& mPIInfoMHD, 
        PIC2DMPI::MPIInfo& mPIInfoPIC, 
        Interface2DMPI::MPIInfo& mPIInfoInterface, 
        int indexOfInterfaceStartMHD, 
        thrust::host_vector<double>& host_interlockingFunctionY, 
        InterfaceNoiseRemover2D& interfaceNoiseRemover2D
    );

    void sendMHDtoPIC_magneticField_y(
        const thrust::device_vector<ConservationParameter>& U, 
        thrust::device_vector<MagneticField>& B
    );

    void sendMHDtoPIC_electricField_y(
        const thrust::device_vector<ConservationParameter>& U, 
        thrust::device_vector<ElectricField>& E
    );

    void sendMHDtoPIC_currentField_y(
        const thrust::device_vector<ConservationParameter>& U, 
        thrust::device_vector<CurrentField>& Current
    );

    //CUDAのラムダ制限のためpublicに移動
    void deleteParticlesSpecies(
        thrust::device_vector<Particle>& particlesSpecies, 
        unsigned long long& existNumSpeciesPerProcs, 
        unsigned long long seed
    );

    //CUDAのラムダ制限のためpublicに移動
    void reloadParticlesSpecies(
        thrust::device_vector<Particle>& particlesSpecies, 
        thrust::device_vector<ReloadParticlesData>& reloadParticlesDataSpecies, 
        thrust::device_vector<Particle>& reloadParticlesSourceSpecies, 
        unsigned long long& existNumSpeciesPerProcs, 
        unsigned long long seed 
    );

    void sendMHDtoPIC_particle(
        const thrust::device_vector<ConservationParameter>& U, 
        const thrust::device_vector<ZerothMoment>& zerothMomentIon, 
        const thrust::device_vector<ZerothMoment>& zerothMomentElectron, 
        const thrust::device_vector<FirstMoment>& firstMomentIon, 
        const thrust::device_vector<FirstMoment>& firstMomentElectron, 
        const thrust::device_vector<SecondMoment>& secondMomentIon, 
        const thrust::device_vector<SecondMoment>& secondMomentElectron, 
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron, 
        unsigned long long seed
    );


    void sendPICtoMHD(
        thrust::device_vector<ConservationParameter>& U
    );


    thrust::device_vector<ConservationParameter>& calculateAndGetSubU(
        const thrust::device_vector<ConservationParameter>& UPast, 
        const thrust::device_vector<ConservationParameter>& UNext, 
        double mixingRatio
    );

    void resetTimeAveragedPICParameters();

    void sumUpTimeAveragedPICParameters(
        const thrust::device_vector<MagneticField>& B, 
        const thrust::device_vector<ZerothMoment>& zerothMomentIon, 
        const thrust::device_vector<ZerothMoment>& zerothMomentElectron, 
        const thrust::device_vector<FirstMoment>& firstMomentIon, 
        const thrust::device_vector<FirstMoment>& firstMomentElectron, 
        const thrust::device_vector<SecondMoment>& secondMomentIon, 
        const thrust::device_vector<SecondMoment>& secondMomentElectron
    );

    void calculateTimeAveragedPICParameters(
        int count
    );

    void setParametersForPICtoMHD();

    void calculateUHalf(
        const thrust::device_vector<ConservationParameter>& UPast, 
        const thrust::device_vector<ConservationParameter>& UNext 
    ); 

    thrust::device_vector<ConservationParameter>& getUHalfRef();

private:

};

#endif

