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
#include "../PIC2D_multiGPU/const.hpp"
#include "../PIC2D_multiGPU/field_parameter_struct.hpp"
#include "../PIC2D_multiGPU/moment_struct.hpp"
#include "../PIC2D_multiGPU/particle_struct.hpp"
#include "../PIC2D_multiGPU/moment_calculater.hpp"
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
    int indexOfInterfaceStartInPIC;

    int localSizeXPIC; 
    int localSizeYPIC; 
    int localSizeXMHD; 
    int localSizeYMHD; 
    int localSizeXInterface; 
    int localSizeYInterface; 

    thrust::device_vector<double> interlockingFunctionY;
    thrust::device_vector<double> interlockingFunctionYHalf;

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

    thrust::device_vector<MagneticField> B_timeAve;
    thrust::device_vector<ZerothMoment> zerothMomentIon_timeAve;
    thrust::device_vector<ZerothMoment> zerothMomentElectron_timeAve;
    thrust::device_vector<FirstMoment> firstMomentIon_timeAve;
    thrust::device_vector<FirstMoment> firstMomentElectron_timeAve;

    thrust::device_vector<ConservationParameter> USub;
    thrust::device_vector<ConservationParameter> UHalf;

    MomentCalculater momentCalculater;
    BoundaryPIC boundaryPIC; 
    InterfaceNoiseRemover2D& interfaceNoiseRemover2D;


public:
    Interface2D(
        IdealMHD2DMPI::MPIInfo& mPIInfoMHD, 
        PIC2DMPI::MPIInfo& mPIInfoPIC, 
        Interface2DMPI::MPIInfo& mPIInfoInterface, 
        int indexOfInterfaceStartMHD, 
        int indexOfInterfaceStartPIC, 
        thrust::host_vector<double>& host_interlockingFunctionY, 
        thrust::host_vector<double>& host_interlockingFunctionYHalf, 
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
        int seed
    );

    //CUDAのラムダ制限のためpublicに移動
    void reloadParticlesSpecies(
        thrust::device_vector<Particle>& particlesSpecies, 
        thrust::device_vector<ReloadParticlesData>& reloadParticlesDataSpecies, 
        thrust::device_vector<Particle>& reloadParticlesSourceSpecies, 
        unsigned long long& existNumSpeciesPerProcs, 
        int seed 
    );

    void sendMHDtoPIC_particle(
        const thrust::device_vector<ConservationParameter>& U, 
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron, 
        int seed
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

#endif

