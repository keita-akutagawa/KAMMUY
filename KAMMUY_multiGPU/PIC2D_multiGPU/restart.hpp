#include <thrust/device_vector.h>
#include <string>
#include <fstream>
#include "const.hpp"
#include "field_parameter_struct.hpp" 
#include "particle_struct.hpp" 
#include "mpi.hpp"


class RestartPIC
{
private: 
    PIC2DMPI::MPIInfo& mPIInfo;

public: 
    RestartPIC(
        PIC2DMPI::MPIInfo& mPIInfo
    ); 

    void loadFields(
        thrust::host_vector<MagneticField> host_B, 
        thrust::host_vector<ElectricField> host_E, 
        thrust::device_vector<MagneticField> B, 
        thrust::device_vector<ElectricField> E, 
        std::string directoryname, 
        std::string filenameWithoutStep, 
        int step
    );

    void loadParticles(
        thrust::host_vector<Particle> host_particlesIon, 
        thrust::host_vector<Particle> host_particlesElectron, 
        thrust::device_vector<Particle> particlesIon, 
        thrust::device_vector<Particle> particlesElectron, 
        std::string directoryname, 
        std::string filenameWithoutStep, 
        int step
    );

private: 

};

