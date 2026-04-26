#ifndef RELOADER_H
#define RELOADER_H

#include <thrust/host_vector.h>
#include "../IdealMHD2D/const.hpp"
#include "../Interface2D/const.hpp"
#include "../PIC2D/const.hpp"
#include "../IdealMHD2D/conservation_parameter_struct.hpp"
#include "../PIC2D/particle_struct.hpp"
#include "../PIC2D/field_parameter_struct.hpp"
#include <string>
#include <fstream>


class Reloader 
{
private: 

public: 

    Reloader();

    void reloadPICData(
        thrust::host_vector<Particle>& host_particlesIon, 
        thrust::host_vector<Particle>& host_particlesElectron, 
        thrust::host_vector<MagneticField>& host_B, 
        thrust::host_vector<ElectricField>& host_E, 
        std::string savedDirectoryName, 
        std::string filenameWithoutStep, 
        int step
    );

    void reloadMHDData(
        thrust::host_vector<ConservationParameter>& host_U, 
        std::string savedDirectoryName, 
        std::string filenameWithoutStep, 
        int step
    );

private:

}; 

#endif
