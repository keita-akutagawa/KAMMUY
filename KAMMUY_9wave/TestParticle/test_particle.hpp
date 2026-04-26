#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "../PIC2D_multiGPU/const.hpp"
#include "../IdealMHD2D_multiGPU/const.hpp"
#include "../Interface2D_multiGPU/const.hpp"
#include "../PIC2D_multiGPU/particle_struct.hpp"
#include "../PIC2D_multiGPU/field_parameter_struct.hpp"
#include "../IdealMHD2D_multiGPU/conservation_parameter_struct.hpp"
#include "const.hpp"


class TestParticle
{
private:

    thrust::device_vector<Particle> particles;
    thrust::device_vector<MagneticField> B;
    thrust::device_vector<ElectricField> E;

    thrust::host_vector<Particle> host_particles;

public:
    TestParticle();
    
    virtual void initialize(
        std::string directoryName, 
        std::string filenameWithoutStep, 
        int step 
    );

    void oneStep();

    void pushVelocity(const double dt);

    void pushPosition(const double dt);

    void saveParticle(
        std::string directoryName, 
        std::string filenameWithoutStep, 
        int step
    );

    void saveField(
        std::string directoryName, 
        std::string filenameWithoutStep, 
        int step
    );

private:

};


