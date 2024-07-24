#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "const.hpp"
#include "../lib_IdealMHD2D_gpu/const.hpp"
#include "../lib_IdealMHD2D_gpu/conservation_parameter_struct.hpp"
#include "../lib_PIC2D_gpu_single/const.hpp"
#include "../lib_PIC2D_gpu_single/field_parameter_struct.hpp"
#include "../lib_PIC2D_gpu_single/particle_struct.hpp"


class Interface2D
{
private:
    thrust::device_vector<float> interlockingFunctionX;
    thrust::device_vector<float> interlockingFunctionY;

    thrust::host_vector<float> host_interlockingFunctionX;
    thrust::host_vector<float> host_interlockingFunctionY;

    
public:
    Interface2D();

    void getQuantityMHDtoPIC();

    void sendMHDtoPIC_MagneticField(
        thrust::device_vector<ConservationParameter>& U, 
        thrust::device_vector<MagneticField>& B
    );

    void sendMHDtoPIC_ElectricField(
        thrust::device_vector<ConservationParameter>& U, 
        thrust::device_vector<ElectricField>& E
    );

    void sendMHDtoPIC_CurrentField(
        thrust::device_vector<ConservationParameter>& U, 
        thrust::device_vector<CurrentField>& Current
    );

    void sendMHDtoPIC_Particle(
        thrust::device_vector<ConservationParameter>& U, 
        thrust::device_vector<Particle>& particlesIon, 
        thrust::device_vector<Particle>& particlesElectron
    );

    void reloadParticles(
        thrust::device_vector<Particle>& particlesSpecies
    );


    void getQuantityPICtoMHD();

    void sendPICtoMHD();

private:

};

