#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "const.hpp"


class Interface2D
{
private:
    thrust::device_vector<float> interlockingFunction;

    thrust::host_vector<float> host_interlockingFunction;

    
public:
    Interface2D();

    void getQuantityMHDtoPIC();

    void sendMHDtoPIC_MagneticField();

    void sendMHDtoPIC_ElectricField();

    void sendMHDtoPIC_CurrentField();

    void sendMHDtoPIC_Particle();

    void reloadParticles();


    void getQuantityPICtoMHD();

    void sendPICtoMHD();

private:

}

