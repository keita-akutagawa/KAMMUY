#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "const.hpp"


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

    void sendMHDtoPIC_MagneticField();

    void sendMHDtoPIC_ElectricField();

    void sendMHDtoPIC_CurrentField();

    void sendMHDtoPIC_Particle();

    void reloadParticles();


    void getQuantityPICtoMHD();

    void sendPICtoMHD();

private:

};

