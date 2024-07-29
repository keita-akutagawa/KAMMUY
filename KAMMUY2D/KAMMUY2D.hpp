#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "../IdealMHD2D_gpu/IdealMHD2D.hpp"
#include "../PIC2D_gpu_single/pic2D.hpp"
#include "../Interface2D/interface.hpp"


class KAMMUY2D
{
private:
    IdealMHD2D idealMHD2D;
    PIC2D pic2D;
    Interface2D interface2D;

    thrust::device_vector<ConservationParameter> UPast;

public:
    KAMMUY2D();

    virtual void initialize();

    void oneStep();

    void saveMHD_U(
        std::string directoryname, 
        std::string filenameWithoutStep, 
        int step
    );

    void savePIC_Fields(
        std::string directoryname, 
        std::string filenameWithoutStep, 
        int step
    );

    void savePIC_ZerothMoments(
        std::string directoryname, 
        std::string filenameWithoutStep, 
        int step
    );

    void savePIC_FirstMoments(
        std::string directoryname, 
        std::string filenameWithoutStep, 
        int step
    );

    void savePIC_SecondMoments(
        std::string directoryname, 
        std::string filenameWithoutStep, 
        int step
    );

    void savePIC_Particle(
        std::string directoryname, 
        std::string filenameWithoutStep, 
        int step
    );

private:

};
