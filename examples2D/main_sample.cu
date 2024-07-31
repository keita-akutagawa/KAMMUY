#include "../IdealMHD2D_gpu/IdealMHD2D.hpp"
#include "../IdealMHD2D_gpu/IdealMHD2D.cu"
#include "../PIC2D_gpu_single/PIC2D.hpp"
#include "../PIC2D_gpu_single/PIC2D.cu"
#include "../Interface2D/interface.hpp"
#include "../Interface2D/interface.cu"
#include "main_sample_const.cu"
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include "main_sample_const.cu"



void initialize()
{

}



int main()
{
    PIC2DConst::initializeDeviceConstants();
    IdealMHD2DConst::initializeDeviceConstants();

    IdealMHD2D idealMHD2D;
    PIC2D pIC2D;
    Interface2D interface2D(
        indexOfInterfaceStartInMHD, 
        indexOfInterfaceStartInPIC, 
        interfaceLength
    );

    size_t free_mem = 0;
    size_t total_mem = 0;
    cudaError_t status = cudaMemGetInfo(&free_mem, &total_mem);

    std::cout << "Free memory: " << free_mem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Total memory: " << total_mem / (1024 * 1024) << " MB" << std::endl;


    for (int step = 0; step < IdealMHD2DConst::totalStep_MHD; step++) {
        
        idealMHD2D.setPastU();
        thrust::device_vector<ConservationParameter>& UPast = idealMHD2D.getURef();
        idealMHD2D.oneStepRK2_predictor();
        thrust::device_vector<ConservationParameter>& UNext = idealMHD2D.getURef();

        int substeps = round(sqrt(PIC2DConst::mRatio_PIC));
        PIC2DConst::dt_PIC = IdealMHD2DConst::dt_MHD / substeps;
        interface2D.resetTimeAveParameters();
        for (int substep = 0; substep < substeps; substep++) {
            pIC2D.oneStepWallXFreeY();

            thrust::device_vector<MagneticField>& B = pIC2D.getBRef();
            thrust::device_vector<ElectricField>& E = pIC2D.getERef();
            thrust::device_vector<CurrentField>& current = pIC2D.getCurrentRef();
            thrust::device_vector<Particle>& particlesIon = pIC2D.getParticlesIonRef();
            thrust::device_vector<Particle>& particlesElectron = pIC2D.getParticlesElectronRef();

            float mixingRatio = (substeps - substep) / substeps;
            thrust::device_vector<ConservationParameter>& USub = interface2D.calculateSubU(UPast, UNext, mixingRatio);

            interface2D.sendMHDtoPIC_magneticField_yDirection(USub, B);
            interface2D.sendMHDtoPIC_electricField_yDirection(USub, E);
            interface2D.sendMHDtoPIC_currentField_yDirection(USub, current);
            interface2D.sendMHDtoPIC_particle(USub, particlesIon, particlesElectron, step * substeps + substep);

            interface2D.sumUpTimeAveParameters(B, partilcesIon, particlesElectron);
        }
        interface2D.calculateTimeAveParameters(substeps);

        interface2D.sendPICtoMHD();
        thrust::device_vector<ConservationParameter>& UHalf = interface2D.getUHalfRef();

        idealMHD2D.oneStepRK2_corrector(UHalf);

    }

    return 0;
}


