#include "../IdealMHD2D_gpu/IdealMHD2D.hpp"
#include "../PIC2D_gpu_single/PIC2D.hpp"
#include "../Interface2D/interface.hpp"
#include "main_sample_const.cu"
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>


std::string directoryname = "results_sample";
std::string filenameWithoutStep = "sample";
std::ofstream logfile("results_sample/log_sample.txt");


void PIC2D::initialize()
{

}


void IdealMHD2D::initializeU()
{

}



int main()
{
    PIC2DConst::initializeDeviceConstants();
    IdealMHD2DConst::initializeDeviceConstants();

    PIC2D pIC2D;
    IdealMHD2D idealMHD2D;
    Interface2D interface2D(
        Interface2DConst::indexOfInterfaceStartInMHD, 
        Interface2DConst::indexOfInterfaceStartInPIC
    );

    size_t free_mem = 0;
    size_t total_mem = 0;
    cudaError_t status = cudaMemGetInfo(&free_mem, &total_mem);

    std::cout << "Free memory: " << free_mem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Total memory: " << total_mem / (1024 * 1024) << " MB" << std::endl;


    pIC2D.initialize();
    idealMHD2D.initializeU();

    for (int step = 0; step < IdealMHD2DConst::totalStep; step++) {
        
    }

    return 0;
}


