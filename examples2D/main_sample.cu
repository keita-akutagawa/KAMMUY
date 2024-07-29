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

    size_t free_mem = 0;
    size_t total_mem = 0;
    cudaError_t status = cudaMemGetInfo(&free_mem, &total_mem);

    std::cout << "Free memory: " << free_mem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Total memory: " << total_mem / (1024 * 1024) << " MB" << std::endl;


    for (int step = 0; step < IdealMHD2DConst::totalStep; step++) {
        
    }

    return 0;
}


