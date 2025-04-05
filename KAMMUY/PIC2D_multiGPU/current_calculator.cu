#include <cmath>
#include "current_calculator.hpp"
#include <thrust/fill.h>


CurrentCalculator::CurrentCalculator(PIC2DMPI::MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo), 
      momentCalculator(mPIInfo)
{
}


//void CurrentCalculator::resetCurrent(
//    thrust::device_vector<CurrentField>& current
//)
//{
//    thrust::fill(current.begin(), current.end(), CurrentField());
//}


__global__ void calculateCurrent_kernel(
    CurrentField* current, 
    const FirstMoment* firstMomentIon, 
    const FirstMoment* firstMomentElectron, 
    const int localSizeX
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localSizeX && j < PIC2DConst::device_ny) {
        int index = j + i * PIC2DConst::device_ny;

        current[index].jX = PIC2DConst::device_qIon * firstMomentIon[index].x
                          + PIC2DConst::device_qElectron * firstMomentElectron[index].x; 
        current[index].jY = PIC2DConst::device_qIon * firstMomentIon[index].y
                          + PIC2DConst::device_qElectron * firstMomentElectron[index].y; 
        current[index].jZ = PIC2DConst::device_qIon * firstMomentIon[index].z
                          + PIC2DConst::device_qElectron * firstMomentElectron[index].z; 

    }
}


void CurrentCalculator::calculateCurrent(
    thrust::device_vector<CurrentField>& current, 
    thrust::device_vector<FirstMoment>& firstMomentIon, 
    thrust::device_vector<FirstMoment>& firstMomentElectron, 
    const thrust::device_vector<Particle>& particlesIon, 
    const thrust::device_vector<Particle>& particlesElectron
)
{
    momentCalculator.calculateFirstMomentOfOneSpecies(
        firstMomentIon, particlesIon, mPIInfo.existNumIonPerProcs
    ); 
    momentCalculator.calculateFirstMomentOfOneSpecies(
        firstMomentElectron, particlesElectron, mPIInfo.existNumElectronPerProcs
    );

    
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (PIC2DConst::ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    calculateCurrent_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(current.data()), 
        thrust::raw_pointer_cast(firstMomentIon.data()), 
        thrust::raw_pointer_cast(firstMomentElectron.data()), 
        mPIInfo.localSizeX
    );
}

