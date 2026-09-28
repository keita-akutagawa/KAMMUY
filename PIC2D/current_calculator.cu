#include "current_calculator.hpp"


CurrentCalculator::CurrentCalculator(
    PICConstParameter& pICConstParameter, 
    const PICGridParameter& pICGridParameter
)
    : NX(pICGridParameter.NX), 
      NY(pICGridParameter.NY), 
      DX(pICGridParameter.DX), 
      DY(pICGridParameter.DY), 
      
      pICConstParameter(pICConstParameter), 
      pICGridParameter(pICGridParameter), 
    
      momentCalculator(
        pICConstParameter, pICGridParameter
      )
{
}


__global__ void calculateCurrent_kernel(
    const FirstMoment* firstMomentIon, 
    const FirstMoment* firstMomentElectron, 
    const PICUnsignedInt NX, PICUnsignedInt NY, 
    const PICFloat Q_ION, PICFloat Q_ELECTRON, 
    CurrentField* current
)
{
    PICUnsignedInt i = blockIdx.x * blockDim.x + threadIdx.x;
    PICUnsignedInt j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < NX && j < NY) {
        PICUnsignedLongLong index = getIndex<PICUnsignedLongLong>(i, j, NX, NY); 

        current[index].jX = Q_ION * firstMomentIon[index].x
                          + Q_ELECTRON * firstMomentElectron[index].x; 
        current[index].jY = Q_ION * firstMomentIon[index].y
                          + Q_ELECTRON * firstMomentElectron[index].y; 
        current[index].jZ = Q_ION * firstMomentIon[index].z
                          + Q_ELECTRON * firstMomentElectron[index].z; 

    }
}


void CurrentCalculator::calculateCurrent(
    const thrust::device_vector<Particle>& particlesIon, 
    const thrust::device_vector<Particle>& particlesElectron, 
    thrust::device_vector<FirstMoment>& firstMomentIon, 
    thrust::device_vector<FirstMoment>& firstMomentElectron, 
    thrust::device_vector<CurrentField>& current
)
{
    momentCalculator.calculateFirstMoment(
        particlesIon, pICConstParameter.EXIST_NUM_ION, firstMomentIon
    ); 
    momentCalculator.calculateFirstMoment(
        particlesElectron, pICConstParameter.EXIST_NUM_ELECTRON, firstMomentElectron
    );

    
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (NY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    calculateCurrent_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(firstMomentIon.data()), 
        thrust::raw_pointer_cast(firstMomentElectron.data()), 
        NX, NY, 
        pICConstParameter.Q_ION, pICConstParameter.Q_ELECTRON, 
        thrust::raw_pointer_cast(current.data())
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed at calculateCurrent_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed at calculateCurrent_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
}

