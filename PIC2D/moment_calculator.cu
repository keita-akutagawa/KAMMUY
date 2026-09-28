#include "moment_calculator.hpp"


MomentCalculator::MomentCalculator(
    PICConstParameter& pICConstParameter, 
    const PICGridParameter& pICGridParameter
)
    : NX(pICGridParameter.NX), 
      NY(pICGridParameter.NY), 
      DX(pICGridParameter.DX), 
      DY(pICGridParameter.DY), 
      
      pICConstParameter(pICConstParameter), 
      pICGridParameter(pICGridParameter)
{
}


void MomentCalculator::resetZerothMoment(
    thrust::device_vector<ZerothMoment>& zerothMoment
)
{
    thrust::fill(
        zerothMoment.begin(), 
        zerothMoment.end(), 
        ZerothMoment()
    );
}

void MomentCalculator::resetFirstMoment(
    thrust::device_vector<FirstMoment>& firstMoment
)
{
    thrust::fill(
        firstMoment.begin(), 
        firstMoment.end(), 
        FirstMoment()
    );
}

void MomentCalculator::resetSecondMoment(
    thrust::device_vector<SecondMoment>& secondMoment
)
{
    thrust::fill(
        secondMoment.begin(), 
        secondMoment.end(), 
        SecondMoment()
    );
}

//////////

__global__ void calculateZerothMoment_kernel(
    const PICUnsignedInt NX, const PICUnsignedInt NY, 
    const PICFloat DX, const PICFloat DY, 
    const PICFloat XMIN, const PICFloat YMIN, 
    const Particle* particles, 
    const PICUnsignedLongLong EXIST_NUM, 
    ZerothMoment* zerothMoment
)
{
    PICUnsignedLongLong i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < EXIST_NUM) {
        
        PICFloat xOverDx = (particles[i].x - XMIN) / DX;
        PICFloat yOverDy = (particles[i].y - YMIN) / DY;

        PICUnsignedInt xIndex1 = floor(xOverDx);
        PICUnsignedInt xIndex2 = xIndex1 + 1;
        xIndex2 = (xIndex2 == NX) ? 0 : xIndex2;
        PICUnsignedInt yIndex1 = floor(yOverDy);
        PICUnsignedInt yIndex2 = yIndex1 + 1;
        yIndex2 = (yIndex2 == NY) ? 0 : yIndex2;

        if (xIndex1 >= NX) printf("x = %f, index = %u, ERROR\n", particles[i].x, xIndex1); 
        if (yIndex1 >= NY) printf("y = %f, index = %u, ERROR\n", particles[i].y, yIndex1);

        PICFloat cx1 = xOverDx - xIndex1;
        PICFloat cx2 = 1.0 - cx1;
        PICFloat cy1 = yOverDy - yIndex1;
        PICFloat cy2 = 1.0 - cy1;

        PICUnsignedLongLong index11 = getIndex<PICUnsignedLongLong>(xIndex1, yIndex1, NX, NY); 
        PICUnsignedLongLong index12 = getIndex<PICUnsignedLongLong>(xIndex1, yIndex2, NX, NY); 
        PICUnsignedLongLong index21 = getIndex<PICUnsignedLongLong>(xIndex2, yIndex1, NX, NY); 
        PICUnsignedLongLong index22 = getIndex<PICUnsignedLongLong>(xIndex2, yIndex2, NX, NY); 
        atomicAdd(&(zerothMoment[index11].n), cx2 * cy2);
        atomicAdd(&(zerothMoment[index12].n), cx2 * cy1);
        atomicAdd(&(zerothMoment[index21].n), cx1 * cy2);
        atomicAdd(&(zerothMoment[index22].n), cx1 * cy1);
    }
};


void MomentCalculator::calculateZerothMoment(
    const thrust::device_vector<Particle>& particles, 
    unsigned long long EXIST_NUM, 
    thrust::device_vector<ZerothMoment>& zerothMoment
)
{
    resetZerothMoment(zerothMoment);

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((EXIST_NUM + threadsPerBlock.x - 1) / threadsPerBlock.x);

    calculateZerothMoment_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        NX, NY, 
        DX, DY, 
        pICGridParameter.XMIN, pICGridParameter.YMIN, 
        thrust::raw_pointer_cast(particles.data()), 
        EXIST_NUM, 
        thrust::raw_pointer_cast(zerothMoment.data())
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed at calculateZerothMoment_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed at calculateZerothMoment_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
}



__global__ void calculateFirstMoment_kernel(
    const PICUnsignedInt NX, const PICUnsignedInt NY, 
    const PICFloat DX, const PICFloat DY,
    const PICFloat XMIN, const PICFloat YMIN, 
    const Particle* particles, 
    const PICUnsignedLongLong EXIST_NUM, 
    FirstMoment* firstMoment
)
{
    PICUnsignedLongLong i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < EXIST_NUM) {

        PICFloat xOverDx = (particles[i].x - XMIN) / DX;
        PICFloat yOverDy = (particles[i].y - YMIN) / DY;

        PICUnsignedInt xIndex1 = floor(xOverDx);
        PICUnsignedInt xIndex2 = xIndex1 + 1;
        xIndex2 = (xIndex2 == NX) ? 0 : xIndex2;
        PICUnsignedInt yIndex1 = floor(yOverDy);
        PICUnsignedInt yIndex2 = yIndex1 + 1;
        yIndex2 = (yIndex2 == NY) ? 0 : yIndex2;

        if (xIndex1 >= NX) printf("x = %f, index = %u, ERROR\n", particles[i].x, xIndex1); 
        if (yIndex1 >= NY) printf("y = %f, index = %u, ERROR\n", particles[i].y, yIndex1);

        PICFloat cx1 = xOverDx - xIndex1;
        PICFloat cx2 = 1.0 - cx1;
        PICFloat cy1 = yOverDy - yIndex1;
        PICFloat cy2 = 1.0 - cy1;

        PICFloat vx = particles[i].ux / particles[i].gamma;
        PICFloat vy = particles[i].uy / particles[i].gamma;
        PICFloat vz = particles[i].uz / particles[i].gamma;

        PICUnsignedLongLong index11 = getIndex<PICUnsignedLongLong>(xIndex1, yIndex1, NX, NY); 
        PICUnsignedLongLong index12 = getIndex<PICUnsignedLongLong>(xIndex1, yIndex2, NX, NY); 
        PICUnsignedLongLong index21 = getIndex<PICUnsignedLongLong>(xIndex2, yIndex1, NX, NY); 
        PICUnsignedLongLong index22 = getIndex<PICUnsignedLongLong>(xIndex2, yIndex2, NX, NY); 

        atomicAdd(&(firstMoment[index11].x), vx * cx2 * cy2);
        atomicAdd(&(firstMoment[index12].x), vx * cx2 * cy1);
        atomicAdd(&(firstMoment[index21].x), vx * cx1 * cy2);
        atomicAdd(&(firstMoment[index22].x), vx * cx1 * cy1);

        atomicAdd(&(firstMoment[index11].y), vy * cx2 * cy2);
        atomicAdd(&(firstMoment[index12].y), vy * cx2 * cy1);
        atomicAdd(&(firstMoment[index21].y), vy * cx1 * cy2);
        atomicAdd(&(firstMoment[index22].y), vy * cx1 * cy1);

        atomicAdd(&(firstMoment[index11].z), vz * cx2 * cy2);
        atomicAdd(&(firstMoment[index12].z), vz * cx2 * cy1);
        atomicAdd(&(firstMoment[index21].z), vz * cx1 * cy2);
        atomicAdd(&(firstMoment[index22].z), vz * cx1 * cy1);
    }
};


void MomentCalculator::calculateFirstMoment(
    const thrust::device_vector<Particle>& particles, 
    unsigned long long EXIST_NUM, 
    thrust::device_vector<FirstMoment>& firstMoment
)
{
    resetFirstMoment(firstMoment);

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((EXIST_NUM + threadsPerBlock.x - 1) / threadsPerBlock.x);

    calculateFirstMoment_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        NX, NY, 
        DX, DY, 
        pICGridParameter.XMIN, pICGridParameter.YMIN, 
        thrust::raw_pointer_cast(particles.data()), 
        EXIST_NUM, 
        thrust::raw_pointer_cast(firstMoment.data())
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed at calculateFirstMoment_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed at calculateFirstMoment_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
}


__global__ void calculateSecondMoment_kernel(
    const PICUnsignedInt NX, const PICUnsignedInt NY, 
    const PICFloat DX, const PICFloat DY,
    const PICFloat XMIN, const PICFloat YMIN, 
    const Particle* particles, 
    const PICUnsignedLongLong EXIST_NUM, 
    SecondMoment* secondMoment
)
{
    PICUnsignedLongLong i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < EXIST_NUM) {
    
        PICFloat xOverDx = (particles[i].x - XMIN) / DX;
        PICFloat yOverDy = (particles[i].y - YMIN) / DY;

        PICUnsignedInt xIndex1 = floor(xOverDx);
        PICUnsignedInt xIndex2 = xIndex1 + 1;
        xIndex2 = (xIndex2 == NX) ? 0 : xIndex2;
        PICUnsignedInt yIndex1 = floor(yOverDy);
        PICUnsignedInt yIndex2 = yIndex1 + 1;
        yIndex2 = (yIndex2 == NY) ? 0 : yIndex2;

        if (xIndex1 >= NX) printf("x = %f, index = %u, ERROR\n", particles[i].x, xIndex1); 
        if (yIndex1 >= NY) printf("y = %f, index = %u, ERROR\n", particles[i].y, yIndex1);

        PICFloat cx1 = xOverDx - xIndex1;
        PICFloat cx2 = 1.0 - cx1;
        PICFloat cy1 = yOverDy - yIndex1;
        PICFloat cy2 = 1.0 - cy1;

        PICFloat vx = particles[i].ux / particles[i].gamma;
        PICFloat vy = particles[i].uy / particles[i].gamma;
        PICFloat vz = particles[i].uz / particles[i].gamma;

        PICUnsignedLongLong index11 = getIndex<PICUnsignedLongLong>(xIndex1, yIndex1, NX, NY); 
        PICUnsignedLongLong index12 = getIndex<PICUnsignedLongLong>(xIndex1, yIndex2, NX, NY); 
        PICUnsignedLongLong index21 = getIndex<PICUnsignedLongLong>(xIndex2, yIndex1, NX, NY); 
        PICUnsignedLongLong index22 = getIndex<PICUnsignedLongLong>(xIndex2, yIndex2, NX, NY); 

        atomicAdd(&(secondMoment[index11].xx), vx * vx * cx2 * cy2);
        atomicAdd(&(secondMoment[index12].xx), vx * vx * cx2 * cy1);
        atomicAdd(&(secondMoment[index21].xx), vx * vx * cx1 * cy2);
        atomicAdd(&(secondMoment[index22].xx), vx * vx * cx1 * cy1);

        atomicAdd(&(secondMoment[index11].yy), vy * vy * cx2 * cy2);
        atomicAdd(&(secondMoment[index12].yy), vy * vy * cx2 * cy1);
        atomicAdd(&(secondMoment[index21].yy), vy * vy * cx1 * cy2);
        atomicAdd(&(secondMoment[index22].yy), vy * vy * cx1 * cy1);

        atomicAdd(&(secondMoment[index11].zz), vz * vz * cx2 * cy2);
        atomicAdd(&(secondMoment[index12].zz), vz * vz * cx2 * cy1);
        atomicAdd(&(secondMoment[index21].zz), vz * vz * cx1 * cy2);
        atomicAdd(&(secondMoment[index22].zz), vz * vz * cx1 * cy1);

        atomicAdd(&(secondMoment[index11].xy), vx * vy * cx2 * cy2);
        atomicAdd(&(secondMoment[index12].xy), vx * vy * cx2 * cy1);
        atomicAdd(&(secondMoment[index21].xy), vx * vy * cx1 * cy2);
        atomicAdd(&(secondMoment[index22].xy), vx * vy * cx1 * cy1);

        atomicAdd(&(secondMoment[index11].xz), vx * vz * cx2 * cy2);
        atomicAdd(&(secondMoment[index12].xz), vx * vz * cx2 * cy1);
        atomicAdd(&(secondMoment[index21].xz), vx * vz * cx1 * cy2);
        atomicAdd(&(secondMoment[index22].xz), vx * vz * cx1 * cy1);

        atomicAdd(&(secondMoment[index11].yz), vy * vz * cx2 * cy2);
        atomicAdd(&(secondMoment[index12].yz), vy * vz * cx2 * cy1);
        atomicAdd(&(secondMoment[index21].yz), vy * vz * cx1 * cy2);
        atomicAdd(&(secondMoment[index22].yz), vy * vz * cx1 * cy1);
    }
};


void MomentCalculator::calculateSecondMoment(
    const thrust::device_vector<Particle>& particles, 
    unsigned long long EXIST_NUM, 
    thrust::device_vector<SecondMoment>& secondMoment
)
{
    resetSecondMoment(secondMoment);

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((EXIST_NUM + threadsPerBlock.x - 1) / threadsPerBlock.x);

    calculateSecondMoment_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        NX, NY, 
        DX, DY, 
        pICGridParameter.XMIN, pICGridParameter.YMIN, 
        thrust::raw_pointer_cast(particles.data()), 
        EXIST_NUM, 
        thrust::raw_pointer_cast(secondMoment.data())
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed at calculateSecondMoment_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed at calculateSecondMoment_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
}



