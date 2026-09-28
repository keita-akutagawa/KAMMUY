#include "interface2D.hpp"


__global__ void calculateTimeInterpolatedU_kernel(
    const MHDUnsignedInt NX_MHD, const MHDUnsignedInt NY_MHD, 
    const MHDValue* UPast, const MHDValue* UNext, 
    const MHDFloat mixingRatio, 
    MHDValue* timeInterpolatedU  
)
{
    MHDUnsignedInt i = blockIdx.x * blockDim.x + threadIdx.x;
    MHDUnsignedInt j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < NX_MHD && j < NY_MHD) { 
        MHDUnsignedLongLong index = getIndex<MHDUnsignedLongLong>(i, j, NX_MHD, NY_MHD); 

        timeInterpolatedU[index] = UPast[index] * (1 - mixingRatio) + UNext[index] * mixingRatio; 
    }
}

void Interface2D::calculateTimeInterpolatedU(
    const thrust::device_vector<MHDValue>& UPast, 
    const thrust::device_vector<MHDValue>& UNext, 
    const PICUnsignedInt substep, 
    const PICUnsignedInt totalSubstep
)
{
    const MHDFloat mixingRatio = static_cast<MHDFloat>(substep) / totalSubstep;
    
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((NX_MHD + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (NY_MHD + threadsPerBlock.y - 1) / threadsPerBlock.y);

    calculateTimeInterpolatedU_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        NX_MHD, NY_MHD, 
        thrust::raw_pointer_cast(UPast.data()), 
        thrust::raw_pointer_cast(UNext.data()), 
        mixingRatio, 
        thrust::raw_pointer_cast(timeInterpolatedU.data())
    );
    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess) {
        printf("Kernel launch failed at calculateTimeInterpolatedU_kernel: %s\n", cudaGetErrorString(err1));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err1 = cudaDeviceSynchronize();
    if (err1 != cudaSuccess) {
        printf("Kernel execution failed at calculateTimeInterpolatedU_kernel: %s\n", cudaGetErrorString(err1));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
}


thrust::device_vector<MHDValue>& Interface2D::getTimeInterpolatedURef()
{
    return timeInterpolatedU; 
}


struct MagneticField_MHD {
    InterfaceFloat bX;
    InterfaceFloat bY;
    InterfaceFloat bZ;
};

__device__ MagneticField_MHD getMagneticField_MHD(
    const MHDValue* U,  
    const MHDUnsignedLongLong indexMHD
)
{
    MagneticField_MHD B_MHD;
    
    B_MHD.bX = U[indexMHD].bX; 
    B_MHD.bY = U[indexMHD].bY; 
    B_MHD.bZ = U[indexMHD].bZ;
    
    return B_MHD; 
}


__global__ void sendMHDtoPIC_B_kernel(
    const PICUnsignedInt NX_PIC, const PICUnsignedInt NY_PIC, 
    const PICUnsignedInt BUFFER_PIC, 
    const MHDUnsignedInt NX_MHD, const MHDUnsignedInt NY_MHD, 
    const MHDValue* U, 
    const InterfaceUnsignedInt START_INDEX_IN_MHD_X, 
    const InterfaceUnsignedInt START_INDEX_IN_MHD_Y, 
    const InterfaceUnsignedInt GRID_SIZE_RATIO, 
    const InterfaceFloat* interlockingFunction, 
    MagneticField* B
)
{
    PICUnsignedInt i = blockIdx.x * blockDim.x + threadIdx.x;
    PICUnsignedInt j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < NX_PIC && j < NY_PIC) { 
        InterfaceInt INT_CASTED_GRID_SIZE_RATIO = static_cast<InterfaceInt>(GRID_SIZE_RATIO);
        InterfaceInt half = INT_CASTED_GRID_SIZE_RATIO / 2; 
        InterfaceInt numX = static_cast<InterfaceInt>(i) - static_cast<InterfaceInt>(BUFFER_PIC) - half + INT_CASTED_GRID_SIZE_RATIO; 
        InterfaceInt numY = static_cast<InterfaceInt>(j) - static_cast<InterfaceInt>(BUFFER_PIC) - half + INT_CASTED_GRID_SIZE_RATIO; 
         
        MHDUnsignedInt indexXMHD = numX / INT_CASTED_GRID_SIZE_RATIO - 1 + START_INDEX_IN_MHD_X; 
        MHDUnsignedInt indexYMHD = numY / INT_CASTED_GRID_SIZE_RATIO - 1 + START_INDEX_IN_MHD_Y; 
        MHDUnsignedLongLong indexMHD = getIndex<MHDUnsignedLongLong>(indexXMHD, indexYMHD, NX_MHD, NY_MHD);
        
        MagneticField_MHD B_MHD_x1y1 = getMagneticField_MHD(U, indexMHD);
        MagneticField_MHD B_MHD_x2y1 = getMagneticField_MHD(U, indexMHD + NY_MHD);
        MagneticField_MHD B_MHD_x1y2 = getMagneticField_MHD(U, indexMHD + 1);
        MagneticField_MHD B_MHD_x2y2 = getMagneticField_MHD(U, indexMHD + NY_MHD + 1);

        InterfaceFloat cx1 = static_cast<InterfaceFloat>(numX % INT_CASTED_GRID_SIZE_RATIO) / GRID_SIZE_RATIO;  
        InterfaceFloat cx2 = 1.0 - cx1; 
        InterfaceFloat cy1 = static_cast<InterfaceFloat>(numY % INT_CASTED_GRID_SIZE_RATIO) / GRID_SIZE_RATIO;  
        InterfaceFloat cy2 = 1.0 - cy1; 

        InterfaceFloat bXMHD = B_MHD_x1y1.bX * cx2 * cy2 + B_MHD_x2y1.bX * cx1 * cy2 + B_MHD_x1y2.bX * cx2 * cy1 + B_MHD_x2y2.bX * cx1 * cy1;
        InterfaceFloat bYMHD = B_MHD_x1y1.bY * cx2 * cy2 + B_MHD_x2y1.bY * cx1 * cy2 + B_MHD_x1y2.bY * cx2 * cy1 + B_MHD_x2y2.bY * cx1 * cy1;
        InterfaceFloat bZMHD = B_MHD_x1y1.bZ * cx2 * cy2 + B_MHD_x2y1.bZ * cx1 * cy2 + B_MHD_x1y2.bZ * cx2 * cy1 + B_MHD_x2y2.bZ * cx1 * cy1;
        
        PICUnsignedLongLong indexPIC = getIndex<PICUnsignedLongLong>(i, j, NX_PIC, NY_PIC); 
        MagneticField& B_PIC = B[indexPIC]; 

        B[indexPIC].bX = (1.0 - interlockingFunction[indexPIC]) * bXMHD + interlockingFunction[indexPIC] * B_PIC.bX;
        B[indexPIC].bY = (1.0 - interlockingFunction[indexPIC]) * bYMHD + interlockingFunction[indexPIC] * B_PIC.bY;
        B[indexPIC].bZ = (1.0 - interlockingFunction[indexPIC]) * bZMHD + interlockingFunction[indexPIC] * B_PIC.bZ;
    }
}


void Interface2D::sendMHDtoPIC_B(
    const thrust::device_vector<MHDValue>& U, 
    thrust::device_vector<MagneticField>& B
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((NX_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (NY_PIC + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendMHDtoPIC_B_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        NX_PIC, NY_PIC, 
        pICGridParameter.BUFFER, 
        NX_MHD, NY_MHD, 
        thrust::raw_pointer_cast(U.data()), 
        START_INDEX_IN_MHD_X, 
        START_INDEX_IN_MHD_Y, 
        GRID_SIZE_RATIO, 
        thrust::raw_pointer_cast(interlockingFunction.data()), 
        thrust::raw_pointer_cast(B.data())
    );
    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess) {
        printf("Kernel launch failed at sendMHDtoPIC_B_kernel: %s\n", cudaGetErrorString(err1));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err1 = cudaDeviceSynchronize();
    if (err1 != cudaSuccess) {
        printf("Kernel execution failed at sendMHDtoPIC_B_kernel: %s\n", cudaGetErrorString(err1));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
}


struct ElectricField_MHD {
    InterfaceFloat eX;
    InterfaceFloat eY;
    InterfaceFloat eZ;
};

__device__ ElectricField_MHD getElectricField_MHD(
    const MHDValue* U, 
    const MHDUnsignedInt NY_MHD, 
    const MHDFloat DX_MHD, const MHDFloat DY_MHD, 
    const MHDUnsignedLongLong indexMHD, 
    const bool ACTIVATE_HALL_EFFECT
)
{
    ElectricField_MHD E_MHD; 

    if (ACTIVATE_HALL_EFFECT) {
        printf("not written yet!"); 
    } else {
        MHDFloat u  = U[indexMHD].u;
        MHDFloat v  = U[indexMHD].v;
        MHDFloat w  = U[indexMHD].w; 
        MHDFloat bX = U[indexMHD].bX; 
        MHDFloat bY = U[indexMHD].bY; 
        MHDFloat bZ = U[indexMHD].bZ;
        E_MHD.eX = -(v * bZ - w * bY);
        E_MHD.eY = -(w * bX - u * bZ);
        E_MHD.eZ = -(u * bY - v * bX);
    }
    
    return E_MHD; 
}


__global__ void sendMHDtoPIC_E_kernel(
    const PICUnsignedInt NX_PIC, const PICUnsignedInt NY_PIC, 
    const PICUnsignedInt BUFFER_PIC, 
    const MHDUnsignedInt NX_MHD, const MHDUnsignedInt NY_MHD, 
    const MHDFloat DX_MHD, const MHDFloat DY_MHD, 
    const MHDValue* U, 
    const InterfaceUnsignedInt START_INDEX_IN_MHD_X, 
    const InterfaceUnsignedInt START_INDEX_IN_MHD_Y, 
    const InterfaceUnsignedInt GRID_SIZE_RATIO, 
    const bool ACTIVATE_HALL_EFFECT, 
    const InterfaceFloat* interlockingFunction, 
    ElectricField* E
)
{
    PICUnsignedInt i = blockIdx.x * blockDim.x + threadIdx.x;
    PICUnsignedInt j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < NX_PIC && j < NY_PIC) { 
        InterfaceInt INT_CASTED_GRID_SIZE_RATIO = static_cast<InterfaceInt>(GRID_SIZE_RATIO);
        InterfaceInt half = INT_CASTED_GRID_SIZE_RATIO / 2; 
        InterfaceInt numX = static_cast<InterfaceInt>(i) - static_cast<InterfaceInt>(BUFFER_PIC) - half + INT_CASTED_GRID_SIZE_RATIO; 
        InterfaceInt numY = static_cast<InterfaceInt>(j) - static_cast<InterfaceInt>(BUFFER_PIC) - half + INT_CASTED_GRID_SIZE_RATIO; 
         
        MHDUnsignedInt indexXMHD = numX / INT_CASTED_GRID_SIZE_RATIO - 1 + START_INDEX_IN_MHD_X; 
        MHDUnsignedInt indexYMHD = numY / INT_CASTED_GRID_SIZE_RATIO - 1 + START_INDEX_IN_MHD_Y; 
        MHDUnsignedLongLong indexMHD = getIndex<MHDUnsignedLongLong>(indexXMHD, indexYMHD, NX_MHD, NY_MHD);
        
        ElectricField_MHD E_MHD_x1y1 = getElectricField_MHD(
            U, NY_MHD, DX_MHD, DY_MHD, indexMHD, ACTIVATE_HALL_EFFECT
        );
        ElectricField_MHD E_MHD_x2y1 = getElectricField_MHD(
            U, NY_MHD, DX_MHD, DY_MHD, indexMHD + NY_MHD, ACTIVATE_HALL_EFFECT
        );
        ElectricField_MHD E_MHD_x1y2 = getElectricField_MHD(
            U, NY_MHD, DX_MHD, DY_MHD, indexMHD + 1, ACTIVATE_HALL_EFFECT
        );
        ElectricField_MHD E_MHD_x2y2 = getElectricField_MHD(
            U, NY_MHD, DX_MHD, DY_MHD, indexMHD + NY_MHD + 1, ACTIVATE_HALL_EFFECT
        );

        InterfaceFloat cx1 = static_cast<InterfaceFloat>(numX % INT_CASTED_GRID_SIZE_RATIO) / GRID_SIZE_RATIO;  
        InterfaceFloat cx2 = 1.0 - cx1; 
        InterfaceFloat cy1 = static_cast<InterfaceFloat>(numY % INT_CASTED_GRID_SIZE_RATIO) / GRID_SIZE_RATIO;  
        InterfaceFloat cy2 = 1.0 - cy1; 

        InterfaceFloat eXMHD = E_MHD_x1y1.eX * cx2 * cy2 + E_MHD_x2y1.eX * cx1 * cy2 + E_MHD_x1y2.eX * cx2 * cy1 + E_MHD_x2y2.eX * cx1 * cy1;
        InterfaceFloat eYMHD = E_MHD_x1y1.eY * cx2 * cy2 + E_MHD_x2y1.eY * cx1 * cy2 + E_MHD_x1y2.eY * cx2 * cy1 + E_MHD_x2y2.eY * cx1 * cy1;
        InterfaceFloat eZMHD = E_MHD_x1y1.eZ * cx2 * cy2 + E_MHD_x2y1.eZ * cx1 * cy2 + E_MHD_x1y2.eZ * cx2 * cy1 + E_MHD_x2y2.eZ * cx1 * cy1;
        
        PICUnsignedLongLong indexPIC = getIndex<PICUnsignedLongLong>(i, j, NX_PIC, NY_PIC); 
        ElectricField& E_PIC = E[indexPIC]; 

        E[indexPIC].eX = (1.0 - interlockingFunction[indexPIC]) * eXMHD + interlockingFunction[indexPIC] * E_PIC.eX;
        E[indexPIC].eY = (1.0 - interlockingFunction[indexPIC]) * eYMHD + interlockingFunction[indexPIC] * E_PIC.eY;
        E[indexPIC].eZ = (1.0 - interlockingFunction[indexPIC]) * eZMHD + interlockingFunction[indexPIC] * E_PIC.eZ;
    }
}


void Interface2D::sendMHDtoPIC_E(
    const thrust::device_vector<MHDValue>& U, 
    thrust::device_vector<ElectricField>& E
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((NX_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (NY_PIC + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendMHDtoPIC_E_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        NX_PIC, NY_PIC, 
        pICGridParameter.BUFFER, 
        NX_MHD, NY_MHD, 
        DX_MHD, DY_MHD, 
        thrust::raw_pointer_cast(U.data()), 
        START_INDEX_IN_MHD_X, 
        START_INDEX_IN_MHD_Y, 
        GRID_SIZE_RATIO, 
        mHDConstParameter.ACTIVATE_HALL_EFFECT, 
        thrust::raw_pointer_cast(interlockingFunction.data()), 
        thrust::raw_pointer_cast(E.data())
    );
    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess) {
        printf("Kernel launch failed at sendMHDtoPIC_E_kernel: %s\n", cudaGetErrorString(err1));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err1 = cudaDeviceSynchronize();
    if (err1 != cudaSuccess) {
        printf("Kernel execution failed at sendMHDtoPIC_E_kernel: %s\n", cudaGetErrorString(err1));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
}


struct CurrentField_MHD {
    InterfaceFloat jX;
    InterfaceFloat jY;
    InterfaceFloat jZ;
};

__device__ CurrentField_MHD getCurrentField_MHD(
    const MHDValue* U, 
    const MHDUnsignedInt NY_MHD, 
    const MHDFloat DX_MHD, const MHDFloat DY_MHD, 
    const MHDUnsignedLongLong indexMHD
)
{
    CurrentField_MHD J_MHD; 

    J_MHD.jX = (U[indexMHD + 1].bZ - U[indexMHD - 1].bZ)
             / (2.0 * DY_MHD);
    J_MHD.jY = -(U[indexMHD + NY_MHD].bZ - U[indexMHD - NY_MHD].bZ)
             / (2.0 * DX_MHD);
    J_MHD.jZ = (U[indexMHD + NY_MHD].bY - U[indexMHD - NY_MHD].bY)
             / (2.0 * DX_MHD)
             - (U[indexMHD + 1].bX - U[indexMHD - 1].bX)
             / (2.0 * DY_MHD);

    return J_MHD; 
}

__global__ void sendMHDtoPIC_current_kernel(
    const PICUnsignedInt NX_PIC, const PICUnsignedInt NY_PIC, 
    const PICUnsignedInt BUFFER_PIC, 
    const MHDUnsignedInt NX_MHD, const MHDUnsignedInt NY_MHD, 
    const MHDFloat DX_MHD, const MHDFloat DY_MHD, 
    const MHDValue* U, 
    const InterfaceUnsignedInt START_INDEX_IN_MHD_X, 
    const InterfaceUnsignedInt START_INDEX_IN_MHD_Y, 
    const InterfaceUnsignedInt GRID_SIZE_RATIO, 
    const InterfaceFloat* interlockingFunction, 
    CurrentField* current
)
{
    PICUnsignedInt i = blockIdx.x * blockDim.x + threadIdx.x;
    PICUnsignedInt j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < NX_PIC && j < NY_PIC) { 
        InterfaceInt INT_CASTED_GRID_SIZE_RATIO = static_cast<InterfaceInt>(GRID_SIZE_RATIO);
        InterfaceInt half = INT_CASTED_GRID_SIZE_RATIO / 2; 
        InterfaceInt numX = static_cast<InterfaceInt>(i) - static_cast<InterfaceInt>(BUFFER_PIC) - half + INT_CASTED_GRID_SIZE_RATIO; 
        InterfaceInt numY = static_cast<InterfaceInt>(j) - static_cast<InterfaceInt>(BUFFER_PIC) - half + INT_CASTED_GRID_SIZE_RATIO; 
         
        MHDUnsignedInt indexXMHD = numX / INT_CASTED_GRID_SIZE_RATIO - 1 + START_INDEX_IN_MHD_X; 
        MHDUnsignedInt indexYMHD = numY / INT_CASTED_GRID_SIZE_RATIO - 1 + START_INDEX_IN_MHD_Y; 
        MHDUnsignedLongLong indexMHD = getIndex<MHDUnsignedLongLong>(indexXMHD, indexYMHD, NX_MHD, NY_MHD);
        
        CurrentField_MHD J_MHD_x1y1 = getCurrentField_MHD(
            U, NY_MHD, DX_MHD, DY_MHD, indexMHD
        );
        CurrentField_MHD J_MHD_x2y1 = getCurrentField_MHD(
            U, NY_MHD, DX_MHD, DY_MHD, indexMHD + NY_MHD
        );
        CurrentField_MHD J_MHD_x1y2 = getCurrentField_MHD(
            U, NY_MHD, DX_MHD, DY_MHD, indexMHD + 1
        );
        CurrentField_MHD J_MHD_x2y2 = getCurrentField_MHD(
            U, NY_MHD, DX_MHD, DY_MHD, indexMHD + NY_MHD + 1
        );

        InterfaceFloat cx1 = static_cast<InterfaceFloat>(numX % INT_CASTED_GRID_SIZE_RATIO) / GRID_SIZE_RATIO;  
        InterfaceFloat cx2 = 1.0 - cx1; 
        InterfaceFloat cy1 = static_cast<InterfaceFloat>(numY % INT_CASTED_GRID_SIZE_RATIO) / GRID_SIZE_RATIO;  
        InterfaceFloat cy2 = 1.0 - cy1; 

        InterfaceFloat jXMHD = J_MHD_x1y1.jX * cx2 * cy2 + J_MHD_x2y1.jX * cx1 * cy2 + J_MHD_x1y2.jX * cx2 * cy1 + J_MHD_x2y2.jX * cx1 * cy1;
        InterfaceFloat jYMHD = J_MHD_x1y1.jY * cx2 * cy2 + J_MHD_x2y1.jY * cx1 * cy2 + J_MHD_x1y2.jY * cx2 * cy1 + J_MHD_x2y2.jY * cx1 * cy1;
        InterfaceFloat jZMHD = J_MHD_x1y1.jZ * cx2 * cy2 + J_MHD_x2y1.jZ * cx1 * cy2 + J_MHD_x1y2.jZ * cx2 * cy1 + J_MHD_x2y2.jZ * cx1 * cy1;
        
        PICUnsignedLongLong indexPIC = getIndex<PICUnsignedLongLong>(i, j, NX_PIC, NY_PIC); 
        CurrentField& J_PIC = current[indexPIC]; 

        current[indexPIC].jX = (1.0 - interlockingFunction[indexPIC]) * jXMHD + interlockingFunction[indexPIC] * J_PIC.jX;
        current[indexPIC].jY = (1.0 - interlockingFunction[indexPIC]) * jYMHD + interlockingFunction[indexPIC] * J_PIC.jY;
        current[indexPIC].jZ = (1.0 - interlockingFunction[indexPIC]) * jZMHD + interlockingFunction[indexPIC] * J_PIC.jZ;
    }
}


void Interface2D::sendMHDtoPIC_current(
    const thrust::device_vector<MHDValue>& U, 
    thrust::device_vector<CurrentField>& current
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((NX_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (NY_PIC + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendMHDtoPIC_current_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        NX_PIC, NY_PIC, 
        pICGridParameter.BUFFER, 
        NX_MHD, NY_MHD, 
        DX_MHD, DY_MHD, 
        thrust::raw_pointer_cast(U.data()), 
        START_INDEX_IN_MHD_X, 
        START_INDEX_IN_MHD_Y, 
        GRID_SIZE_RATIO, 
        thrust::raw_pointer_cast(interlockingFunction.data()), 
        thrust::raw_pointer_cast(current.data())
    );
    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess) {
        printf("Kernel launch failed at sendMHDtoPIC_current_kernel: %s\n", cudaGetErrorString(err1));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err1 = cudaDeviceSynchronize();
    if (err1 != cudaSuccess) {
        printf("Kernel execution failed at sendMHDtoPIC_current_kernel: %s\n", cudaGetErrorString(err1));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
}


__global__ void deleteParticles_kernel(
    const PICUnsignedInt NX_PIC, const PICUnsignedInt NY_PIC, 
    const PICFloat DX_PIC, const PICFloat DY_PIC, 
    const PICFloat XMIN_PIC, const PICFloat YMIN_PIC, 
    const PICUnsignedLongLong EXIST_NUM, 
    const InterfaceFloat* interlockingFunction, 
    const InterfaceUnsignedLongLong seed, 
    Particle* particles
)
{
    PICUnsignedLongLong k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < EXIST_NUM) {
        PICFloat xOverDx = (particles[k].x - XMIN_PIC) / DX_PIC;
        PICFloat yOverDy = (particles[k].y - YMIN_PIC) / DY_PIC;

        PICUnsignedInt i = floor(xOverDx);
        PICUnsignedInt j = floor(yOverDy);

        if (i >= NX_PIC || j >= NY_PIC) {
            particles[k].isExist = false; 
            return; 
        }

        PICUnsignedLongLong indexPIC = getIndex<PICUnsignedLongLong>(i, j, NX_PIC, NY_PIC);

        curandState state; 
        curand_init(seed, k, 0, &state);
        InterfaceFloat randomValue = curand_uniform_double(&state);
        if (randomValue > interlockingFunction[indexPIC]) {
            particles[k].isExist = false;
        }
    }
}


void Interface2D::deleteParticles(
    const InterfaceUnsignedLongLong seed, 
    PICUnsignedLongLong& EXIST_NUM, 
    thrust::device_vector<Particle>& particles
)
{

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((EXIST_NUM + threadsPerBlock.x - 1) / threadsPerBlock.x);
    
    deleteParticles_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        NX_PIC, NY_PIC, 
        DX_PIC, DY_PIC, 
        pICGridParameter.XMIN, pICGridParameter.YMIN, 
        EXIST_NUM, 
        thrust::raw_pointer_cast(interlockingFunction.data()), 
        seed, 
        thrust::raw_pointer_cast(particles.data())
    );
    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess) {
        printf("Kernel launch failed at deleteParticles_kernel: %s\n", cudaGetErrorString(err1));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err1 = cudaDeviceSynchronize();
    if (err1 != cudaSuccess) {
        printf("Kernel execution failed at deleteParticles_kernel: %s\n", cudaGetErrorString(err1));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }

    auto partitionEnd = thrust::partition(
        particles.begin(), particles.begin() + EXIST_NUM, 
        [] __device__ (const Particle& p) { return p.isExist; }
    );

    EXIST_NUM = static_cast<PICUnsignedLongLong>(cuda::std::distance(particles.begin(), partitionEnd));
}


__global__ void reloadParticles_kernel(
    const PICUnsignedInt NX_PIC, const PICUnsignedInt NY_PIC, 
    const PICFloat DX_PIC, const PICFloat DY_PIC, 
    const PICFloat XMIN_PIC, const PICFloat YMIN_PIC, 
    const PICFloat C, 
    const ReloadParticlesData* reloadParticlesData, 
    const InterfaceUnsignedLongLong seed, 
    const InterfaceFloat* interlockingFunction, 
    const InterfaceFloat EPS, 
    Particle* particles, 
    InterfaceUnsignedLongLong* particlesNumCounter
)
{
    PICUnsignedInt i = blockIdx.x * blockDim.x + threadIdx.x;
    PICUnsignedInt j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < NX_PIC && j < NY_PIC) {
        PICUnsignedLongLong indexPIC = getIndex<PICUnsignedLongLong>(i, j, NX_PIC, NY_PIC); 

        const InterfaceFloat& u = reloadParticlesData[indexPIC].u;
        const InterfaceFloat& v = reloadParticlesData[indexPIC].v;
        const InterfaceFloat& w = reloadParticlesData[indexPIC].w;
        const InterfaceFloat& L11 = reloadParticlesData[indexPIC].L11;
        const InterfaceFloat& L21 = reloadParticlesData[indexPIC].L21;
        const InterfaceFloat& L22 = reloadParticlesData[indexPIC].L22;
        const InterfaceFloat& L31 = reloadParticlesData[indexPIC].L31;
        const InterfaceFloat& L32 = reloadParticlesData[indexPIC].L32;
        const InterfaceFloat& L33 = reloadParticlesData[indexPIC].L33;

        curandState state; 
        curand_init(seed, indexPIC, 0, &state);
        for (InterfaceUnsignedInt k = 0; k < reloadParticlesData[indexPIC].number; k++) {
            InterfaceFloat randomValue = curand_uniform_double(&state);

            if (randomValue > interlockingFunction[indexPIC]) {
            
                InterfaceFloat x = (static_cast<InterfaceFloat>(i) + curand_uniform_double(&state)) * DX_PIC + XMIN_PIC;
                InterfaceFloat y = (static_cast<InterfaceFloat>(j) + curand_uniform_double(&state)) * DY_PIC + YMIN_PIC;
                InterfaceFloat z = 0.0; 
                
                PICFloat vx, vy, vz; 
                //InterfaceFloat ksi1 = curand_normal_double(&state); 
                //InterfaceFloat ksi2 = curand_normal_double(&state); 
                //InterfaceFloat ksi3 = curand_normal_double(&state); 
                //vx = u + L11 * ksi1;
                //vy = v + L21 * ksi1 + L22 * ksi2;
                //vz = w + L31 * ksi1 + L32 * ksi2 + L33 * ksi3;
                //if (1.0 - (vx * vx + vy * vy + vz * vz) / pow(C, 2) < 0.0) {
                //    InterfaceFloat normalizedVelocity = sqrt(vx * vx + vy * vy + vz * vz);
                //    vx = vx / normalizedVelocity * 0.9 * C;
                //    vy = vy / normalizedVelocity * 0.9 * C;
                //    vz = vz / normalizedVelocity * 0.9 * C;
                //};
                do {
                    InterfaceFloat ksi1 = curand_normal_double(&state); 
                    InterfaceFloat ksi2 = curand_normal_double(&state); 
                    InterfaceFloat ksi3 = curand_normal_double(&state); 
                    vx = u + L11 * ksi1;
                    vy = v + L21 * ksi1 + L22 * ksi2;
                    vz = w + L31 * ksi1 + L32 * ksi2 + L33 * ksi3;
                } while ((vx * vx + vy * vy + vz * vz) >= C * C); 
                
                PICFloat gamma = 1.0 / sqrt(1.0 - (vx * vx + vy * vy + vz * vz) / pow(C, 2));

                Particle particle; 
                particle.x = x; 
                particle.y = y; 
                particle.z = z; 
                particle.ux = vx * gamma; 
                particle.uy = vy * gamma; 
                particle.uz = vz * gamma; 
                particle.gamma = gamma;
                particle.isExist = true;

                PICUnsignedLongLong loadIndex = atomicAdd(&(particlesNumCounter[0]), 1);
                particles[loadIndex] = particle;
            }
        }
    }
}


void Interface2D::reloadParticles(
    const thrust::device_vector<ReloadParticlesData>& reloadParticlesData, 
    const InterfaceUnsignedLongLong seed, 
    thrust::device_vector<Particle>& particles, 
    PICUnsignedLongLong& EXIST_NUM
)
{
    thrust::device_vector<InterfaceUnsignedLongLong> particlesNumCounter(1, 0);
    particlesNumCounter[0] = EXIST_NUM;

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((NX_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (NY_PIC + threadsPerBlock.y - 1) / threadsPerBlock.y);

    reloadParticles_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        NX_PIC, NY_PIC, 
        DX_PIC, DY_PIC, 
        pICGridParameter.XMIN, pICGridParameter.YMIN, 
        pICConstParameter.C, 
        thrust::raw_pointer_cast(reloadParticlesData.data()), 
        seed, 
        thrust::raw_pointer_cast(interlockingFunction.data()), 
        interfaceConstParameter.EPS, 
        thrust::raw_pointer_cast(particles.data()), 
        thrust::raw_pointer_cast(particlesNumCounter.data())
    );
    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess) {
        printf("Kernel launch failed at reloadParticles_kernel: %s\n", cudaGetErrorString(err1));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err1 = cudaDeviceSynchronize();
    if (err1 != cudaSuccess) {
        printf("Kernel execution failed at reloadParticles_kernel: %s\n", cudaGetErrorString(err1));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }

    EXIST_NUM = particlesNumCounter[0];
}


template <typename MomentType>
__device__ MomentType getConvolvedMomentForMHDtoPIC(
    const PICUnsignedInt NX_PIC, const PICUnsignedInt NY_PIC, 
    const InterfaceUnsignedInt GRID_SIZE_RATIO, 
    const MomentType* moment, 
    const PICUnsignedInt i, const PICUnsignedInt j
)
{
    MomentType convolvedMoment; 

    PICFloat weightSum = 0.0;
    for (PICInt dx = -static_cast<PICInt>(GRID_SIZE_RATIO) / 2; dx <= static_cast<PICInt>(GRID_SIZE_RATIO) / 2; dx++) {
        for (PICInt dy = -static_cast<PICInt>(GRID_SIZE_RATIO) / 2; dy <= static_cast<PICInt>(GRID_SIZE_RATIO) / 2; dy++) {
            PICInt localI = static_cast<PICInt>(i) + dx;
            PICInt localJ = static_cast<PICInt>(j) + dy;

            if (0 <= localI && localI < NX_PIC && 0 <= localJ && localJ < NY_PIC)
            {
                PICUnsignedLongLong index = getIndex<PICUnsignedLongLong>(
                    static_cast<PICUnsignedInt>(localI), static_cast<PICUnsignedInt>(localJ), NX_PIC, NY_PIC
                );
                convolvedMoment += moment[index]; 
                weightSum += 1.0;
            }
        }
    }
    convolvedMoment = convolvedMoment / weightSum;

    return convolvedMoment;
}


__global__ void sendMHDtoPIC_moment_kernel(
    const PICUnsignedInt NX_PIC, const PICUnsignedInt NY_PIC, 
    const PICUnsignedInt BUFFER_PIC, 
    const MHDUnsignedInt NX_MHD, const MHDUnsignedInt NY_MHD, 
    const MHDFloat DX_MHD, const MHDFloat DY_MHD, 
    const InterfaceUnsignedInt GRID_SIZE_RATIO, 
    const InterfaceUnsignedInt START_INDEX_IN_MHD_X, 
    const InterfaceUnsignedInt START_INDEX_IN_MHD_Y, 
    const PICFloat M_ION, const PICFloat M_ELECTRON, 
    const PICFloat Q_ION, const PICFloat Q_ELECTRON, 
    const MHDValue* U, 
    const ZerothMoment* zerothMomentIon, 
    const ZerothMoment* zerothMomentElectron, 
    const FirstMoment* firstMomentIon, 
    const FirstMoment* firstMomentElectron, 
    const SecondMoment* secondMomentIon, 
    const SecondMoment* secondMomentElectron, 
    const InterfaceFloat EPS, 
    const InterfaceFloat* interlockingFunction, 
    ReloadParticlesData* reloadParticlesDataIon, 
    ReloadParticlesData* reloadParticlesDataElectron 
)
{
    PICUnsignedInt i = blockIdx.x * blockDim.x + threadIdx.x;
    PICUnsignedInt j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < NX_PIC && j < NY_PIC) {
        InterfaceInt INT_CASTED_GRID_SIZE_RATIO = static_cast<InterfaceInt>(GRID_SIZE_RATIO);
        InterfaceInt half = INT_CASTED_GRID_SIZE_RATIO / 2; 
        InterfaceInt numX = static_cast<InterfaceInt>(i) - static_cast<InterfaceInt>(BUFFER_PIC) - half + INT_CASTED_GRID_SIZE_RATIO; 
        InterfaceInt numY = static_cast<InterfaceInt>(j) - static_cast<InterfaceInt>(BUFFER_PIC) - half + INT_CASTED_GRID_SIZE_RATIO; 
         
        MHDUnsignedInt indexXMHD = numX / INT_CASTED_GRID_SIZE_RATIO - 1 + START_INDEX_IN_MHD_X; 
        MHDUnsignedInt indexYMHD = numY / INT_CASTED_GRID_SIZE_RATIO - 1 + START_INDEX_IN_MHD_Y; 
        MHDUnsignedLongLong indexMHD = getIndex<MHDUnsignedLongLong>(indexXMHD, indexYMHD, NX_MHD, NY_MHD);
        
        const MHDValue& mHDValue_x1y1 = U[indexMHD]; 
        const MHDValue& mHDValue_x2y1 = U[indexMHD + NY_MHD]; 
        const MHDValue& mHDValue_x1y2 = U[indexMHD + 1]; 
        const MHDValue& mHDValue_x2y2 = U[indexMHD + NY_MHD + 1]; 
        CurrentField_MHD J_MHD_x1y1 = getCurrentField_MHD(
            U, NY_MHD, DX_MHD, DY_MHD, indexMHD
        );
        CurrentField_MHD J_MHD_x2y1 = getCurrentField_MHD(
            U, NY_MHD, DX_MHD, DY_MHD, indexMHD + NY_MHD
        );
        CurrentField_MHD J_MHD_x1y2 = getCurrentField_MHD(
            U, NY_MHD, DX_MHD, DY_MHD, indexMHD + 1
        );
        CurrentField_MHD J_MHD_x2y2 = getCurrentField_MHD(
            U, NY_MHD, DX_MHD, DY_MHD, indexMHD + NY_MHD + 1
        );

        InterfaceFloat cx1 = static_cast<InterfaceFloat>(numX % INT_CASTED_GRID_SIZE_RATIO) / GRID_SIZE_RATIO;  
        InterfaceFloat cx2 = 1.0 - cx1; 
        InterfaceFloat cy1 = static_cast<InterfaceFloat>(numY % INT_CASTED_GRID_SIZE_RATIO) / GRID_SIZE_RATIO;  
        InterfaceFloat cy2 = 1.0 - cy1; 

        MHDFloat rhoMHD = mHDValue_x1y1.rho * cx2 * cy2 + mHDValue_x2y1.rho * cx1 * cy2 + mHDValue_x1y2.rho * cx2 * cy1 + mHDValue_x2y2.rho * cx1 * cy1;
        MHDFloat uMHD   = mHDValue_x1y1.u * cx2 * cy2 + mHDValue_x2y1.u * cx1 * cy2 + mHDValue_x1y2.u * cx2 * cy1 + mHDValue_x2y2.u * cx1 * cy1;
        MHDFloat vMHD   = mHDValue_x1y1.v * cx2 * cy2 + mHDValue_x2y1.v * cx1 * cy2 + mHDValue_x1y2.v * cx2 * cy1 + mHDValue_x2y2.v * cx1 * cy1;
        MHDFloat wMHD   = mHDValue_x1y1.w * cx2 * cy2 + mHDValue_x2y1.w * cx1 * cy2 + mHDValue_x1y2.w * cx2 * cy1 + mHDValue_x2y2.w * cx1 * cy1;
        MHDFloat jXMHD  = J_MHD_x1y1.jX * cx2 * cy2 + J_MHD_x2y1.jX * cx1 * cy2 + J_MHD_x1y2.jX * cx2 * cy1 + J_MHD_x2y2.jX * cx1 * cy1;
        MHDFloat jYMHD  = J_MHD_x1y1.jY * cx2 * cy2 + J_MHD_x2y1.jY * cx1 * cy2 + J_MHD_x1y2.jY * cx2 * cy1 + J_MHD_x2y2.jY * cx1 * cy1;
        MHDFloat jZMHD  = J_MHD_x1y1.jZ * cx2 * cy2 + J_MHD_x2y1.jZ * cx1 * cy2 + J_MHD_x1y2.jZ * cx2 * cy1 + J_MHD_x2y2.jZ * cx1 * cy1;
        MHDFloat pXXMHD = mHDValue_x1y1.pXX * cx2 * cy2 + mHDValue_x2y1.pXX * cx1 * cy2 + mHDValue_x1y2.pXX * cx2 * cy1 + mHDValue_x2y2.pXX * cx1 * cy1;
        MHDFloat pYYMHD = mHDValue_x1y1.pYY * cx2 * cy2 + mHDValue_x2y1.pYY * cx1 * cy2 + mHDValue_x1y2.pYY * cx2 * cy1 + mHDValue_x2y2.pYY * cx1 * cy1;
        MHDFloat pZZMHD = mHDValue_x1y1.pZZ * cx2 * cy2 + mHDValue_x2y1.pZZ * cx1 * cy2 + mHDValue_x1y2.pZZ * cx2 * cy1 + mHDValue_x2y2.pZZ * cx1 * cy1;
        MHDFloat pXYMHD = mHDValue_x1y1.pXY * cx2 * cy2 + mHDValue_x2y1.pXY * cx1 * cy2 + mHDValue_x1y2.pXY * cx2 * cy1 + mHDValue_x2y2.pXY * cx1 * cy1;
        MHDFloat pXZMHD = mHDValue_x1y1.pXZ * cx2 * cy2 + mHDValue_x2y1.pXZ * cx1 * cy2 + mHDValue_x1y2.pXZ * cx2 * cy1 + mHDValue_x2y2.pXZ * cx1 * cy1;
        MHDFloat pYZMHD = mHDValue_x1y1.pYZ * cx2 * cy2 + mHDValue_x2y1.pYZ * cx1 * cy2 + mHDValue_x1y2.pYZ * cx2 * cy1 + mHDValue_x2y2.pYZ * cx1 * cy1;

        PICUnsignedLongLong indexPIC = getIndex<PICUnsignedLongLong>(i, j, NX_PIC, NY_PIC); 

        ZerothMoment convolvedZerothMomentIon, convolvedZerothMomentElectron; 
        FirstMoment convolvedFirstMomentIon, convolvedFirstMomentElectron;
        SecondMoment convolvedSecondMomentIon, convolvedSecondMomentElectron;
        convolvedZerothMomentIon      = getConvolvedMomentForMHDtoPIC(NX_PIC, NY_PIC, GRID_SIZE_RATIO, zerothMomentIon, i, j);
        convolvedZerothMomentElectron = getConvolvedMomentForMHDtoPIC(NX_PIC, NY_PIC, GRID_SIZE_RATIO, zerothMomentElectron, i, j);
        convolvedFirstMomentIon       = getConvolvedMomentForMHDtoPIC(NX_PIC, NY_PIC, GRID_SIZE_RATIO, firstMomentIon, i, j);
        convolvedFirstMomentElectron  = getConvolvedMomentForMHDtoPIC(NX_PIC, NY_PIC, GRID_SIZE_RATIO, firstMomentElectron, i, j);
        convolvedSecondMomentIon      = getConvolvedMomentForMHDtoPIC(NX_PIC, NY_PIC, GRID_SIZE_RATIO, secondMomentIon, i, j);
        convolvedSecondMomentElectron = getConvolvedMomentForMHDtoPIC(NX_PIC, NY_PIC, GRID_SIZE_RATIO, secondMomentElectron, i, j);

        PICFloat rhoPIC =  M_ION * convolvedZerothMomentIon.n + M_ELECTRON * convolvedZerothMomentElectron.n;
        PICFloat uPIC   = (M_ION * convolvedFirstMomentIon.x  + M_ELECTRON * convolvedFirstMomentElectron.x) / rhoPIC;
        PICFloat vPIC   = (M_ION * convolvedFirstMomentIon.y  + M_ELECTRON * convolvedFirstMomentElectron.y) / rhoPIC;
        PICFloat wPIC   = (M_ION * convolvedFirstMomentIon.z  + M_ELECTRON * convolvedFirstMomentElectron.z) / rhoPIC;
        PICFloat jXPIC  = Q_ION * convolvedFirstMomentIon.x + Q_ELECTRON * convolvedFirstMomentElectron.x; 
        PICFloat jYPIC  = Q_ION * convolvedFirstMomentIon.y + Q_ELECTRON * convolvedFirstMomentElectron.y; 
        PICFloat jZPIC  = Q_ION * convolvedFirstMomentIon.z + Q_ELECTRON * convolvedFirstMomentElectron.z;  
        PICFloat pXXPIC = M_ION * (
                           convolvedSecondMomentIon.xx
                        - pow(convolvedFirstMomentIon.x, 2) / convolvedZerothMomentIon.n
                       ) + M_ELECTRON * (
                           convolvedSecondMomentElectron.xx
                        - pow(convolvedFirstMomentElectron.x, 2) / convolvedZerothMomentElectron.n
                       ); 
        PICFloat pYYPIC = M_ION * (
                           convolvedSecondMomentIon.yy
                        - pow(convolvedFirstMomentIon.y, 2) / convolvedZerothMomentIon.n
                       ) + M_ELECTRON * (
                           convolvedSecondMomentElectron.yy
                        - pow(convolvedFirstMomentElectron.y, 2) / convolvedZerothMomentElectron.n
                       );
        PICFloat pZZPIC = M_ION * (
                           convolvedSecondMomentIon.zz
                        - pow(convolvedFirstMomentIon.z, 2) / convolvedZerothMomentIon.n
                       ) + M_ELECTRON * (
                           convolvedSecondMomentElectron.zz
                        - pow(convolvedFirstMomentElectron.z, 2) / convolvedZerothMomentElectron.n
                       );
        PICFloat pXYPIC = M_ION * (
                           convolvedSecondMomentIon.xy
                        - convolvedFirstMomentIon.x * convolvedFirstMomentIon.y / convolvedZerothMomentIon.n
                       ) + M_ELECTRON * (
                           convolvedSecondMomentElectron.xy
                        - convolvedFirstMomentElectron.x * convolvedFirstMomentElectron.y / convolvedZerothMomentElectron.n
                       );
        PICFloat pXZPIC = M_ION * (
                           convolvedSecondMomentIon.xz
                        - convolvedFirstMomentIon.x * convolvedFirstMomentIon.z / convolvedZerothMomentIon.n
                       ) + M_ELECTRON * (
                           convolvedSecondMomentElectron.xz
                        - convolvedFirstMomentElectron.x * convolvedFirstMomentElectron.z / convolvedZerothMomentElectron.n
                       );
        PICFloat pYZPIC = M_ION * (
                           convolvedSecondMomentIon.yz
                        - convolvedFirstMomentIon.y * convolvedFirstMomentIon.z / convolvedZerothMomentIon.n
                       ) + M_ELECTRON * (
                           convolvedSecondMomentElectron.yz
                        - convolvedFirstMomentElectron.y * convolvedFirstMomentElectron.z / convolvedZerothMomentElectron.n
                       );

        InterfaceFloat rho = (1.0 - interlockingFunction[indexPIC]) * rhoMHD + interlockingFunction[indexPIC] * rhoPIC;
        InterfaceFloat u = (1.0 - interlockingFunction[indexPIC]) * uMHD + interlockingFunction[indexPIC] * uPIC;
        InterfaceFloat v = (1.0 - interlockingFunction[indexPIC]) * vMHD + interlockingFunction[indexPIC] * vPIC;
        InterfaceFloat w = (1.0 - interlockingFunction[indexPIC]) * wMHD + interlockingFunction[indexPIC] * wPIC;
        InterfaceFloat jX = (1.0 - interlockingFunction[indexPIC]) * jXMHD + interlockingFunction[indexPIC] * jXPIC;
        InterfaceFloat jY = (1.0 - interlockingFunction[indexPIC]) * jYMHD + interlockingFunction[indexPIC] * jYPIC;
        InterfaceFloat jZ = (1.0 - interlockingFunction[indexPIC]) * jZMHD + interlockingFunction[indexPIC] * jZPIC;               
        InterfaceFloat pXX = (1.0 - interlockingFunction[indexPIC]) * pXXMHD + interlockingFunction[indexPIC] * pXXPIC;
        InterfaceFloat pYY = (1.0 - interlockingFunction[indexPIC]) * pYYMHD + interlockingFunction[indexPIC] * pYYPIC;
        InterfaceFloat pZZ = (1.0 - interlockingFunction[indexPIC]) * pZZMHD + interlockingFunction[indexPIC] * pZZPIC;               
        InterfaceFloat pXY = (1.0 - interlockingFunction[indexPIC]) * pXYMHD + interlockingFunction[indexPIC] * pXYPIC;
        InterfaceFloat pXZ = (1.0 - interlockingFunction[indexPIC]) * pXZMHD + interlockingFunction[indexPIC] * pXZPIC;
        InterfaceFloat pYZ = (1.0 - interlockingFunction[indexPIC]) * pYZMHD + interlockingFunction[indexPIC] * pYZPIC;               

        InterfaceUnsignedInt nIon = static_cast<InterfaceUnsignedInt>(round(rho / (M_ION + M_ELECTRON))); 
        InterfaceUnsignedInt nElectron = nIon;  

        //InterfaceFloat uIon = u;  
        //InterfaceFloat vIon = v; 
        //InterfaceFloat wIon = w; 
        //InterfaceFloat uElectron = u + jX / nElectron / Q_ELECTRON;
        //InterfaceFloat vElectron = v + jY / nElectron / Q_ELECTRON;
        //InterfaceFloat wElectron = w + jZ / nElectron / Q_ELECTRON;
        InterfaceFloat uIon = u + M_ELECTRON / Q_ION * jX / rho;  
        InterfaceFloat vIon = v + M_ELECTRON / Q_ION * jY / rho; 
        InterfaceFloat wIon = w + M_ELECTRON / Q_ION * jZ / rho; 
        InterfaceFloat uElectron = u + M_ION / Q_ELECTRON * jX / rho;
        InterfaceFloat vElectron = v + M_ION / Q_ELECTRON * jY / rho;
        InterfaceFloat wElectron = w + M_ION / Q_ELECTRON * jZ / rho;

        //各方向の熱速度を計算
        //圧力テンソルのイオン・電子への分配は等分配を仮定する
        InterfaceFloat rhoIon = rho * M_ION / (M_ION + M_ELECTRON);
        InterfaceFloat CXXIon = pXX / 2.0 / rhoIon;
        InterfaceFloat CYYIon = pYY / 2.0 / rhoIon;
        InterfaceFloat CZZIon = pZZ / 2.0 / rhoIon;
        InterfaceFloat CXYIon = pXY / 2.0 / rhoIon;
        InterfaceFloat CXZIon = pXZ / 2.0 / rhoIon;
        InterfaceFloat CYZIon = pYZ / 2.0 / rhoIon;
        InterfaceFloat L11Ion = sqrt(CXXIon + EPS);
        InterfaceFloat L21Ion = CXYIon / L11Ion;
        InterfaceFloat L31Ion = CXZIon / L11Ion;
        InterfaceFloat L22Ion = sqrt(CYYIon - L21Ion * L21Ion + EPS);
        InterfaceFloat L32Ion = (CYZIon - L31Ion * L21Ion) / L22Ion;
        InterfaceFloat L33Ion = sqrt(CZZIon - L31Ion * L31Ion - L32Ion * L32Ion + EPS);

        InterfaceFloat rhoElectron = rho * M_ELECTRON / (M_ION + M_ELECTRON);
        InterfaceFloat CXXElectron = pXX / 2.0 / rhoElectron;
        InterfaceFloat CYYElectron = pYY / 2.0 / rhoElectron;
        InterfaceFloat CZZElectron = pZZ / 2.0 / rhoElectron;
        InterfaceFloat CXYElectron = pXY / 2.0 / rhoElectron;
        InterfaceFloat CXZElectron = pXZ / 2.0 / rhoElectron;
        InterfaceFloat CYZElectron = pYZ / 2.0 / rhoElectron;
        InterfaceFloat L11Electron = sqrt(CXXElectron + EPS);
        InterfaceFloat L21Electron = CXYElectron / L11Electron;
        InterfaceFloat L31Electron = CXZElectron / L11Electron;
        InterfaceFloat L22Electron = sqrt(CYYElectron - L21Electron * L21Electron + EPS);
        InterfaceFloat L32Electron = (CYZElectron - L31Electron * L21Electron) / L22Electron;
        InterfaceFloat L33Electron = sqrt(CZZElectron - L31Electron * L31Electron - L32Electron * L32Electron + EPS);


        reloadParticlesDataIon     [indexPIC].number = nIon;
        reloadParticlesDataElectron[indexPIC].number = nElectron;
        reloadParticlesDataIon     [indexPIC].u      = uIon;
        reloadParticlesDataIon     [indexPIC].v      = vIon;
        reloadParticlesDataIon     [indexPIC].w      = wIon;
        reloadParticlesDataElectron[indexPIC].u      = uElectron;
        reloadParticlesDataElectron[indexPIC].v      = vElectron; 
        reloadParticlesDataElectron[indexPIC].w      = wElectron; 
        reloadParticlesDataIon     [indexPIC].L11   = L11Ion;
        reloadParticlesDataIon     [indexPIC].L21   = L21Ion;
        reloadParticlesDataIon     [indexPIC].L22   = L22Ion;
        reloadParticlesDataIon     [indexPIC].L31   = L31Ion;
        reloadParticlesDataIon     [indexPIC].L32   = L32Ion;
        reloadParticlesDataIon     [indexPIC].L33   = L33Ion;
        reloadParticlesDataElectron[indexPIC].L11   = L11Electron;
        reloadParticlesDataElectron[indexPIC].L21   = L21Electron;
        reloadParticlesDataElectron[indexPIC].L22   = L22Electron;
        reloadParticlesDataElectron[indexPIC].L31   = L31Electron;
        reloadParticlesDataElectron[indexPIC].L32   = L32Electron;
        reloadParticlesDataElectron[indexPIC].L33   = L33Electron;

        if (reloadParticlesDataIon[indexPIC].number > 100000) printf("Too much ion reloaded (over 100000)!");
        if (reloadParticlesDataElectron[indexPIC].number > 100000) printf("Too much electron reloaded (over 100000)!");
    }
}


void Interface2D::sendMHDtoPIC_particle(
    const thrust::device_vector<MHDValue>& U,  
    const thrust::device_vector<ZerothMoment>& zerothMomentIon, 
    const thrust::device_vector<ZerothMoment>& zerothMomentElectron, 
    const thrust::device_vector<FirstMoment>& firstMomentIon, 
    const thrust::device_vector<FirstMoment>& firstMomentElectron, 
    const thrust::device_vector<SecondMoment>& secondMomentIon, 
    const thrust::device_vector<SecondMoment>& secondMomentElectron, 
    const InterfaceUnsignedLongLong seed, 
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((NX_PIC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (NY_PIC + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendMHDtoPIC_moment_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        NX_PIC, NY_PIC, 
        pICGridParameter.BUFFER, 
        NX_MHD, NY_MHD, 
        DX_MHD, DY_MHD, 
        GRID_SIZE_RATIO, 
        START_INDEX_IN_MHD_X, START_INDEX_IN_MHD_Y, 
        pICConstParameter.M_ION, pICConstParameter.M_ELECTRON, 
        pICConstParameter.Q_ION, pICConstParameter.Q_ELECTRON, 
        thrust::raw_pointer_cast(U.data()),  
        thrust::raw_pointer_cast(zerothMomentIon.data()),  
        thrust::raw_pointer_cast(zerothMomentElectron.data()),  
        thrust::raw_pointer_cast(firstMomentIon.data()),  
        thrust::raw_pointer_cast(firstMomentElectron.data()),  
        thrust::raw_pointer_cast(secondMomentIon.data()),  
        thrust::raw_pointer_cast(secondMomentElectron.data()),  
        interfaceConstParameter.EPS, 
        thrust::raw_pointer_cast(interlockingFunction.data()), 
        thrust::raw_pointer_cast(reloadParticlesDataIon.data()), 
        thrust::raw_pointer_cast(reloadParticlesDataElectron.data())
    );
    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess) {
        printf("Kernel launch failed at sendMHDtoPIC_moment_kernel: %s\n", cudaGetErrorString(err1));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err1 = cudaDeviceSynchronize();
    if (err1 != cudaSuccess) {
        printf("Kernel execution failed at sendMHDtoPIC_moment_kernel: %s\n", cudaGetErrorString(err1));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }

    deleteParticles(
        seed, pICConstParameter.EXIST_NUM_ION, particlesIon 
    );
    deleteParticles(
        seed + 100000000, pICConstParameter.EXIST_NUM_ELECTRON, particlesElectron
    );

    reloadParticles(
        reloadParticlesDataIon, 
        seed + 200000000, 
        particlesIon, 
        pICConstParameter.EXIST_NUM_ION
    ); 
    reloadParticles(
        reloadParticlesDataElectron, 
        seed + 300000000,  
        particlesElectron, 
        pICConstParameter.EXIST_NUM_ELECTRON
    ); 

    if (pICConstParameter.EXIST_NUM_ION > pICConstParameter.TOTAL_NUM_ION) {
        std::cout << "exist number of ion particles exceeds total number (with buffer)" << std::endl;
    }
    if (pICConstParameter.EXIST_NUM_ELECTRON > pICConstParameter.TOTAL_NUM_ELECTRON) {
        std::cout << "exist number of electron particles exceeds total number (with buffer)" << std::endl;
    }
}


