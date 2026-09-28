#include "interface2D.hpp"


void Interface2D::resetPICtoMHDParameters()
{
    thrust::fill(
        B_PICtoMHD.begin(), 
        B_PICtoMHD.end(), 
        MagneticField()
    );

    thrust::fill(
        zerothMomentIon_PICtoMHD.begin(), 
        zerothMomentIon_PICtoMHD.end(), 
        ZerothMoment()
    );
    thrust::fill(
        zerothMomentElectron_PICtoMHD.begin(), 
        zerothMomentElectron_PICtoMHD.end(), 
        ZerothMoment()
    );

    thrust::fill(
        firstMomentIon_PICtoMHD.begin(), 
        firstMomentIon_PICtoMHD.end(), 
        FirstMoment()
    );
    thrust::fill(
        firstMomentElectron_PICtoMHD.begin(), 
        firstMomentElectron_PICtoMHD.end(), 
        FirstMoment()
    );

    thrust::fill(
        secondMomentIon_PICtoMHD.begin(), 
        secondMomentIon_PICtoMHD.end(), 
        SecondMoment()
    );
    thrust::fill(
        secondMomentElectron_PICtoMHD.begin(), 
        secondMomentElectron_PICtoMHD.end(), 
        SecondMoment()
    );
}


__global__ void calculateSpaceAveragedPICtoMHDParameters_kernel(
    const PICUnsignedInt NX_PIC, const PICUnsignedInt NY_PIC, 
    const PICUnsignedInt BUFFER_PIC, 
    const InterfaceUnsignedInt GRID_SIZE_RATIO, 
    const MagneticField* B, 
    const ZerothMoment* zerothMomentIon, 
    const ZerothMoment* zerothMomentElectron, 
    const FirstMoment* firstMomentIon, 
    const FirstMoment* firstMomentElectron, 
    const SecondMoment* secondMomentIon, 
    const SecondMoment* secondMomentElectron, 
    MagneticField* B_PICtoMHD, 
    ZerothMoment* zerothMomentIon_PICtoMHD, 
    ZerothMoment* zerothMomentElectron_PICtoMHD, 
    FirstMoment* firstMomentIon_PICtoMHD, 
    FirstMoment* firstMomentElectron_PICtoMHD, 
    SecondMoment* secondMomentIon_PICtoMHD, 
    SecondMoment* secondMomentElectron_PICtoMHD
)
{
    InterfaceUnsignedInt i = blockIdx.x * blockDim.x + threadIdx.x;
    InterfaceUnsignedInt j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < (NX_PIC - 2 * BUFFER_PIC) / GRID_SIZE_RATIO && j < (NY_PIC - 2 * BUFFER_PIC) / GRID_SIZE_RATIO) {
        MHDUnsignedLongLong indexPICtoMHD = getIndex<MHDUnsignedLongLong>(
            i, j, (NX_PIC - 2 * BUFFER_PIC) / GRID_SIZE_RATIO, (NY_PIC - 2 * BUFFER_PIC) / GRID_SIZE_RATIO
        );

        InterfaceFloat averagedCoef = 1.0 / (GRID_SIZE_RATIO * GRID_SIZE_RATIO); 
        for (InterfaceUnsignedInt ii = i * GRID_SIZE_RATIO + BUFFER_PIC; ii < (i + 1) * GRID_SIZE_RATIO + BUFFER_PIC; ii++) {
            for (InterfaceUnsignedInt jj = j * GRID_SIZE_RATIO + BUFFER_PIC; jj < (j + 1) * GRID_SIZE_RATIO + BUFFER_PIC; jj++) {
                MHDUnsignedLongLong indexPIC = getIndex<MHDUnsignedLongLong>(ii, jj, NX_PIC, NY_PIC);

                B_PICtoMHD[indexPICtoMHD] += B[indexPIC] * averagedCoef;
                zerothMomentIon_PICtoMHD[indexPICtoMHD] += zerothMomentIon[indexPIC] * averagedCoef;
                zerothMomentElectron_PICtoMHD[indexPICtoMHD] += zerothMomentElectron[indexPIC] * averagedCoef;
                firstMomentIon_PICtoMHD[indexPICtoMHD] += firstMomentIon[indexPIC] * averagedCoef;
                firstMomentElectron_PICtoMHD[indexPICtoMHD] += firstMomentElectron[indexPIC] * averagedCoef;
                secondMomentIon_PICtoMHD[indexPICtoMHD] += secondMomentIon[indexPIC] * averagedCoef;
                secondMomentElectron_PICtoMHD[indexPICtoMHD] += secondMomentElectron[indexPIC] * averagedCoef;
            }
        }
    }
}


void Interface2D::calculateSpaceAveragedPICtoMHDParameters(
    const thrust::device_vector<MagneticField>& B, 
    const thrust::device_vector<ZerothMoment>& zerothMomentIon, 
    const thrust::device_vector<ZerothMoment>& zerothMomentElectron, 
    const thrust::device_vector<FirstMoment>& firstMomentIon, 
    const thrust::device_vector<FirstMoment>& firstMomentElectron, 
    const thrust::device_vector<SecondMoment>& secondMomentIon, 
    const thrust::device_vector<SecondMoment>& secondMomentElectron
)
{
    resetPICtoMHDParameters(); 

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(((NX_PIC - 2 * pICGridParameter.BUFFER) / GRID_SIZE_RATIO + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       ((NY_PIC - 2 * pICGridParameter.BUFFER) / GRID_SIZE_RATIO + threadsPerBlock.y - 1) / threadsPerBlock.y);

    calculateSpaceAveragedPICtoMHDParameters_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        NX_PIC, NY_PIC, 
        pICGridParameter.BUFFER, 
        GRID_SIZE_RATIO, 
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(zerothMomentIon.data()), 
        thrust::raw_pointer_cast(zerothMomentElectron.data()), 
        thrust::raw_pointer_cast(firstMomentIon.data()), 
        thrust::raw_pointer_cast(firstMomentElectron.data()), 
        thrust::raw_pointer_cast(secondMomentIon.data()), 
        thrust::raw_pointer_cast(secondMomentElectron.data()), 
        thrust::raw_pointer_cast(B_PICtoMHD.data()), 
        thrust::raw_pointer_cast(zerothMomentIon_PICtoMHD.data()), 
        thrust::raw_pointer_cast(zerothMomentElectron_PICtoMHD.data()), 
        thrust::raw_pointer_cast(firstMomentIon_PICtoMHD.data()), 
        thrust::raw_pointer_cast(firstMomentElectron_PICtoMHD.data()), 
        thrust::raw_pointer_cast(secondMomentIon_PICtoMHD.data()), 
        thrust::raw_pointer_cast(secondMomentElectron_PICtoMHD.data())
    );
    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess) {
        printf("Kernel launch failed at calculateSpaceAveragedPICtoMHDParameters_kernel: %s\n", cudaGetErrorString(err1));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err1 = cudaDeviceSynchronize();
    if (err1 != cudaSuccess) {
        printf("Kernel execution failed at calculateSpaceAveragedPICtoMHDParameters_kernel: %s\n", cudaGetErrorString(err1));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
}


__global__ void sendPICtoMHD_kernel(
    const PICUnsignedInt NX_PIC, const PICUnsignedInt NY_PIC, 
    const PICUnsignedInt BUFFER_PIC, 
    const MHDUnsignedInt NX_MHD, const MHDUnsignedInt NY_MHD, 
    const InterfaceUnsignedInt INDEX_START_IN_MHD_X, 
    const InterfaceUnsignedInt INDEX_START_IN_MHD_Y, 
    const InterfaceUnsignedInt GRID_SIZE_RATIO, 
    const PICFloat M_ION, const PICFloat M_ELECTRON, 
    const MagneticField* B_PICtoMHD, 
    const ZerothMoment* zerothMomentIon_PICtoMHD, 
    const ZerothMoment* zerothMomentElectron_PICtoMHD, 
    const FirstMoment* firstMomentIon_PICtoMHD, 
    const FirstMoment* firstMomentElectron_PICtoMHD, 
    const SecondMoment* secondMomentIon_PICtoMHD, 
    const SecondMoment* secondMomentElectron_PICtoMHD, 
    const InterfaceFloat* interlockingFunction, 
    MHDValue* U
)
{
    MHDUnsignedInt i = blockIdx.x * blockDim.x + threadIdx.x;
    MHDUnsignedInt j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < (NX_PIC - 2 * BUFFER_PIC) / GRID_SIZE_RATIO && j < (NY_PIC - 2 * BUFFER_PIC) / GRID_SIZE_RATIO) {
        MHDUnsignedLongLong indexMHD = getIndex<MHDUnsignedLongLong>(
            i + INDEX_START_IN_MHD_X, j + INDEX_START_IN_MHD_Y, NX_MHD, NY_MHD
        );

        const MHDValue& mHDValue = U[indexMHD];

        const MHDFloat& rhoMHD = mHDValue.rho;
        const MHDFloat& uMHD   = mHDValue.u;
        const MHDFloat& vMHD   = mHDValue.v;
        const MHDFloat& wMHD   = mHDValue.w;
        const MHDFloat& bXMHD  = mHDValue.bX;
        const MHDFloat& bYMHD  = mHDValue.bY;
        const MHDFloat& bZMHD  = mHDValue.bZ;
        const MHDFloat& pXXMHD = mHDValue.pXX; 
        const MHDFloat& pYYMHD = mHDValue.pYY; 
        const MHDFloat& pZZMHD = mHDValue.pZZ; 
        const MHDFloat& pXYMHD = mHDValue.pXY; 
        const MHDFloat& pXZMHD = mHDValue.pXZ; 
        const MHDFloat& pYZMHD = mHDValue.pYZ; 
        
        MHDUnsignedLongLong indexPICtoMHD = getIndex<MHDUnsignedLongLong>(
            i, j, (NX_PIC - 2 * BUFFER_PIC) / GRID_SIZE_RATIO, (NY_PIC - 2 * BUFFER_PIC) / GRID_SIZE_RATIO
        );

        MHDFloat rhoPIC =  M_ION * zerothMomentIon_PICtoMHD[indexPICtoMHD].n + M_ELECTRON * zerothMomentElectron_PICtoMHD[indexPICtoMHD].n;
        MHDFloat uPIC   = (M_ION * firstMomentIon_PICtoMHD[indexPICtoMHD].x  + M_ELECTRON * firstMomentElectron_PICtoMHD[indexPICtoMHD].x) / rhoPIC;
        MHDFloat vPIC   = (M_ION * firstMomentIon_PICtoMHD[indexPICtoMHD].y  + M_ELECTRON * firstMomentElectron_PICtoMHD[indexPICtoMHD].y) / rhoPIC;
        MHDFloat wPIC   = (M_ION * firstMomentIon_PICtoMHD[indexPICtoMHD].z  + M_ELECTRON * firstMomentElectron_PICtoMHD[indexPICtoMHD].z) / rhoPIC;
        MHDFloat bXPIC  = B_PICtoMHD[indexPICtoMHD].bX; 
        MHDFloat bYPIC  = B_PICtoMHD[indexPICtoMHD].bY; 
        MHDFloat bZPIC  = B_PICtoMHD[indexPICtoMHD].bZ; 
        MHDFloat pXXPIC = M_ION * (
                                secondMomentIon_PICtoMHD[indexPICtoMHD].xx
                             - pow(firstMomentIon_PICtoMHD[indexPICtoMHD].x, 2) / zerothMomentIon_PICtoMHD[indexPICtoMHD].n
                            ) + M_ELECTRON * (
                                secondMomentElectron_PICtoMHD[indexPICtoMHD].xx
                             - pow(firstMomentElectron_PICtoMHD[indexPICtoMHD].x, 2) / zerothMomentElectron_PICtoMHD[indexPICtoMHD].n
                            ); 
        MHDFloat pYYPIC = M_ION * (
                                secondMomentIon_PICtoMHD[indexPICtoMHD].yy
                             - pow(firstMomentIon_PICtoMHD[indexPICtoMHD].y, 2) / zerothMomentIon_PICtoMHD[indexPICtoMHD].n
                            ) + M_ELECTRON * (
                                secondMomentElectron_PICtoMHD[indexPICtoMHD].yy
                             - pow(firstMomentElectron_PICtoMHD[indexPICtoMHD].y, 2) / zerothMomentElectron_PICtoMHD[indexPICtoMHD].n
                            );
        MHDFloat pZZPIC = M_ION * (
                                secondMomentIon_PICtoMHD[indexPICtoMHD].zz
                             - pow(firstMomentIon_PICtoMHD[indexPICtoMHD].z, 2) / zerothMomentIon_PICtoMHD[indexPICtoMHD].n
                            ) + M_ELECTRON * (
                                secondMomentElectron_PICtoMHD[indexPICtoMHD].zz
                             - pow(firstMomentElectron_PICtoMHD[indexPICtoMHD].z, 2) / zerothMomentElectron_PICtoMHD[indexPICtoMHD].n
                            );
        MHDFloat pXYPIC = M_ION * (
                                secondMomentIon_PICtoMHD[indexPICtoMHD].xy
                             - firstMomentIon_PICtoMHD[indexPICtoMHD].x * firstMomentIon_PICtoMHD[indexPICtoMHD].y / zerothMomentIon_PICtoMHD[indexPICtoMHD].n
                            ) + M_ELECTRON * (
                                secondMomentElectron_PICtoMHD[indexPICtoMHD].xy
                             - firstMomentElectron_PICtoMHD[indexPICtoMHD].x * firstMomentElectron_PICtoMHD[indexPICtoMHD].y / zerothMomentElectron_PICtoMHD[indexPICtoMHD].n
                            );
        MHDFloat pXZPIC = M_ION * (
                                secondMomentIon_PICtoMHD[indexPICtoMHD].xz
                             - firstMomentIon_PICtoMHD[indexPICtoMHD].x * firstMomentIon_PICtoMHD[indexPICtoMHD].z / zerothMomentIon_PICtoMHD[indexPICtoMHD].n
                            ) + M_ELECTRON * (
                                secondMomentElectron_PICtoMHD[indexPICtoMHD].xz
                             - firstMomentElectron_PICtoMHD[indexPICtoMHD].x * firstMomentElectron_PICtoMHD[indexPICtoMHD].z / zerothMomentElectron_PICtoMHD[indexPICtoMHD].n
                            );
        MHDFloat pYZPIC = M_ION * (
                                secondMomentIon_PICtoMHD[indexPICtoMHD].yz
                             - firstMomentIon_PICtoMHD[indexPICtoMHD].y * firstMomentIon_PICtoMHD[indexPICtoMHD].z / zerothMomentIon_PICtoMHD[indexPICtoMHD].n
                            ) + M_ELECTRON * (
                                secondMomentElectron_PICtoMHD[indexPICtoMHD].yz
                             - firstMomentElectron_PICtoMHD[indexPICtoMHD].y * firstMomentElectron_PICtoMHD[indexPICtoMHD].z / zerothMomentElectron_PICtoMHD[indexPICtoMHD].n
                            );
        
        PICUnsignedLongLong indexPIC = getIndex<PICUnsignedLongLong>(
            i * GRID_SIZE_RATIO + BUFFER_PIC, j * GRID_SIZE_RATIO + BUFFER_PIC, NX_PIC, NY_PIC 
        ); 

        InterfaceFloat rho = (1.0 - interlockingFunction[indexPIC]) * rhoMHD + interlockingFunction[indexPIC] * rhoPIC;
        InterfaceFloat u   = (1.0 - interlockingFunction[indexPIC]) * uMHD   + interlockingFunction[indexPIC] * uPIC;
        InterfaceFloat v   = (1.0 - interlockingFunction[indexPIC]) * vMHD   + interlockingFunction[indexPIC] * vPIC;
        InterfaceFloat w   = (1.0 - interlockingFunction[indexPIC]) * wMHD   + interlockingFunction[indexPIC] * wPIC;
        InterfaceFloat bX  = (1.0 - interlockingFunction[indexPIC]) * bXMHD  + interlockingFunction[indexPIC] * bXPIC;
        InterfaceFloat bY  = (1.0 - interlockingFunction[indexPIC]) * bYMHD  + interlockingFunction[indexPIC] * bYPIC;
        InterfaceFloat bZ  = (1.0 - interlockingFunction[indexPIC]) * bZMHD  + interlockingFunction[indexPIC] * bZPIC;               
        InterfaceFloat pXX = (1.0 - interlockingFunction[indexPIC]) * pXXMHD + interlockingFunction[indexPIC] * pXXPIC;
        InterfaceFloat pYY = (1.0 - interlockingFunction[indexPIC]) * pYYMHD + interlockingFunction[indexPIC] * pYYPIC;
        InterfaceFloat pZZ = (1.0 - interlockingFunction[indexPIC]) * pZZMHD + interlockingFunction[indexPIC] * pZZPIC;               
        InterfaceFloat pXY = (1.0 - interlockingFunction[indexPIC]) * pXYMHD + interlockingFunction[indexPIC] * pXYPIC;
        InterfaceFloat pXZ = (1.0 - interlockingFunction[indexPIC]) * pXZMHD + interlockingFunction[indexPIC] * pXZPIC;
        InterfaceFloat pYZ = (1.0 - interlockingFunction[indexPIC]) * pYZMHD + interlockingFunction[indexPIC] * pYZPIC;               
            
        U[indexMHD].rho = rho;
        U[indexMHD].u   = u;
        U[indexMHD].v   = v;
        U[indexMHD].w   = w;
        U[indexMHD].bX  = bX;
        U[indexMHD].bY  = bY;
        U[indexMHD].bZ  = bZ;
        U[indexMHD].pXX = pXX;
        U[indexMHD].pYY = pYY;
        U[indexMHD].pZZ = pZZ;
        U[indexMHD].pXY = pXY;
        U[indexMHD].pXZ = pXZ;
        U[indexMHD].pYZ = pYZ;
    }
}


void Interface2D::sendPICtoMHD(
    thrust::device_vector<MHDValue>& U
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((NX_PIC / GRID_SIZE_RATIO + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (NY_PIC / GRID_SIZE_RATIO + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendPICtoMHD_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        NX_PIC, NY_PIC, 
        pICGridParameter.BUFFER, 
        NX_MHD, NY_MHD,
        START_INDEX_IN_MHD_X, START_INDEX_IN_MHD_Y,
        GRID_SIZE_RATIO,
        pICConstParameter.M_ION, pICConstParameter.M_ELECTRON,
        thrust::raw_pointer_cast(B_PICtoMHD.data()), 
        thrust::raw_pointer_cast(zerothMomentIon_PICtoMHD.data()), 
        thrust::raw_pointer_cast(zerothMomentElectron_PICtoMHD.data()), 
        thrust::raw_pointer_cast(firstMomentIon_PICtoMHD.data()), 
        thrust::raw_pointer_cast(firstMomentElectron_PICtoMHD.data()), 
        thrust::raw_pointer_cast(secondMomentIon_PICtoMHD.data()), 
        thrust::raw_pointer_cast(secondMomentElectron_PICtoMHD.data()), 
        thrust::raw_pointer_cast(interlockingFunction.data()), 
        thrust::raw_pointer_cast(U.data())
    );
    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess) {
        printf("Kernel launch failed at sendPICtoMHD_kernel: %s\n", cudaGetErrorString(err1));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err1 = cudaDeviceSynchronize();
    if (err1 != cudaSuccess) {
        printf("Kernel execution failed at sendPICtoMHD_kernel: %s\n", cudaGetErrorString(err1));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
}


