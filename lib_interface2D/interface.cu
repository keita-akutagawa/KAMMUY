#include "interface.hpp"
#include <cmath>


using namespace Interface2DConst;

Interface2D::Interface2D(
    int indexStartMHD, 
    int indexStartPIC
)
    :  indexOfInterfaceStartInMHD(indexStartMHD), 
       indexOfInterfaceStartInPIC(indexStartPIC), 
       interlockingFunctionY(interfaceLength), 
       interlockingFunctionYHalf(interfaceLength - 1),
       host_interlockingFunctionY(interfaceLength), 
       host_interlockingFunctionYHalf(interfaceLength - 1),

       interfacePIC_B(PIC2DConst::nx * Interface2DConst::interfaceLength), 
       interfacePIC_E(PIC2DConst::nx * Interface2DConst::interfaceLength), 
       interfacePIC_current(PIC2DConst::nx * Interface2DConst::interfaceLength), 
       interfacePIC_zerothMomentIon(PIC2DConst::nx * Interface2DConst::interfaceLength), 
       interfacePIC_zerothMomentElectron(PIC2DConst::nx * Interface2DConst::interfaceLength), 
       interfacePIC_firstMomentIon(PIC2DConst::nx * Interface2DConst::interfaceLength), 
       interfacePIC_firstMomentElectron(PIC2DConst::nx * Interface2DConst::interfaceLength), 
       interfacePIC_secondMomentIon(PIC2DConst::nx * Interface2DConst::interfaceLength), 
       interfacePIC_secondMomentElectron(PIC2DConst::nx * Interface2DConst::interfaceLength), 
       interfaceMHD_U(PIC2DConst::nx * Interface2DConst::interfaceLength), 

       tmp_interfacePIC_zerothMomentIon(PIC2DConst::nx * PIC2DConst::ny), 
       tmp_interfacePIC_zerothMomentElectron(PIC2DConst::nx * PIC2DConst::ny), 
       tmp_interfacePIC_firstMomentIon(PIC2DConst::nx * PIC2DConst::ny), 
       tmp_interfacePIC_firstMomentElectron(PIC2DConst::nx * PIC2DConst::ny), 
       tmp_interfacePIC_secondMomentIon(PIC2DConst::nx * PIC2DConst::ny), 
       tmp_interfacePIC_secondMomentElectron(PIC2DConst::nx * PIC2DConst::ny), 
{
    indexOfInterfaceEndInMHD = indexOfInterfaceStartInMHD + Interface2DConst::interfaceLength;
    indexOfInterfaceEndInPIC = indexOfInterfaceStartInPIC + Interface2DConst::interfaceLength;

    for(int i = 0; interfaceLength; i++) {
        host_interlockingFunctionY[i] = 0.5f * (
            1.0f + cos(Interface2DConst::PI  * (i - 0.0f) / (interfaceLength - 0.0f))
        );
    }
    for(int i = 0; interfaceLength - 1; i++) {
        host_interlockingFunctionY[i] = 0.5f * (
            1.0f + cos(Interface2DConst::PI  * (i + 0.5f - 0.0f) / (interfaceLength - 0.0f))
        );
    }

    interlockingFunctionX = host_interlockingFunctionX;
    interlockingFunctionY = host_interlockingFunctionY;
}


__global__ void sendMHDtoPIC_magneticField_yDirection_kernel(
    const float* interlockingFunctionY, 
    const ConservationParameter* U, 
    MagneticField* B, 
    int indexOfInterfaceStartInMHD, 
    int indexOfInterfaceStartInPIC
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < PIC2DConst::device_nx && 0 < j && j < Interface2DConst::interfaceLength - 1) {
        float bXPIC, bYPIC, bZPIC;
        float bXMHD, bYMHD, bZMHD;
        float bXInterface, bYInterface, bZInterface;

        int indexPIC = indexOfInterfaceStartInPIC +  j + i * PIC2DConst::device_nx;
        int indexMHD = indexOfInterfaceStartInMHD +  j + i * IdealMHD2DConst::device_nx;

        //PICのグリッドにMHDを合わせる
        bXPIC = B[indexPIC].bX;
        bYPIC = B[indexPIC].bY;
        bZPIC = B[indexPIC].bZ;
        bXMHD = 0.25f * (U[indexMHD].bX + U[indexMHD - IdealMHD2DConst::device_nx].bX + U[indexMHD + 1].bX + U[indexMHD + 1 - IdealMHD2DConst::device_nx].bX);
        bYMHD = 0.25f * (U[indexMHD].bY + U[indexMHD + IdealMHD2DConst::device_nx].bY + U[indexMHD - 1].bY + U[indexMHD - 1 + IdealMHD2DConst::device_nx].bY);
        bZMHD = 0.25f * (U[indexMHD].bZ + U[indexMHD + IdealMHD2DConst::device_nx].bZ + U[indexMHD + 1].bZ + U[indexMHD + 1 + IdealMHD2DConst::device_nx].bZ);

        bXInterface = interlockingFunctionYHalf[j] * bXMHD + (1.0f - interlockingFunctionYHalf[j]) * bXPIC;
        bYInterface = interlockingFunctionY[j]     * bYMHD + (1.0f - interlockingFunctionY[j])     * bYPIC;
        bZInterface = interlockingFunctionYHalf[j] * bZMHD + (1.0f - interlockingFunctionYHalf[j]) * bZPIC;
        
        B[indexPIC].bX = bXInterface;
        B[indexPIC].bY = bYInterface;
        B[indexPIC].bZ = bZInterface;
    }
}

void Interface2D::sendMHDtoPIC_magneticField_yDirection(
    const thrust::device_vector<ConservationParameter>& U, 
    thrust::device_vector<MagneticField>& B
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((PIC2DConst::nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (Interface2DConst::interfaceLength + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendMHDtoPIC_magneticField_yDirection_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()),
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(B.data()), 
        indexOfInterfaceStartInMHD, 
        indexOfInterfaceStartInPIC 
    );

    cudaDeviceSynchronize();
}


__global__ void sendMHDtoPIC_electricField_yDirection_kernel(
    const float* interlockingFunctionY, 
    const ConservationParameter* U, 
    ElectricField* E, 
    int indexOfInterfaceStartInMHD, 
    int indexOfInterfaceStartInPIC
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < PIC2DConst::device_nx && 0 < j &&  j < Interface2DConst::interfaceLength - 1) {
        float eXPIC, eYPIC, eZPIC;
        float eXMHD, eYMHD, eZMHD;
        float eXPlusX1MHD;
        float eYPlusY1MHD;
        float eXMHD, eYMHD, eZMHD;
        float rho, u, v, z;
        float bXMHD, bYMHD, bZMHD;
        float eXInterface, eYInterface, eZInterface;

        int indexPIC = indexOfInterfaceStartInPIC +  j + i * PIC2DConst::device_nx;
        int indexMHD = indexOfInterfaceStartInMHD +  j + i * IdealMHD2DConst::device_nx;

        //PICのグリッドにMHDを合わせる
        eXPIC = E[indexPIC].eX;
        eYPIC = E[indexPIC].eY;
        eZPIC = E[indexPIC].eZ;

        rho = U[indexMHD].rho;
        u = U[indexMHD].rhoU / rho;
        v = U[indexMHD].rhoV / rho;
        w = U[indexMHD].rhoW / rho; 
        bXMHD = 0.5f * (U[indexMHD].bX + U[indexMHD - IdealMHD2D::device_nx].bX);
        bYMHD = 0.5f * (U[indexMHD].bY + U[indexMHD - 1].bY);
        bZMHD = U[indexMHD].bZ;
        eXMHD = -(v * bZMHD - w * bYMHD);
        eYMHD = -(w * bXMHD - u * bZMHD);
        eZMHD = -(u * bYMHD - v * bXMHD);

        rho = U[indexMHD + IdealMHD2D::device_nx].rho;
        u = U[indexMHD + IdealMHD2D::device_nx].rhoU / rho;
        v = U[indexMHD + IdealMHD2D::device_nx].rhoV / rho;
        w = U[indexMHD + IdealMHD2D::device_nx].rhoW / rho; 
        bXMHD = 0.5f * (U[indexMHD + IdealMHD2D::device_nx].bX + U[indexMHD].bX);
        bYMHD = 0.5f * (U[indexMHD + IdealMHD2D::device_nx].bY + U[indexMHD - 1 + IdealMHD2D::device_nx].bY);
        bZMHD = U[indexMHD + IdealMHD2D::device_nx].bZ;
        eXPlusX1MHD = -(v * bZMHD - w * bYMHD);

        rho = U[indexMHD + 1].rho;
        u = U[indexMHD + 1].rhoU / rho;
        v = U[indexMHD + 1].rhoV / rho;
        w = U[indexMHD + 1].rhoW / rho; 
        bXMHD = 0.5f * (U[indexMHD + 1].bX + U[indexMHD + 1 - IdealMHD2D::device_nx].bX);
        bYMHD = 0.5f * (U[indexMHD + 1].bY + U[indexMHD].bY);
        bZMHD = U[indexMHD + 1].bZ;
        eYPlusY1MHD = -(w * bXMHD - u * bZMHD);


        eXInterface = interlockingFunctionY[j]     * 0.5f * (eXMHD + eXPlusX1MHD) + (1.0f - interlockingFunctionY[j])     * eXPIC;
        eYInterface = interlockingFunctionYHalf[j] * 0.5f * (eYMHD + eYPlusY1MHD) + (1.0f - interlockingFunctionYHalf[j]) * eYPIC;
        eZInterface = interlockingFunctionY[j]     * eZMHD                        + (1.0f - interlockingFunctionY[j])     * eZPIC;
         
        E[indexPIC].eX = eXInterface;
        E[indexPIC].eY = eYInterface;
        E[indexPIC].eZ = eZInterface;
    }
}

void Interface2D::sendMHDtoPIC_electricField_yDirection(
    const thrust::device_vector<ConservationParameter>& U, 
    thrust::device_vector<ElectricField>& E
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((PIC2DConst::nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (Interface2DConst::interfaceLength + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendMHDtoPIC_electricField_yDirection_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()),
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(E.data()), 
        indexOfInterfaceStartInMHD, 
        indexOfInterfaceStartInPIC 
    );

    cudaDeviceSynchronize();
}


__global__ void sendMHDtoPIC_currentField_yDirection_kernel(
    const float* interlockingFunctionY, 
    const ConservationParameter* U, 
    CurrentField* current, 
    int indexOfInterfaceStartInMHD, 
    int indexOfInterfaceStartInPIC
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < PIC2DConst::device_nx && 0 < j && j < Interface2DConst::interfaceLength - 1) {
        float jXPIC, jYPIC, jZPIC;
        float jXMHD, jYMHD, jZMHD;
        float jXPlusX1MHD; 
        float jYPlusY1MHD; 
        float jXInterface, jYInterface, jZInterface;
        int nx = IdealMHD2DConst::device_nx;
        float dx = IdealMHD2DConst::device_dx, dy = IdealMHD2DConst::device_dy;

        int indexPIC = indexOfInterfaceStartInPIC +  j + i * PIC2DConst::device_nx;
        int indexMHD = indexOfInterfaceStartInMHD +  j + i * IdealMHD2DConst::device_nx;

        //PICのグリッドにMHDを合わせる
        jXPIC = current[indexPIC].jX;
        jYPIC = current[indexPIC].jY;
        jZPIC = current[indexPIC].jZ;
        jXMHD = (U[indexMHD + 1].bZ - U[indexMHD - 1].bZ) / (2.0f * dy);
        jYMHD = -(U[indexMHD + nx].bZ - U[indexMHD - nx].bZ) / (2.0f * dx);
        jZMHD = 0.25f * ((U[indexMHD + nx].bY - U[indexMHD].bY) / dx - (U[indexMHD + 1].bX - U[indexMHD].bX) / dy 
                       + (U[indexMHD].bY - U[indexMHD - nx].bY) / dx - (U[indexMHD + 1 - nx].bX - U[indexMHD - nx].bX) / dy
                       + (U[indexMHD - 1 + nx].bY - U[indexMHD - 1].bY) / dx - (U[indexMHD].bX - U[indexMHD - 1].bX) / dy
                       + (U[indexMHD - 1].bY - U[indexMHD - 1 - nx].bY) / dx - (U[indexMHD - nx].bX - U[indexMHD - 1 - nx].bX) / dy);

        jXPlusX1MHD = (U[indexMHD + 2].bZ - U[indexMHD].bZ) / (2.0f * dy);
        jYPlusY1MHD = -(U[indexMHD + 2 * nx].bZ - U[indexMHD].bZ) / (2.0f * dx);

        jXInterface = interlockingFunctionY[j]     * 0.5f * (jXMHD + jXPlusX1MHD) + (1.0f - interlockingFunctionY[j])     * jXPIC;
        jYInterface = interlockingFunctionYHalf[j] * 0.5f * (jYMHD + jYPlusY1MHD) + (1.0f - interlockingFunctionYHalf[j]) * jYPIC;
        jZInterface = interlockingFunctionY[j]     * jZMHD                        + (1.0f - interlockingFunctionY[j])     * jZPIC;
        
        current[indexPIC].jX = jXInterface;
        current[indexPIC].jY = jYInterface;
        current[indexPIC].jZ = jZInterface;
    }
}

void Interface2D::sendMHDtoPIC_currentField_yDirection(
    const thrust::device_vector<ConservationParameter>& U, 
    thrust::device_vector<CurrentField>& current
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((PIC2DConst::nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (Interface2DConst::interfaceLength + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sendMHDtoPIC_currentField_yDirection_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(interlockingFunctionY.data()),
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(current.data()), 
        indexOfInterfaceStartInMHD, 
        indexOfInterfaceStartInPIC 
    );

    cudaDeviceSynchronize();
}

