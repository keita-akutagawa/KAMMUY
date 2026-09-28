#include "10momentMHD2D.hpp"


OROCHI2D::OROCHI2D(
    nlohmann::json& configJSON, 
    nlohmann::json& constJSON, 
    nlohmann::json& gridJSON
)
  : mHDConstParameter(constJSON), 
    mHDGridParameter(gridJSON)
{
    for (MHDUnsignedInt level = 0; level < mHDGridParameter.NUMBER_OF_LEVELS; level++) {
        timeIntegrators.push_back(
            TimeIntegratorFactory::create(
                configJSON, level, mHDConstParameter, mHDGridParameter
            )
        );
        outputs.push_back(
            OutputFactory::create(
                configJSON, mHDGridParameter.NX[level], mHDGridParameter.NY[level], 
                mHDConstParameter
            )
        );
    }
    for (MHDUnsignedInt level = 1; level < mHDGridParameter.NUMBER_OF_LEVELS; level++) {
        smrs.push_back(
            std::make_unique<SMR>(
                configJSON, level, mHDConstParameter, mHDGridParameter
            )
        );
    }
}


void OROCHI2D::oneStep(
    const MHDUnsignedInt step
)
{
    MHDUnsignedInt level = 0;
    timeIntegrators[level]->setUPast();
    timeIntegrators[level]->push(mHDConstParameter.DT);
    if (mHDGridParameter.NUMBER_OF_LEVELS == 1 || !mHDGridParameter.SMR_AVAIL) return;

    level = 1;
    smrStep(timeIntegrators, smrs, level, mHDConstParameter.DT);
    timeIntegrators[0]->getBoundaryRef().applyUForAllDirection(
        timeIntegrators[0]->getURef()
    );
}


void OROCHI2D::smrStep(
    std::vector<std::unique_ptr<TimeIntegrator>>& timeIntegrators, 
    std::vector<std::unique_ptr<SMR>>& smrs, 
    MHDUnsignedInt level, 
    const MHDFloat DT
)
{
    for (MHDInt substep = 0; substep < 2; substep++) {
        smrs[level - 1]->pushOneLayer(
            timeIntegrators[level - 1]->getUPastRef(), 
            timeIntegrators[level - 1]->getURef(), 
            DT / 2.0, substep, timeIntegrators[level]
        );
        if (level < mHDGridParameter.NUMBER_OF_LEVELS - 1) {
            smrStep(timeIntegrators, smrs, level + 1, DT / 2.0);
        }
    }

    smrs[level - 1]->synchronizeOneLayer(
        timeIntegrators[level]->getURef(), 
        timeIntegrators[level - 1]->getURef()
    );
}


void OROCHI2D::save()
{
    for (MHDUnsignedInt level = 0; level < mHDGridParameter.NUMBER_OF_LEVELS; level++) {  
        std::string addName = "_" + std::to_string(level);
        outputs[level]->save(timeIntegrators[level]->getURef(), addName);
    }
}


__global__ static void isCrashed_kernel(
    const MHDValue* U, 
    const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
    const MHDUnsignedInt BUFFER, 
    bool* isNegativeRho, bool* isNegativePressure
)
{
    MHDUnsignedInt i = blockIdx.x * blockDim.x + threadIdx.x;
    MHDUnsignedInt j = blockIdx.y * blockDim.y + threadIdx.y;

    if (BUFFER <= i && i < NX - BUFFER && BUFFER <= j && j < NY - BUFFER) {
        MHDUnsignedLongLong index = getIndex<MHDUnsignedLongLong>(i, j, NX, NY);
        
        if (U[index].rho < 0.0 && !isNegativeRho[0]) {
            isNegativeRho[0] = true; 
            printf("rho becomes negative at (%u, %u)\n", i, j);
        }
        if (U[index].pXX < 0.0) {
            isNegativePressure[0] = true; 
            printf("pXX becomes negative at (%u, %u)\n", i, j);
        }
        if (U[index].pYY < 0.0) {
            isNegativePressure[0] = true; 
            printf("pYY becomes negative at (%u, %u)\n", i, j);
        }
        if (U[index].pZZ < 0.0) {
            isNegativePressure[0] = true; 
            printf("pZZ becomes negative at (%u, %u)\n", i, j);
        }
    }
}


bool OROCHI2D::isCrashed()
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mHDGridParameter.NX[0] + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (mHDGridParameter.NY[0] + threadsPerBlock.y - 1) / threadsPerBlock.y);

    thrust::device_vector<bool> isNegativeRho(1), isNegativePressure(1);
    isNegativeRho[0] = false; 
    isNegativePressure[0] = false; 
    isCrashed_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(timeIntegrators[0]->getURef().data()), 
        mHDGridParameter.NX[0], mHDGridParameter.NY[0], 
        mHDGridParameter.BUFFER, 
        thrust::raw_pointer_cast(isNegativeRho.data()), 
        thrust::raw_pointer_cast(isNegativePressure.data())
    ); 
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed at isCrashed_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed at isCrashed_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }

    bool isCrashed = isNegativeRho[0] || isNegativePressure[0];

    return isCrashed;
}


MHDGridParameter& OROCHI2D::getMHDGridParameterRef()
{
    return mHDGridParameter;
}


MHDConstParameter& OROCHI2D::getMHDConstParameterRef()
{
    return mHDConstParameter;
}


std::vector<std::unique_ptr<TimeIntegrator>>& OROCHI2D::getTimeIntegratorsRef()
{
    return timeIntegrators;
}

