#include "ssprk3.hpp"


SSPRK3::SSPRK3(
    const nlohmann::json& configJSON, 
    const MHDUnsignedInt level, 
    MHDConstParameter& mHDConstParameter, 
    const MHDGridParameter& mHDGridParameter
)
  : NX(mHDGridParameter.NX[level]), 
    NY(mHDGridParameter.NY[level]), 
    DX(mHDGridParameter.DX[level]), 
    DY(mHDGridParameter.DY[level]), 
    level(level), 
    mHDConstParameter(mHDConstParameter), 
    mHDGridParameter(mHDGridParameter), 

    order(getRequiredForJSON<MHDInt>(configJSON, "order")), 

    dtVector(NX * NY), 

    U(NX * NY),
    U1(NX * NY),
    U2(NX * NY),
    UPast(NX * NY),  

    filteringFluxX(NX * NY), 
    filteringFluxY(NX * NY), 

    sourceTermCalculator(level, mHDConstParameter, mHDGridParameter)
{
    reconstructor = ReconstructorFactory::create(configJSON, NX, NY, mHDConstParameter);

    if (level == 0) {
        boundary = std::make_unique<Boundary>(configJSON, NX, NY, mHDConstParameter, mHDGridParameter);
    }
    if (level > 0) {
        smrBoundary = std::make_unique<SMRBoundary>(
            configJSON, 
            level, 
            mHDConstParameter, 
            mHDGridParameter
        );
    }
}


void SSPRK3::setUPast()
{
    thrust::copy(U.begin(), U.end(), UPast.begin());
}


__global__ void calculateFilteringFlux_kernel(
    const MHDValue* leftMHDValue, 
    const MHDValue* centerMHDValue, 
    const MHDValue* rightMHDValue, 
    const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
    const MHDFloat CDIFF, const MHDFloat maxSpeed, 
    const MHDUnsignedInt shift, 
    FilteringFluxValue* filteringFlux
)
{
    MHDUnsignedInt i = blockIdx.x * blockDim.x + threadIdx.x;
    MHDUnsignedInt j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < NX - 1 && j < NY - 1) {
        MHDUnsignedLongLong index = getIndex<MHDUnsignedLongLong>(i, j, NX, NY);

        const MHDValue& left  = leftMHDValue[index];
        const MHDValue& right = rightMHDValue[index];

        filteringFlux[index].f0  = CDIFF * maxSpeed / 2.0 * (right.rho - left.rho); 
        filteringFlux[index].f1  = CDIFF * maxSpeed / 2.0 * (right.u - left.u);
        filteringFlux[index].f2  = CDIFF * maxSpeed / 2.0 * (right.v - left.v);
        filteringFlux[index].f3  = CDIFF * maxSpeed / 2.0 * (right.w - left.w);
        filteringFlux[index].f4  = CDIFF * maxSpeed / 2.0 * (right.bX - left.bX);
        filteringFlux[index].f5  = CDIFF * maxSpeed / 2.0 * (right.bY - left.bY);
        filteringFlux[index].f6  = CDIFF * maxSpeed / 2.0 * (right.bZ - left.bZ);
        filteringFlux[index].f7  = CDIFF * maxSpeed / 2.0 * (right.pXX - left.pXX);
        filteringFlux[index].f8  = CDIFF * maxSpeed / 2.0 * (right.pYY - left.pYY);
        filteringFlux[index].f9  = CDIFF * maxSpeed / 2.0 * (right.pZZ - left.pZZ);
        filteringFlux[index].f10 = CDIFF * maxSpeed / 2.0 * (right.pXY - left.pXY);
        filteringFlux[index].f11 = CDIFF * maxSpeed / 2.0 * (right.pXZ - left.pXZ);
        filteringFlux[index].f12 = CDIFF * maxSpeed / 2.0 * (right.pYZ - left.pYZ);
        filteringFlux[index].f13 = CDIFF * maxSpeed / 2.0 * (right.psi - left.psi);
    }
}


void SSPRK3::calculateFilteringFluxForOneDirection(
    const thrust::device_vector<MHDValue>& U, 
    const MHDFloat maxSpeed, const MHDUnsignedInt shift, 
    thrust::device_vector<FilteringFluxValue>& filteringFlux
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (NY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    reconstructor->calculateReconstructedMHDValue(U, shift); 
    const thrust::device_vector<MHDValue>& leftMHDValue = reconstructor->getLeftMHDValueRef();
    const thrust::device_vector<MHDValue>& centerMHDValue = reconstructor->getCenterMHDValueRef();
    const thrust::device_vector<MHDValue>& rightMHDValue = reconstructor->getRightMHDValueRef();

    calculateFilteringFlux_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(leftMHDValue.data()), 
        thrust::raw_pointer_cast(centerMHDValue.data()), 
        thrust::raw_pointer_cast(rightMHDValue.data()), 
        NX, NY,  
        mHDConstParameter.CDIFF, maxSpeed, 
        shift, 
        thrust::raw_pointer_cast(filteringFlux.data())
    ); 
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed at calculateFilteringFlux_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed at calculateFilteringFlux_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
}


void SSPRK3::calculateFilteringFlux(
    const thrust::device_vector<MHDValue>& U, 
    const MHDFloat maxSpeed 
)
{
    calculateFilteringFluxForOneDirection(U, maxSpeed, NY, filteringFluxX);
    calculateFilteringFluxForOneDirection(U, maxSpeed, 1,  filteringFluxY);
}


__device__ HeatingValue calculateViscousHeatingTerm(
    const MHDValue* centerMHDValue, 
    const FilteringFluxValue* fluxX, const FilteringFluxValue* fluxY, 
    const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
    const MHDFloat DX, const MHDFloat DY, 
    const MHDUnsignedLongLong index
)
{
    const MHDValue& center = centerMHDValue[index];
    const MHDValue& left   = centerMHDValue[index - NY];
    const MHDValue& right  = centerMHDValue[index + NY];
    const MHDValue& down   = centerMHDValue[index - 1];
    const MHDValue& up     = centerMHDValue[index + 1];
    const FilteringFluxValue& fX     = fluxX[index];
    const FilteringFluxValue& fXLeft = fluxX[index - NY];
    const FilteringFluxValue& fY     = fluxY[index];
    const FilteringFluxValue& fYDown = fluxY[index - 1];

    MHDFloat QvisTotal = (
         + (0.5 * (center.rho + right.rho)) * fX.f1 * (right.u - center.u) / DX
         + (0.5 * (left.rho + center.rho)) * fXLeft.f1 * (center.u - left.u) / DX
         + (0.5 * (center.rho + up.rho)) * fY.f1 * (up.u - center.u) / DY
         + (0.5 * (down.rho + center.rho)) * fYDown.f1 * (center.u - down.u) / DY
    ) + (
         + (0.5 * (center.rho + right.rho)) * fX.f2 * (right.v - center.v) / DX
         + (0.5 * (left.rho + center.rho)) * fXLeft.f2 * (center.v - left.v) / DX
         + (0.5 * (center.rho + up.rho)) * fY.f2 * (up.v - center.v) / DY
         + (0.5 * (down.rho + center.rho)) * fYDown.f2 * (center.v - down.v) / DY
    ) + (
         + (0.5 * (center.rho + right.rho)) * fX.f3 * (right.w - center.w) / DX
         + (0.5 * (left.rho + center.rho)) * fXLeft.f3 * (center.w - left.w) / DX
         + (0.5 * (center.rho + up.rho)) * fY.f3 * (up.w - center.w) / DY
         + (0.5 * (down.rho + center.rho)) * fYDown.f3 * (center.w - down.w) / DY
    );
    
    HeatingValue Q; 
    Q.XX = QvisTotal / 3.0; 
    Q.YY = QvisTotal / 3.0;
    Q.ZZ = QvisTotal / 3.0;

    return Q; 
}



__device__ HeatingValue calculateResistiveHeatingTerm(
    const MHDValue* centerMHDValue, 
    const FilteringFluxValue* fluxX, const FilteringFluxValue* fluxY, 
    const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
    const MHDFloat DX, const MHDFloat DY, 
    const MHDUnsignedLongLong index 
)
{
    const MHDValue& center = centerMHDValue[index];
    const MHDValue& left   = centerMHDValue[index - NY];
    const MHDValue& right  = centerMHDValue[index + NY];
    const MHDValue& down   = centerMHDValue[index - 1];
    const MHDValue& up     = centerMHDValue[index + 1];
    const FilteringFluxValue& fX     = fluxX[index];
    const FilteringFluxValue& fXLeft = fluxX[index - NY];
    const FilteringFluxValue& fY     = fluxY[index];
    const FilteringFluxValue& fYDown = fluxY[index - 1];

    MHDFloat QresTotal = (
          fX.f4 * (right.bX - center.bX) / DX
        + fXLeft.f4 * (center.bX - left.bX) / DX
        + fY.f4 * (up.bX - center.bX) / DY
        + fYDown.f4 * (center.bX - down.bX) / DY
    ) + (
          fX.f5 * (right.bY - center.bY) / DX
        + fXLeft.f5 * (center.bY - left.bY) / DX
        + fY.f5 * (up.bY - center.bY) / DY
        + fYDown.f5 * (center.bY - down.bY) / DY
    ) + (
          fX.f6 * (right.bZ - center.bZ) / DX
        + fXLeft.f6 * (center.bZ - left.bZ) / DX
        + fY.f6 * (up.bZ - center.bZ) / DY
        + fYDown.f6 * (center.bZ - down.bZ) / DY
    );

    HeatingValue Q; 
    Q.XX = QresTotal / 3.0; 
    Q.YY = QresTotal / 3.0;
    Q.ZZ = QresTotal / 3.0;

    return Q; 
}


__device__ HeatingValue calculate9WaveHeatingTerm(
    const MHDValue* centerMHDValue, 
    const MHDUnsignedInt NX, const MHDUnsignedInt NY,
    const MHDFloat DX, const MHDFloat DY, 
    const MHDUnsignedLongLong index, 
    const MHDInt order
)
{
    MHDUnsignedInt shiftX = NY, shiftY = 1;

    const MHDValue& centerMinus3X = centerMHDValue[index - 3 * shiftX];
    const MHDValue& centerMinus2X = centerMHDValue[index - 2 * shiftX];
    const MHDValue& centerMinus1X = centerMHDValue[index - 1 * shiftX];
    const MHDValue& center        = centerMHDValue[index];
    const MHDValue& centerPlus1X  = centerMHDValue[index + shiftX];
    const MHDValue& centerPlus2X  = centerMHDValue[index + 2 * shiftX];
    const MHDValue& centerPlus3X  = centerMHDValue[index + 3 * shiftX];

    const MHDValue& centerMinus3Y = centerMHDValue[index - 3 * shiftY];
    const MHDValue& centerMinus2Y = centerMHDValue[index - 2 * shiftY];
    const MHDValue& centerMinus1Y = centerMHDValue[index - 1 * shiftY];
    const MHDValue& centerPlus1Y  = centerMHDValue[index + shiftY];
    const MHDValue& centerPlus2Y  = centerMHDValue[index + 2 * shiftY];
    const MHDValue& centerPlus3Y  = centerMHDValue[index + 3 * shiftY];


    MHDFloat QpsiTotal = -2.0 * center.psi * (
        differenceOperator(
            centerMinus3X.bX, centerMinus2X.bX, centerMinus1X.bX, 
            center.bX, 
            centerPlus1X.bX, centerPlus2X.bX, centerPlus3X.bX, 
            DX, order 
        ) + differenceOperator(
            centerMinus3Y.bY, centerMinus2Y.bY, centerMinus1Y.bY, 
            center.bY, 
            centerPlus1Y.bY, centerPlus2Y.bY, centerPlus3Y.bY, 
            DY, order 
        )
    );

    HeatingValue Q; 
    Q.XX = QpsiTotal / 3.0;
    Q.YY = QpsiTotal / 3.0;
    Q.ZZ = QpsiTotal / 3.0;

    return Q;
}


__device__ RHSValue calculateRHSValue(
    const MHDValue* centerMHDValue, 
    const FilteringFluxValue* filteringFluxX, const FilteringFluxValue* filteringFluxY, 
    const HeatingValue& QVis, const HeatingValue& QRes, const HeatingValue& Q9Wave, 
    const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
    const MHDFloat DX, const MHDFloat DY, 
    const MHDFloat cPsi, const MHDFloat tauPsi, 
    const MHDUnsignedLongLong index, const MHDInt order
)
{
    MHDUnsignedInt shiftX = NY, shiftY = 1;

    const MHDValue& centerMinus3X = centerMHDValue[index - 3 * shiftX];
    const MHDValue& centerMinus2X = centerMHDValue[index - 2 * shiftX];
    const MHDValue& centerMinus1X = centerMHDValue[index - 1 * shiftX];
    const MHDValue& center        = centerMHDValue[index];
    const MHDValue& centerPlus1X  = centerMHDValue[index + shiftX];
    const MHDValue& centerPlus2X  = centerMHDValue[index + 2 * shiftX];
    const MHDValue& centerPlus3X  = centerMHDValue[index + 3 * shiftX];

    const MHDValue& centerMinus3Y = centerMHDValue[index - 3 * shiftY];
    const MHDValue& centerMinus2Y = centerMHDValue[index - 2 * shiftY];
    const MHDValue& centerMinus1Y = centerMHDValue[index - 1 * shiftY];
    const MHDValue& centerPlus1Y  = centerMHDValue[index + shiftY];
    const MHDValue& centerPlus2Y  = centerMHDValue[index + 2 * shiftY];
    const MHDValue& centerPlus3Y  = centerMHDValue[index + 3 * shiftY];

    const FilteringFluxValue& fX = filteringFluxX[index];
    const FilteringFluxValue& fMinus1X = filteringFluxX[index - shiftX];
    const FilteringFluxValue& fY = filteringFluxY[index];
    const FilteringFluxValue& fMinus1Y = filteringFluxY[index - shiftY];

    RHSValue RHS; 

    RHS.v0 = -differenceOperator(
        centerMinus3X.rho * centerMinus3X.u, 
        centerMinus2X.rho * centerMinus2X.u, 
        centerMinus1X.rho * centerMinus1X.u, 
        center.rho * center.u, 
        centerPlus1X.rho * centerPlus1X.u, 
        centerPlus2X.rho * centerPlus2X.u,
        centerPlus3X.rho * centerPlus3X.u, 
        DX, order
    ) - differenceOperator(
        centerMinus3Y.rho * centerMinus3Y.v, 
        centerMinus2Y.rho * centerMinus2Y.v, 
        centerMinus1Y.rho * centerMinus1Y.v, 
        center.rho * center.v, 
        centerPlus1Y.rho * centerPlus1Y.v, 
        centerPlus2Y.rho * centerPlus2Y.v, 
        centerPlus3Y.rho * centerPlus3Y.v, 
        DY, order
    ) + (fX.f0 - fMinus1X.f0) / DX 
      + (fY.f0 - fMinus1Y.f0) / DY;

    RHS.v1 = -0.5 * differenceOperator(
        centerMinus3X.rho * centerMinus3X.u * centerMinus3X.u, 
        centerMinus2X.rho * centerMinus2X.u * centerMinus2X.u, 
        centerMinus1X.rho * centerMinus1X.u * centerMinus1X.u, 
        center.rho * center.u * center.u, 
        centerPlus1X.rho * centerPlus1X.u * centerPlus1X.u, 
        centerPlus2X.rho * centerPlus2X.u * centerPlus2X.u,
        centerPlus3X.rho * centerPlus3X.u * centerPlus3X.u, 
        DX, order
    ) - 0.5 * differenceOperator(
        centerMinus3Y.rho * centerMinus3Y.v * centerMinus3Y.u, 
        centerMinus2Y.rho * centerMinus2Y.v * centerMinus2Y.u, 
        centerMinus1Y.rho * centerMinus1Y.v * centerMinus1Y.u, 
        center.rho * center.v * center.u, 
        centerPlus1Y.rho * centerPlus1Y.v * centerPlus1Y.u, 
        centerPlus2Y.rho * centerPlus2Y.v * centerPlus2Y.u, 
        centerPlus3Y.rho * centerPlus3Y.v * centerPlus3Y.u, 
        DY, order
    ) - 0.5 * center.rho * center.u * differenceOperator(
        centerMinus3X.u, centerMinus2X.u, centerMinus1X.u, 
        center.u, 
        centerPlus1X.u, centerPlus2X.u, centerPlus3X.u, 
        DX, order
    ) - 0.5 * center.rho * center.v * differenceOperator(
        centerMinus3Y.u, centerMinus2Y.u, centerMinus1Y.u, 
        center.u, 
        centerPlus1Y.u, centerPlus2Y.u, centerPlus3Y.u, 
        DY, order
    ) - 0.5 * center.u * differenceOperator(
        centerMinus3X.rho * centerMinus3X.u, 
        centerMinus2X.rho * centerMinus2X.u, 
        centerMinus1X.rho * centerMinus1X.u, 
        center.rho * center.u, 
        centerPlus1X.rho * centerPlus1X.u,
        centerPlus2X.rho * centerPlus2X.u, 
        centerPlus3X.rho * centerPlus3X.u, 
        DX, order
    ) - 0.5 * center.u * differenceOperator(
        centerMinus3Y.rho * centerMinus3Y.v, 
        centerMinus2Y.rho * centerMinus2Y.v, 
        centerMinus1Y.rho * centerMinus1Y.v, 
        center.rho * center.v, 
        centerPlus1Y.rho * centerPlus1Y.v, 
        centerPlus2Y.rho * centerPlus2Y.v, 
        centerPlus3Y.rho * centerPlus3Y.v, 
        DY, order
    ) - differenceOperator(
        centerMinus3X.pXX, centerMinus2X.pXX, centerMinus1X.pXX, 
        center.pXX, 
        centerPlus1X.pXX, centerPlus2X.pXX, centerPlus3X.pXX, 
        DX, order 
    ) - differenceOperator(
        centerMinus3Y.pXY, centerMinus2Y.pXY, centerMinus1Y.pXY, 
        center.pXY, 
        centerPlus1Y.pXY, centerPlus2Y.pXY, centerPlus3Y.pXY, 
        DY, order 
    ) - center.bY * differenceOperator(
        centerMinus3X.bY, centerMinus2X.bY, centerMinus1X.bY, 
        center.bY, 
        centerPlus1X.bY, centerPlus2X.bY, centerPlus3X.bY, 
        DX, order 
    ) - center.bZ * differenceOperator(
        centerMinus3X.bZ, centerMinus2X.bZ, centerMinus1X.bZ, 
        center.bZ, 
        centerPlus1X.bZ, centerPlus2X.bZ, centerPlus3X.bZ, 
        DX, order 
    ) + center.bY * differenceOperator(
        centerMinus3Y.bX, centerMinus2Y.bX, centerMinus1Y.bX, 
        center.bX, 
        centerPlus1Y.bX, centerPlus2Y.bX, centerPlus3Y.bX, 
        DY, order 
    ) + ((0.5 * (center.rho + centerPlus1X.rho) * fX.f1 + 0.5 * (center.u + centerPlus1X.u) * fX.f0)
       - (0.5 * (centerMinus1X.rho + center.rho) * fMinus1X.f1 + 0.5 * (centerMinus1X.u + center.u) * fMinus1X.f0)) / DX 
      + ((0.5 * (center.rho + centerPlus1Y.rho) * fY.f1 + 0.5 * (center.u + centerPlus1Y.u) * fY.f0)
       - (0.5 * (centerMinus1Y.rho + center.rho) * fMinus1Y.f1 + 0.5 * (centerMinus1Y.u + center.u) * fMinus1Y.f0)) / DY; 
    
    RHS.v2 = -0.5 * differenceOperator(
        centerMinus3X.rho * centerMinus3X.u * centerMinus3X.v, 
        centerMinus2X.rho * centerMinus2X.u * centerMinus2X.v, 
        centerMinus1X.rho * centerMinus1X.u * centerMinus1X.v, 
        center.rho * center.u * center.v, 
        centerPlus1X.rho * centerPlus1X.u * centerPlus1X.v, 
        centerPlus2X.rho * centerPlus2X.u * centerPlus2X.v,
        centerPlus3X.rho * centerPlus3X.u * centerPlus3X.v, 
        DX, order
    ) - 0.5 * differenceOperator(
        centerMinus3Y.rho * centerMinus3Y.v * centerMinus3Y.v, 
        centerMinus2Y.rho * centerMinus2Y.v * centerMinus2Y.v, 
        centerMinus1Y.rho * centerMinus1Y.v * centerMinus1Y.v, 
        center.rho * center.v * center.v, 
        centerPlus1Y.rho * centerPlus1Y.v * centerPlus1Y.v, 
        centerPlus2Y.rho * centerPlus2Y.v * centerPlus2Y.v, 
        centerPlus3Y.rho * centerPlus3Y.v * centerPlus3Y.v, 
        DY, order
    ) - 0.5 * center.rho * center.u * differenceOperator(
        centerMinus3X.v, centerMinus2X.v, centerMinus1X.v, 
        center.v, 
        centerPlus1X.v,centerPlus2X.v, centerPlus3X.v, 
        DX, order
    ) - 0.5 * center.rho * center.v * differenceOperator(
        centerMinus3Y.v, centerMinus2Y.v, centerMinus1Y.v, 
        center.v, 
        centerPlus1Y.v, centerPlus2Y.v, centerPlus3Y.v, 
        DY, order
    ) - 0.5 * center.v * differenceOperator(
        centerMinus3X.rho * centerMinus3X.u, 
        centerMinus2X.rho * centerMinus2X.u, 
        centerMinus1X.rho * centerMinus1X.u, 
        center.rho * center.u, 
        centerPlus1X.rho * centerPlus1X.u, 
        centerPlus2X.rho * centerPlus2X.u, 
        centerPlus3X.rho * centerPlus3X.u, 
        DX, order
    ) - 0.5 * center.v * differenceOperator(
        centerMinus3Y.rho * centerMinus3Y.v, 
        centerMinus2Y.rho * centerMinus2Y.v, 
        centerMinus1Y.rho * centerMinus1Y.v, 
        center.rho * center.v, 
        centerPlus1Y.rho * centerPlus1Y.v, 
        centerPlus2Y.rho * centerPlus2Y.v,
        centerPlus3Y.rho * centerPlus3Y.v, 
        DY, order
    ) - differenceOperator(
        centerMinus3X.pXY, centerMinus2X.pXY, centerMinus1X.pXY, 
        center.pXY, 
        centerPlus1X.pXY, centerPlus2X.pXY, centerPlus3X.pXY, 
        DX, order 
    ) - differenceOperator(
        centerMinus3Y.pYY, centerMinus2Y.pYY, centerMinus1Y.pYY, 
        center.pYY, 
        centerPlus1Y.pYY, centerPlus2Y.pYY, centerPlus3Y.pYY, 
        DY, order 
    ) - center.bX * differenceOperator(
        centerMinus3Y.bX, centerMinus2Y.bX, centerMinus1Y.bX, 
        center.bX, 
        centerPlus1Y.bX, centerPlus2Y.bX, centerPlus3Y.bX, 
        DY, order 
    ) - center.bZ * differenceOperator(
        centerMinus3Y.bZ, centerMinus2Y.bZ, centerMinus1Y.bZ, 
        center.bZ, 
        centerPlus1Y.bZ, centerPlus2Y.bZ, centerPlus3Y.bZ, 
        DY, order 
    ) + center.bX * differenceOperator(
        centerMinus3X.bY, centerMinus2X.bY, centerMinus1X.bY, 
        center.bY, 
        centerPlus1X.bY, centerPlus2X.bY, centerPlus3X.bY, 
        DX, order 
    ) + ((0.5 * (center.rho + centerPlus1X.rho) * fX.f2 + 0.5 * (center.v + centerPlus1X.v) * fX.f0)
       - (0.5 * (centerMinus1X.rho + center.rho) * fMinus1X.f2 + 0.5 * (centerMinus1X.v + center.v) * fMinus1X.f0)) / DX 
      + ((0.5 * (center.rho + centerPlus1Y.rho) * fY.f2 + 0.5 * (center.v + centerPlus1Y.v) * fY.f0)
       - (0.5 * (centerMinus1Y.rho + center.rho) * fMinus1Y.f2 + 0.5 * (centerMinus1Y.v + center.v) * fMinus1Y.f0)) / DY; 
    
    RHS.v3 = -0.5 * differenceOperator(
        centerMinus3X.rho * centerMinus3X.u * centerMinus3X.w, 
        centerMinus2X.rho * centerMinus2X.u * centerMinus2X.w, 
        centerMinus1X.rho * centerMinus1X.u * centerMinus1X.w, 
        center.rho * center.u * center.w, 
        centerPlus1X.rho * centerPlus1X.u * centerPlus1X.w, 
        centerPlus2X.rho * centerPlus2X.u * centerPlus2X.w,
        centerPlus3X.rho * centerPlus3X.u * centerPlus3X.w, 
        DX, order
    ) - 0.5 * differenceOperator(
        centerMinus3Y.rho * centerMinus3Y.v * centerMinus3Y.w, 
        centerMinus2Y.rho * centerMinus2Y.v * centerMinus2Y.w, 
        centerMinus1Y.rho * centerMinus1Y.v * centerMinus1Y.w, 
        center.rho * center.v * center.w, 
        centerPlus1Y.rho * centerPlus1Y.v * centerPlus1Y.w, 
        centerPlus2Y.rho * centerPlus2Y.v * centerPlus2Y.w, 
        centerPlus3Y.rho * centerPlus3Y.v * centerPlus3Y.w, 
        DY, order
    ) - 0.5 * center.rho * center.u * differenceOperator(
        centerMinus3X.w, centerMinus2X.w, centerMinus1X.w, 
        center.w, 
        centerPlus1X.w, centerPlus2X.w, centerPlus3X.w, 
        DX, order
    ) - 0.5 * center.rho * center.v * differenceOperator(
        centerMinus3Y.w, centerMinus2Y.w, centerMinus1Y.w,
        center.w, 
        centerPlus1Y.w, centerPlus2Y.w, centerPlus3Y.w, 
        DY, order
    ) - 0.5 * center.w * differenceOperator(
        centerMinus3X.rho * centerMinus3X.u, 
        centerMinus2X.rho * centerMinus2X.u, 
        centerMinus1X.rho * centerMinus1X.u, 
        center.rho * center.u, 
        centerPlus1X.rho * centerPlus1X.u, 
        centerPlus2X.rho * centerPlus2X.u, 
        centerPlus3X.rho * centerPlus3X.u, 
        DX, order
    ) - 0.5 * center.w * differenceOperator(
        centerMinus3Y.rho * centerMinus3Y.v, 
        centerMinus2Y.rho * centerMinus2Y.v, 
        centerMinus1Y.rho * centerMinus1Y.v, 
        center.rho * center.v, 
        centerPlus1Y.rho * centerPlus1Y.v, 
        centerPlus2Y.rho * centerPlus2Y.v, 
        centerPlus3Y.rho * centerPlus3Y.v, 
        DY, order
    ) - differenceOperator(
        centerMinus3X.pXZ, centerMinus2X.pXZ, centerMinus1X.pXZ, 
        center.pXZ, 
        centerPlus1X.pXZ, centerPlus2X.pXZ, centerPlus3X.pXZ, 
        DX, order 
    ) - differenceOperator(
        centerMinus3Y.pYZ, centerMinus2Y.pYZ, centerMinus1Y.pYZ, 
        center.pYZ, 
        centerPlus1Y.pYZ, centerPlus2Y.pYZ, centerPlus3Y.pYZ, 
        DY, order 
    ) + center.bX * differenceOperator(
        centerMinus3X.bZ, centerMinus2X.bZ, centerMinus1X.bZ, 
        center.bZ, 
        centerPlus1X.bZ, centerPlus2X.bZ, centerPlus3X.bZ, 
        DX, order 
    ) + center.bY * differenceOperator(
        centerMinus3Y.bZ, centerMinus2Y.bZ, centerMinus1Y.bZ, 
        center.bZ, 
        centerPlus1Y.bZ, centerPlus2Y.bZ, centerPlus3Y.bZ, 
        DY, order 
    ) + ((0.5 * (center.rho + centerPlus1X.rho) * fX.f3 + 0.5 * (center.w + centerPlus1X.w) * fX.f0)
       - (0.5 * (centerMinus1X.rho + center.rho) * fMinus1X.f3 + 0.5 * (centerMinus1X.w + center.w) * fMinus1X.f0)) / DX 
      + ((0.5 * (center.rho + centerPlus1Y.rho) * fY.f3 + 0.5 * (center.w + centerPlus1Y.w) * fY.f0)
       - (0.5 * (centerMinus1Y.rho + center.rho) * fMinus1Y.f3 + 0.5 * (centerMinus1Y.w + center.w) * fMinus1Y.f0)) / DY; 
    
    RHS.v4 = -differenceOperator(
        centerMinus3Y.v * centerMinus3Y.bX - centerMinus3Y.u * centerMinus3Y.bY,  
        centerMinus2Y.v * centerMinus2Y.bX - centerMinus2Y.u * centerMinus2Y.bY,  
        centerMinus1Y.v * centerMinus1Y.bX - centerMinus1Y.u * centerMinus1Y.bY,  
        center.v * center.bX - center.u * center.bY,  
        centerPlus1Y.v * centerPlus1Y.bX - centerPlus1Y.u * centerPlus1Y.bY,  
        centerPlus2Y.v * centerPlus2Y.bX - centerPlus2Y.u * centerPlus2Y.bY,  
        centerPlus3Y.v * centerPlus3Y.bX - centerPlus3Y.u * centerPlus3Y.bY,  
        DY, order 
    ) - differenceOperator(
        centerMinus3X.psi, centerMinus2X.psi, centerMinus1X.psi,   
        center.psi, 
        centerPlus1X.psi, centerPlus2X.psi, centerPlus3X.psi, 
        DX, order 
    ) + (fX.f4 - fMinus1X.f4) / DX 
      + (fY.f4 - fMinus1Y.f4) / DY;

    RHS.v5 = -differenceOperator(
        centerMinus3X.u * centerMinus3X.bY - centerMinus3X.v * centerMinus3X.bX,  
        centerMinus2X.u * centerMinus2X.bY - centerMinus2X.v * centerMinus2X.bX,  
        centerMinus1X.u * centerMinus1X.bY - centerMinus1X.v * centerMinus1X.bX,  
        center.u * center.bY - center.v * center.bX,  
        centerPlus1X.u * centerPlus1X.bY - centerPlus1X.v * centerPlus1X.bX,  
        centerPlus2X.u * centerPlus2X.bY - centerPlus2X.v * centerPlus2X.bX,  
        centerPlus3X.u * centerPlus3X.bY - centerPlus3X.v * centerPlus3X.bX,  
        DX, order 
    ) - differenceOperator(
        centerMinus3Y.psi, centerMinus2Y.psi, centerMinus1Y.psi,   
        center.psi, 
        centerPlus1Y.psi, centerPlus2Y.psi, centerPlus3Y.psi, 
        DY, order 
    ) + (fX.f5 - fMinus1X.f5) / DX 
      + (fY.f5 - fMinus1Y.f5) / DY;

    RHS.v6 = -differenceOperator(
        centerMinus3X.u * centerMinus3X.bZ - centerMinus3X.w * centerMinus3X.bX,  
        centerMinus2X.u * centerMinus2X.bZ - centerMinus2X.w * centerMinus2X.bX,  
        centerMinus1X.u * centerMinus1X.bZ - centerMinus1X.w * centerMinus1X.bX,  
        center.u * center.bZ - center.w * center.bX,  
        centerPlus1X.u * centerPlus1X.bZ - centerPlus1X.w * centerPlus1X.bX,  
        centerPlus2X.u * centerPlus2X.bZ - centerPlus2X.w * centerPlus2X.bX,  
        centerPlus3X.u * centerPlus3X.bZ - centerPlus3X.w * centerPlus3X.bX,  
        DX, order 
    ) - differenceOperator(
        centerMinus3Y.v * centerMinus3Y.bZ - centerMinus3Y.w * centerMinus3Y.bY,  
        centerMinus2Y.v * centerMinus2Y.bZ - centerMinus2Y.w * centerMinus2Y.bY,  
        centerMinus1Y.v * centerMinus1Y.bZ - centerMinus1Y.w * centerMinus1Y.bY,  
        center.v * center.bZ - center.w * center.bY,  
        centerPlus1Y.v * centerPlus1Y.bZ - centerPlus1Y.w * centerPlus1Y.bY,  
        centerPlus2Y.v * centerPlus2Y.bZ - centerPlus2Y.w * centerPlus2Y.bY,  
        centerPlus3Y.v * centerPlus3Y.bZ - centerPlus3Y.w * centerPlus3Y.bY,  
        DY, order 
    ) + (fX.f6 - fMinus1X.f6) / DX 
      + (fY.f6 - fMinus1Y.f6) / DY;

    RHS.v7 = -differenceOperator(
        centerMinus3X.pXX * centerMinus3X.u + 2.0 * centerMinus3X.pXX * centerMinus3X.u, 
        centerMinus2X.pXX * centerMinus2X.u + 2.0 * centerMinus2X.pXX * centerMinus2X.u, 
        centerMinus1X.pXX * centerMinus1X.u + 2.0 * centerMinus1X.pXX * centerMinus1X.u, 
        center.pXX * center.u + 2.0 * center.pXX * center.u, 
        centerPlus1X.pXX * centerPlus1X.u + 2.0 * centerPlus1X.pXX * centerPlus1X.u, 
        centerPlus2X.pXX * centerPlus2X.u + 2.0 * centerPlus2X.pXX * centerPlus2X.u, 
        centerPlus3X.pXX * centerPlus3X.u + 2.0 * centerPlus3X.pXX * centerPlus3X.u, 
        DX, order 
    ) - differenceOperator(
        centerMinus3Y.pXX * centerMinus3Y.v + 2.0 * centerMinus3Y.pXY * centerMinus3Y.u, 
        centerMinus2Y.pXX * centerMinus2Y.v + 2.0 * centerMinus2Y.pXY * centerMinus2Y.u, 
        centerMinus1Y.pXX * centerMinus1Y.v + 2.0 * centerMinus1Y.pXY * centerMinus1Y.u, 
        center.pXX * center.v + 2.0 * center.pXY * center.u, 
        centerPlus1Y.pXX * centerPlus1Y.v + 2.0 * centerPlus1Y.pXY * centerPlus1Y.u, 
        centerPlus2Y.pXX * centerPlus2Y.v + 2.0 * centerPlus2Y.pXY * centerPlus2Y.u, 
        centerPlus3Y.pXX * centerPlus3Y.v + 2.0 * centerPlus3Y.pXY * centerPlus3Y.u, 
        DY, order 
    ) + 2.0 * center.u * differenceOperator(
        centerMinus3X.pXX, centerMinus2X.pXX, centerMinus1X.pXX, 
        center.pXX, 
        centerPlus1X.pXX, centerPlus2X.pXX, centerPlus3X.pXX, 
        DX, order 
    ) + 2.0 * center.u * differenceOperator(
        centerMinus3Y.pXY, centerMinus2Y.pXY, centerMinus1Y.pXY, 
        center.pXY, 
        centerPlus1Y.pXY, centerPlus2Y.pXY, centerPlus3Y.pXY, 
        DY, order 
    ) + (fX.f7 - fMinus1X.f7) / DX 
      + (fY.f7 - fMinus1Y.f7) / DY
      + QVis.XX + QRes.XX + Q9Wave.XX;

    RHS.v8 = -differenceOperator(
        centerMinus3X.pYY * centerMinus3X.u + 2.0 * centerMinus3X.pXY * centerMinus3X.v, 
        centerMinus2X.pYY * centerMinus2X.u + 2.0 * centerMinus2X.pXY * centerMinus2X.v, 
        centerMinus1X.pYY * centerMinus1X.u + 2.0 * centerMinus1X.pXY * centerMinus1X.v, 
        center.pYY * center.u + 2.0 * center.pXY * center.v, 
        centerPlus1X.pYY * centerPlus1X.u + 2.0 * centerPlus1X.pXY * centerPlus1X.v, 
        centerPlus2X.pYY * centerPlus2X.u + 2.0 * centerPlus2X.pXY * centerPlus2X.v, 
        centerPlus3X.pYY * centerPlus3X.u + 2.0 * centerPlus3X.pXY * centerPlus3X.v, 
        DX, order 
    ) - differenceOperator(
        centerMinus3Y.pYY * centerMinus3Y.v + 2.0 * centerMinus3Y.pYY * centerMinus3Y.v, 
        centerMinus2Y.pYY * centerMinus2Y.v + 2.0 * centerMinus2Y.pYY * centerMinus2Y.v, 
        centerMinus1Y.pYY * centerMinus1Y.v + 2.0 * centerMinus1Y.pYY * centerMinus1Y.v, 
        center.pYY * center.v + 2.0 * center.pYY * center.v, 
        centerPlus1Y.pYY * centerPlus1Y.v + 2.0 * centerPlus1Y.pYY * centerPlus1Y.v, 
        centerPlus2Y.pYY * centerPlus2Y.v + 2.0 * centerPlus2Y.pYY * centerPlus2Y.v, 
        centerPlus3Y.pYY * centerPlus3Y.v + 2.0 * centerPlus3Y.pYY * centerPlus3Y.v, 
        DY, order 
    ) + 2.0 * center.v * differenceOperator(
        centerMinus3X.pXY, centerMinus2X.pXY, centerMinus1X.pXY, 
        center.pXY, 
        centerPlus1X.pXY, centerPlus2X.pXY, centerPlus3X.pXY, 
        DX, order 
    ) + 2.0 * center.v * differenceOperator(
        centerMinus3Y.pYY, centerMinus2Y.pYY, centerMinus1Y.pYY, 
        center.pYY, 
        centerPlus1Y.pYY, centerPlus2Y.pYY, centerPlus3Y.pYY, 
        DY, order 
    ) + (fX.f8 - fMinus1X.f8) / DX 
      + (fY.f8 - fMinus1Y.f8) / DY
      + QVis.YY + QRes.YY + Q9Wave.YY;
    
    RHS.v9 = -differenceOperator(
        centerMinus3X.pZZ * centerMinus3X.u + 2.0 * centerMinus3X.pXZ * centerMinus3X.w, 
        centerMinus2X.pZZ * centerMinus2X.u + 2.0 * centerMinus2X.pXZ * centerMinus2X.w, 
        centerMinus1X.pZZ * centerMinus1X.u + 2.0 * centerMinus1X.pXZ * centerMinus1X.w, 
        center.pZZ * center.u + 2.0 * center.pXZ * center.w, 
        centerPlus1X.pZZ * centerPlus1X.u + 2.0 * centerPlus1X.pXZ * centerPlus1X.w, 
        centerPlus2X.pZZ * centerPlus2X.u + 2.0 * centerPlus2X.pXZ * centerPlus2X.w, 
        centerPlus3X.pZZ * centerPlus3X.u + 2.0 * centerPlus3X.pXZ * centerPlus3X.w, 
        DX, order 
    ) - differenceOperator(
        centerMinus3Y.pZZ * centerMinus3Y.v + 2.0 * centerMinus3Y.pYZ * centerMinus3Y.w, 
        centerMinus2Y.pZZ * centerMinus2Y.v + 2.0 * centerMinus2Y.pYZ * centerMinus2Y.w, 
        centerMinus1Y.pZZ * centerMinus1Y.v + 2.0 * centerMinus1Y.pYZ * centerMinus1Y.w, 
        center.pZZ * center.v + 2.0 * center.pYZ * center.w, 
        centerPlus1Y.pZZ * centerPlus1Y.v + 2.0 * centerPlus1Y.pYZ * centerPlus1Y.w, 
        centerPlus2Y.pZZ * centerPlus2Y.v + 2.0 * centerPlus2Y.pYZ * centerPlus2Y.w, 
        centerPlus3Y.pZZ * centerPlus3Y.v + 2.0 * centerPlus3Y.pYZ * centerPlus3Y.w, 
        DY, order 
    ) + 2.0 * center.w * differenceOperator(
        centerMinus3X.pXZ, centerMinus2X.pXZ, centerMinus1X.pXZ, 
        center.pXZ, 
        centerPlus1X.pXZ, centerPlus2X.pXZ, centerPlus3X.pXZ, 
        DX, order 
    ) + 2.0 * center.w * differenceOperator(
        centerMinus3Y.pYZ, centerMinus2Y.pYZ, centerMinus1Y.pYZ, 
        center.pYZ, 
        centerPlus1Y.pYZ, centerPlus2Y.pYZ, centerPlus3Y.pYZ, 
        DY, order 
    ) + (fX.f9 - fMinus1X.f9) / DX 
      + (fY.f9 - fMinus1Y.f9) / DY
      + QVis.ZZ + QRes.ZZ + Q9Wave.ZZ;

    RHS.v10 = -differenceOperator(
        centerMinus3X.pXY * centerMinus3X.u + centerMinus3X.pXX * centerMinus3X.v + centerMinus3X.pXY * centerMinus3X.u, 
        centerMinus2X.pXY * centerMinus2X.u + centerMinus2X.pXX * centerMinus2X.v + centerMinus2X.pXY * centerMinus2X.u, 
        centerMinus1X.pXY * centerMinus1X.u + centerMinus1X.pXX * centerMinus1X.v + centerMinus1X.pXY * centerMinus1X.u, 
        center.pXY * center.u + center.pXX * center.v + center.pXY * center.u, 
        centerPlus1X.pXY * centerPlus1X.u + centerPlus1X.pXX * centerPlus1X.v + centerPlus1X.pXY * centerPlus1X.u, 
        centerPlus2X.pXY * centerPlus2X.u + centerPlus2X.pXX * centerPlus2X.v + centerPlus2X.pXY * centerPlus2X.u, 
        centerPlus3X.pXY * centerPlus3X.u + centerPlus3X.pXX * centerPlus3X.v + centerPlus3X.pXY * centerPlus3X.u, 
        DX, order 
    ) - differenceOperator(
        centerMinus3Y.pXY * centerMinus3Y.v + centerMinus3Y.pXY * centerMinus3Y.v + centerMinus3Y.pYY * centerMinus3Y.u, 
        centerMinus2Y.pXY * centerMinus2Y.v + centerMinus2Y.pXY * centerMinus2Y.v + centerMinus2Y.pYY * centerMinus2Y.u, 
        centerMinus1Y.pXY * centerMinus1Y.v + centerMinus1Y.pXY * centerMinus1Y.v + centerMinus1Y.pYY * centerMinus1Y.u, 
        center.pXY * center.v + center.pXY * center.v + center.pYY * center.u, 
        centerPlus1Y.pXY * centerPlus1Y.v + centerPlus1Y.pXY * centerPlus1Y.v + centerPlus1Y.pYY * centerPlus1Y.u, 
        centerPlus2Y.pXY * centerPlus2Y.v + centerPlus2Y.pXY * centerPlus2Y.v + centerPlus2Y.pYY * centerPlus2Y.u, 
        centerPlus3Y.pXY * centerPlus3Y.v + centerPlus3Y.pXY * centerPlus3Y.v + centerPlus3Y.pYY * centerPlus3Y.u, 
        DY, order 
    ) + center.u * differenceOperator(
        centerMinus3X.pXY, centerMinus2X.pXY, centerMinus1X.pXY, 
        center.pXY, 
        centerPlus1X.pXY, centerPlus2X.pXY, centerPlus3X.pXY, 
        DX, order 
    ) + center.u * differenceOperator(
        centerMinus3Y.pYY, centerMinus2Y.pYY, centerMinus1Y.pYY, 
        center.pYY, 
        centerPlus1Y.pYY, centerPlus2Y.pYY, centerPlus3Y.pYY, 
        DY, order 
    ) + center.v * differenceOperator(
        centerMinus3X.pXX, centerMinus2X.pXX, centerMinus1X.pXX, 
        center.pXX, 
        centerPlus1X.pXX, centerPlus2X.pXX, centerPlus3X.pXX, 
        DX, order 
    ) + center.v * differenceOperator(
        centerMinus3Y.pXY, centerMinus2Y.pXY, centerMinus1Y.pXY, 
        center.pXY, 
        centerPlus1Y.pXY, centerPlus2Y.pXY, centerPlus3Y.pXY, 
        DY, order 
    ) + (fX.f10 - fMinus1X.f10) / DX 
      + (fY.f10 - fMinus1Y.f10) / DY
      + QVis.XY + QRes.XY + Q9Wave.XY;

    RHS.v11 = -differenceOperator(
        centerMinus3X.pXZ * centerMinus3X.u + centerMinus3X.pXX * centerMinus3X.w + centerMinus3X.pXZ * centerMinus3X.u, 
        centerMinus2X.pXZ * centerMinus2X.u + centerMinus2X.pXX * centerMinus2X.w + centerMinus2X.pXZ * centerMinus2X.u, 
        centerMinus1X.pXZ * centerMinus1X.u + centerMinus1X.pXX * centerMinus1X.w + centerMinus1X.pXZ * centerMinus1X.u, 
        center.pXZ * center.u + center.pXX * center.w + center.pXZ * center.u, 
        centerPlus1X.pXZ * centerPlus1X.u + centerPlus1X.pXX * centerPlus1X.w + centerPlus1X.pXZ * centerPlus1X.u, 
        centerPlus2X.pXZ * centerPlus2X.u + centerPlus2X.pXX * centerPlus2X.w + centerPlus2X.pXZ * centerPlus2X.u, 
        centerPlus3X.pXZ * centerPlus3X.u + centerPlus3X.pXX * centerPlus3X.w + centerPlus3X.pXZ * centerPlus3X.u, 
        DX, order 
    ) - differenceOperator(
        centerMinus3Y.pXZ * centerMinus3Y.v + centerMinus3Y.pXY * centerMinus3Y.w + centerMinus3Y.pYZ * centerMinus3Y.u, 
        centerMinus2Y.pXZ * centerMinus2Y.v + centerMinus2Y.pXY * centerMinus2Y.w + centerMinus2Y.pYZ * centerMinus2Y.u, 
        centerMinus1Y.pXZ * centerMinus1Y.v + centerMinus1Y.pXY * centerMinus1Y.w + centerMinus1Y.pYZ * centerMinus1Y.u, 
        center.pXZ * center.v + center.pXY * center.w + center.pYZ * center.u, 
        centerPlus1Y.pXZ * centerPlus1Y.v + centerPlus1Y.pXY * centerPlus1Y.w + centerPlus1Y.pYZ * centerPlus1Y.u, 
        centerPlus2Y.pXZ * centerPlus2Y.v + centerPlus2Y.pXY * centerPlus2Y.w + centerPlus2Y.pYZ * centerPlus2Y.u, 
        centerPlus3Y.pXZ * centerPlus3Y.v + centerPlus3Y.pXY * centerPlus3Y.w + centerPlus3Y.pYZ * centerPlus3Y.u, 
        DY, order 
    ) + center.u * differenceOperator(
        centerMinus3X.pXZ, centerMinus2X.pXZ, centerMinus1X.pXZ, 
        center.pXZ, 
        centerPlus1X.pXZ, centerPlus2X.pXZ, centerPlus3X.pXZ, 
        DX, order 
    ) + center.u * differenceOperator(
        centerMinus3Y.pYZ, centerMinus2Y.pYZ, centerMinus1Y.pYZ, 
        center.pYZ, 
        centerPlus1Y.pYZ, centerPlus2Y.pYZ, centerPlus3Y.pYZ, 
        DY, order 
    ) + center.w * differenceOperator(
        centerMinus3X.pXX, centerMinus2X.pXX, centerMinus1X.pXX, 
        center.pXX, 
        centerPlus1X.pXX, centerPlus2X.pXX, centerPlus3X.pXX, 
        DX, order 
    ) + center.w * differenceOperator(
        centerMinus3Y.pXY, centerMinus2Y.pXY, centerMinus1Y.pXY, 
        center.pXY, 
        centerPlus1Y.pXY, centerPlus2Y.pXY, centerPlus3Y.pXY, 
        DY, order 
    ) + (fX.f11 - fMinus1X.f11) / DX 
      + (fY.f11 - fMinus1Y.f11) / DY
      + QVis.XZ + QRes.XZ + Q9Wave.XZ;

    RHS.v12 = -differenceOperator(
        centerMinus3X.pYZ * centerMinus3X.u + centerMinus3X.pXY * centerMinus3X.w + centerMinus3X.pXZ * centerMinus3X.v, 
        centerMinus2X.pYZ * centerMinus2X.u + centerMinus2X.pXY * centerMinus2X.w + centerMinus2X.pXZ * centerMinus2X.v, 
        centerMinus1X.pYZ * centerMinus1X.u + centerMinus1X.pXY * centerMinus1X.w + centerMinus1X.pXZ * centerMinus1X.v, 
        center.pYZ * center.u + center.pXY * center.w + center.pXZ * center.v, 
        centerPlus1X.pYZ * centerPlus1X.u + centerPlus1X.pXY * centerPlus1X.w + centerPlus1X.pXZ * centerPlus1X.v, 
        centerPlus2X.pYZ * centerPlus2X.u + centerPlus2X.pXY * centerPlus2X.w + centerPlus2X.pXZ * centerPlus2X.v, 
        centerPlus3X.pYZ * centerPlus3X.u + centerPlus3X.pXY * centerPlus3X.w + centerPlus3X.pXZ * centerPlus3X.v, 
        DX, order 
    ) - differenceOperator(
        centerMinus3Y.pYZ * centerMinus3Y.v + centerMinus3Y.pYY * centerMinus3Y.w + centerMinus3Y.pYZ * centerMinus3Y.v, 
        centerMinus2Y.pYZ * centerMinus2Y.v + centerMinus2Y.pYY * centerMinus2Y.w + centerMinus2Y.pYZ * centerMinus2Y.v, 
        centerMinus1Y.pYZ * centerMinus1Y.v + centerMinus1Y.pYY * centerMinus1Y.w + centerMinus1Y.pYZ * centerMinus1Y.v, 
        center.pYZ * center.v + center.pYY * center.w + center.pYZ * center.v, 
        centerPlus1Y.pYZ * centerPlus1Y.v + centerPlus1Y.pYY * centerPlus1Y.w + centerPlus1Y.pYZ * centerPlus1Y.v, 
        centerPlus2Y.pYZ * centerPlus2Y.v + centerPlus2Y.pYY * centerPlus2Y.w + centerPlus2Y.pYZ * centerPlus2Y.v, 
        centerPlus3Y.pYZ * centerPlus3Y.v + centerPlus3Y.pYY * centerPlus3Y.w + centerPlus3Y.pYZ * centerPlus3Y.v, 
        DY, order 
    ) + center.v * differenceOperator(
        centerMinus3X.pXZ, centerMinus2X.pXZ, centerMinus1X.pXZ, 
        center.pXZ, 
        centerPlus1X.pXZ, centerPlus2X.pXZ, centerPlus3X.pXZ, 
        DX, order 
    ) + center.v * differenceOperator(
        centerMinus3Y.pYZ, centerMinus2Y.pYZ, centerMinus1Y.pYZ, 
        center.pYZ, 
        centerPlus1Y.pYZ, centerPlus2Y.pYZ, centerPlus3Y.pYZ, 
        DY, order 
    ) + center.w * differenceOperator(
        centerMinus3X.pXY, centerMinus2X.pXY, centerMinus1X.pXY, 
        center.pXY, 
        centerPlus1X.pXY, centerPlus2X.pXY, centerPlus3X.pXY, 
        DX, order 
    ) + center.w * differenceOperator(
        centerMinus3Y.pYY, centerMinus2Y.pYY, centerMinus1Y.pYY, 
        center.pYY, 
        centerPlus1Y.pYY, centerPlus2Y.pYY, centerPlus3Y.pYY, 
        DY, order 
    ) + (fX.f12 - fMinus1X.f12) / DX 
      + (fY.f12 - fMinus1Y.f12) / DY
      + QVis.YZ + QRes.YZ + Q9Wave.YZ;

    RHS.v13 = -cPsi * cPsi * (differenceOperator(
        centerMinus3X.bX, centerMinus2X.bX, centerMinus1X.bX, 
        center.bX, 
        centerPlus1X.bX, centerPlus2X.bX, centerPlus3X.bX, 
        DX, order 
    ) + differenceOperator(
        centerMinus3Y.bY, centerMinus2Y.bY, centerMinus1Y.bY, 
        center.bY, 
        centerPlus1Y.bY, centerPlus2Y.bY, centerPlus3Y.bY, 
        DY, order 
    )) - center.psi / tauPsi 
      + (fX.f13 - fMinus1X.f13) / DX 
      + (fY.f13 - fMinus1Y.f13) / DY; 

    return RHS; 
} 


__device__ void applyIsotropization(
    MHDFloat& dpXX, MHDFloat& dpYY, MHDFloat& dpZZ,
    MHDFloat& dpXY, MHDFloat& dpXZ, MHDFloat& dpYZ,
    const MHDFloat& pXX, const MHDFloat& pYY, const MHDFloat& pZZ,
    const MHDFloat& pXY, const MHDFloat& pXZ, const MHDFloat& pYZ,
    const MHDFloat bX, const MHDFloat bY, const MHDFloat bZ,
    const MHDFloat NUG_COEF, const MHDFloat DT, 
    const MHDFloat EPS, 
    const bool ACTIVATE_ISOTROPIC_EFFECT, 
    const bool ACTIVATE_GYROTROPIC_EFFECT
)
{
    MHDFloat B = sqrt(bX * bX + bY * bY + bZ * bZ);
    
    if (B < EPS) return;

    //isotropization to gyrotropic MHD
    MHDFloat bx = bX / B;
    MHDFloat by = bY / B;
    MHDFloat bz = bZ / B;

    MHDFloat pParallel = pXX * bx * bx
                        + pYY * by * by
                        + pZZ * bz * bz
                        + 2.0 * pXY * bx * by
                        + 2.0 * pXZ * bx * bz
                        + 2.0 * pYZ * by * bz;

    MHDFloat trP   = pXX + pYY + pZZ;
    MHDFloat pPerp = (trP - pParallel) / 2.0;

    MHDFloat pgXX, pgYY, pgZZ, pgXY, pgXZ, pgYZ; 
    if (ACTIVATE_ISOTROPIC_EFFECT) {
        pgXX = trP / 3.0; 
        pgYY = pgXX; 
        pgZZ = pgXX; 
        pgXY = 0.0; 
        pgXZ = 0.0; 
        pgYZ = 0.0;
    }
    if (ACTIVATE_GYROTROPIC_EFFECT) {
        pgXX = pPerp + (pParallel - pPerp) * bx * bx;
        pgYY = pPerp + (pParallel - pPerp) * by * by;
        pgZZ = pPerp + (pParallel - pPerp) * bz * bz;
        pgXY =         (pParallel - pPerp) * bx * by;
        pgXZ =         (pParallel - pPerp) * bx * bz;
        pgYZ =         (pParallel - pPerp) * by * bz;
    } 

    const MHDFloat NUG = NUG_COEF * B; //|B|に比例するモデル

    dpXX = (1.0 - exp(-NUG * DT)) * (pgXX - pXX);
    dpYY = (1.0 - exp(-NUG * DT)) * (pgYY - pYY);
    dpZZ = (1.0 - exp(-NUG * DT)) * (pgZZ - pZZ);
    dpXY = (1.0 - exp(-NUG * DT)) * (pgXY - pXY);
    dpXZ = (1.0 - exp(-NUG * DT)) * (pgXZ - pXZ);
    dpYZ = (1.0 - exp(-NUG * DT)) * (pgYZ - pYZ);
}


__global__ static void firstStep_kernel(
    MHDValue* U1, 
    const MHDValue* UPast, 
    const MHDValue* center, 
    const FilteringFluxValue* filteringFluxX, const FilteringFluxValue* filteringFluxY, 
    const SourceValue* source, 
    const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
    const MHDUnsignedInt BUFFER, 
    const MHDFloat DX, const MHDFloat DY, 
    const MHDFloat DT, 
    const MHDFloat CH, const MHDFloat CP, 
    const MHDFloat NUG_COEF, 
    const MHDFloat EPS, 
    const MHDInt order, 
    const bool ACTIVATE_ISOTROPIC_EFFECT, 
    const bool ACTIVATE_GYROTROPIC_EFFECT
)
{
    MHDUnsignedInt i = blockIdx.x * blockDim.x + threadIdx.x;
    MHDUnsignedInt j = blockIdx.y * blockDim.y + threadIdx.y;

    if (BUFFER <= i && i < NX - BUFFER && BUFFER <= j && j < NY - BUFFER) {
        MHDUnsignedLongLong index = getIndex<MHDUnsignedLongLong>(i, j, NX, NY);

        HeatingValue QVis = calculateViscousHeatingTerm(
            center, 
            filteringFluxX, filteringFluxY, 
            NX, NY, DX, DY, index 
        );
        HeatingValue QRes = calculateResistiveHeatingTerm(
            center, 
            filteringFluxX, filteringFluxY, 
            NX, NY, DX, DY, index
        ); 
        HeatingValue Q9Wave = calculate9WaveHeatingTerm(
            center, 
            NX, NY, DX, DY, index, order
        );
        
        const MHDFloat cPsi = CH; 
        const MHDFloat tauPsi = CP * CP / (CH * CH);
        RHSValue RHS = calculateRHSValue(
            center, 
            filteringFluxX, filteringFluxY,  
            QVis, QRes, Q9Wave, 
            NX, NY, DX, DY, 
            cPsi, tauPsi, 
            index, order
        );

        MHDFloat rho  = UPast[index].rho                  + DT * (RHS.v0 + source[index].s0); 
        MHDFloat rhoU = UPast[index].rho * UPast[index].u + DT * (RHS.v1 + source[index].s1); 
        MHDFloat rhoV = UPast[index].rho * UPast[index].v + DT * (RHS.v2 + source[index].s2); 
        MHDFloat rhoW = UPast[index].rho * UPast[index].w + DT * (RHS.v3 + source[index].s3); 
        MHDFloat bX   = UPast[index].bX                   + DT * (RHS.v4 + source[index].s4); 
        MHDFloat bY   = UPast[index].bY                   + DT * (RHS.v5 + source[index].s5); 
        MHDFloat bZ   = UPast[index].bZ                   + DT * (RHS.v6 + source[index].s6); 
        MHDFloat pXX  = UPast[index].pXX                  + DT * (RHS.v7 + source[index].s7); 
        MHDFloat pYY  = UPast[index].pYY                  + DT * (RHS.v8 + source[index].s8); 
        MHDFloat pZZ  = UPast[index].pZZ                  + DT * (RHS.v9 + source[index].s9); 
        MHDFloat pXY  = UPast[index].pXY                  + DT * (RHS.v10 + source[index].s10); 
        MHDFloat pXZ  = UPast[index].pXZ                  + DT * (RHS.v11 + source[index].s11); 
        MHDFloat pYZ  = UPast[index].pYZ                  + DT * (RHS.v12 + source[index].s12); 
        MHDFloat psi  = UPast[index].psi                  + DT * (RHS.v13 + source[index].s13); 

        MHDFloat dpXX = 0.0, dpYY = 0.0, dpZZ = 0.0, dpXY = 0.0, dpXZ = 0.0, dpYZ = 0.0; 
        applyIsotropization(
            dpXX, dpYY, dpZZ,
            dpXY, dpXZ, dpYZ,
            pXX, pYY, pZZ,
            pXY, pXZ, pYZ,
            bX, bY, bZ,
            NUG_COEF, DT, 
            EPS, 
            ACTIVATE_ISOTROPIC_EFFECT, 
            ACTIVATE_GYROTROPIC_EFFECT
        ); 
        pXX += dpXX; 
        pYY += dpYY; 
        pZZ += dpZZ; 
        pXY += dpXY; 
        pXZ += dpXZ; 
        pYZ += dpYZ; 

        U1[index].rho = rho; 
        U1[index].u   = rhoU / rho; 
        U1[index].v   = rhoV / rho; 
        U1[index].w   = rhoW / rho; 
        U1[index].bX  = bX; 
        U1[index].bY  = bY; 
        U1[index].bZ  = bZ; 
        U1[index].pXX = pXX; 
        U1[index].pYY = pYY; 
        U1[index].pZZ = pZZ; 
        U1[index].pXY = pXY; 
        U1[index].pXZ = pXZ; 
        U1[index].pYZ = pYZ; 
        U1[index].psi = psi; 
    }
}


__global__ static void secondStep_kernel(
    MHDValue* U2, 
    const MHDValue* U1, 
    const MHDValue* UPast, 
    const MHDValue* center, 
    const FilteringFluxValue* filteringFluxX, const FilteringFluxValue* filteringFluxY,  
    const SourceValue* source, 
    const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
    const MHDUnsignedInt BUFFER, 
    const MHDFloat DX, const MHDFloat DY,  
    const MHDFloat DT, 
    const MHDFloat CH, const MHDFloat CP, 
    const MHDFloat NUG_COEF, 
    const MHDFloat EPS, 
    const MHDInt order, 
    const bool ACTIVATE_ISOTROPIC_EFFECT, 
    const bool ACTIVATE_GYROTROPIC_EFFECT 
)
{
    MHDUnsignedInt i = blockIdx.x * blockDim.x + threadIdx.x;
    MHDUnsignedInt j = blockIdx.y * blockDim.y + threadIdx.y;

    if (BUFFER <= i && i < NX - BUFFER && BUFFER <= j && j < NY - BUFFER) {
        MHDUnsignedLongLong index = getIndex<MHDUnsignedLongLong>(i, j, NX, NY);

        HeatingValue QVis = calculateViscousHeatingTerm(
            center, 
            filteringFluxX, filteringFluxY,  
            NX, NY, DX, DY, index 
        );
        HeatingValue QRes = calculateResistiveHeatingTerm(
            center, 
            filteringFluxX, filteringFluxY,  
            NX, NY, DX, DY, index
        ); 
        HeatingValue Q9Wave = calculate9WaveHeatingTerm(
            center, 
            NX, NY, DX, DY, index, order
        );
        
        const MHDFloat cPsi = CH; 
        const MHDFloat tauPsi = CP * CP / (CH * CH);
        RHSValue RHS = calculateRHSValue(
            center, 
            filteringFluxX, filteringFluxY,  
            QVis, QRes, Q9Wave, 
            NX, NY, DX, DY, 
            cPsi, tauPsi, 
            index, order
        );

        MHDFloat rho  = 3.0 / 4.0 * UPast[index].rho                  + 1.0 / 4.0 * (U1[index].rho               + DT * (RHS.v0 + source[index].s0)); 
        MHDFloat rhoU = 3.0 / 4.0 * UPast[index].rho * UPast[index].u + 1.0 / 4.0 * (U1[index].rho * U1[index].u + DT * (RHS.v1 + source[index].s1)); 
        MHDFloat rhoV = 3.0 / 4.0 * UPast[index].rho * UPast[index].v + 1.0 / 4.0 * (U1[index].rho * U1[index].v + DT * (RHS.v2 + source[index].s2)); 
        MHDFloat rhoW = 3.0 / 4.0 * UPast[index].rho * UPast[index].w + 1.0 / 4.0 * (U1[index].rho * U1[index].w + DT * (RHS.v3 + source[index].s3)); 
        MHDFloat bX   = 3.0 / 4.0 * UPast[index].bX                   + 1.0 / 4.0 * (U1[index].bX                + DT * (RHS.v4 + source[index].s4)); 
        MHDFloat bY   = 3.0 / 4.0 * UPast[index].bY                   + 1.0 / 4.0 * (U1[index].bY                + DT * (RHS.v5 + source[index].s5)); 
        MHDFloat bZ   = 3.0 / 4.0 * UPast[index].bZ                   + 1.0 / 4.0 * (U1[index].bZ                + DT * (RHS.v6 + source[index].s6)); 
        MHDFloat pXX  = 3.0 / 4.0 * UPast[index].pXX                  + 1.0 / 4.0 * (U1[index].pXX               + DT * (RHS.v7 + source[index].s7)); 
        MHDFloat pYY  = 3.0 / 4.0 * UPast[index].pYY                  + 1.0 / 4.0 * (U1[index].pYY               + DT * (RHS.v8 + source[index].s8)); 
        MHDFloat pZZ  = 3.0 / 4.0 * UPast[index].pZZ                  + 1.0 / 4.0 * (U1[index].pZZ               + DT * (RHS.v9 + source[index].s9)); 
        MHDFloat pXY  = 3.0 / 4.0 * UPast[index].pXY                  + 1.0 / 4.0 * (U1[index].pXY               + DT * (RHS.v10 + source[index].s10)); 
        MHDFloat pXZ  = 3.0 / 4.0 * UPast[index].pXZ                  + 1.0 / 4.0 * (U1[index].pXZ               + DT * (RHS.v11 + source[index].s11)); 
        MHDFloat pYZ  = 3.0 / 4.0 * UPast[index].pYZ                  + 1.0 / 4.0 * (U1[index].pYZ               + DT * (RHS.v12 + source[index].s12)); 
        MHDFloat psi  = 3.0 / 4.0 * UPast[index].psi                  + 1.0 / 4.0 * (U1[index].psi               + DT * (RHS.v13 + source[index].s13)); 

        MHDFloat dpXX = 0.0, dpYY = 0.0, dpZZ = 0.0, dpXY = 0.0, dpXZ = 0.0, dpYZ = 0.0; 
        applyIsotropization(
            dpXX, dpYY, dpZZ,
            dpXY, dpXZ, dpYZ,
            pXX, pYY, pZZ,
            pXY, pXZ, pYZ,
            bX, bY, bZ,
            NUG_COEF, DT, 
            EPS, 
            ACTIVATE_ISOTROPIC_EFFECT, 
            ACTIVATE_GYROTROPIC_EFFECT
        ); 
        pXX += dpXX; 
        pYY += dpYY; 
        pZZ += dpZZ; 
        pXY += dpXY; 
        pXZ += dpXZ; 
        pYZ += dpYZ; 

        U2[index].rho = rho; 
        U2[index].u   = rhoU / rho; 
        U2[index].v   = rhoV / rho; 
        U2[index].w   = rhoW / rho; 
        U2[index].bX  = bX; 
        U2[index].bY  = bY; 
        U2[index].bZ  = bZ; 
        U2[index].pXX = pXX; 
        U2[index].pYY = pYY; 
        U2[index].pZZ = pZZ; 
        U2[index].pXY = pXY; 
        U2[index].pXZ = pXZ; 
        U2[index].pYZ = pYZ; 
        U2[index].psi = psi; 
    }
}


__global__ static void thirdStep_kernel(
    MHDValue* U, 
    const MHDValue* U2, 
    const MHDValue* UPast, 
    const MHDValue* center, 
    const FilteringFluxValue* filteringFluxX, const FilteringFluxValue* filteringFluxY, 
    const SourceValue* source, 
    const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
    const MHDUnsignedInt BUFFER, 
    const MHDFloat DX, const MHDFloat DY,
    const MHDFloat DT, 
    const MHDFloat CH, const MHDFloat CP, 
    const MHDFloat NUG_COEF, 
    const MHDFloat EPS, 
    const MHDInt order, 
    const bool ACTIVATE_ISOTROPIC_EFFECT, 
    const bool ACTIVATE_GYROTROPIC_EFFECT 
)
{
    MHDUnsignedInt i = blockIdx.x * blockDim.x + threadIdx.x;
    MHDUnsignedInt j = blockIdx.y * blockDim.y + threadIdx.y;

    if (BUFFER <= i && i < NX - BUFFER && BUFFER <= j && j < NY - BUFFER) {
        MHDUnsignedLongLong index = getIndex<MHDUnsignedLongLong>(i, j, NX, NY);

        HeatingValue QVis = calculateViscousHeatingTerm(
            center, 
            filteringFluxX, filteringFluxY,  
            NX, NY, DX, DY, index 
        );
        HeatingValue QRes = calculateResistiveHeatingTerm(
            center, 
            filteringFluxX, filteringFluxY,  
            NX, NY, DX, DY, index
        ); 
        HeatingValue Q9Wave = calculate9WaveHeatingTerm(
            center, 
            NX, NY, DX, DY, index, order
        );
        
        const MHDFloat cPsi = CH; 
        const MHDFloat tauPsi = CP * CP / (CH * CH);
        RHSValue RHS = calculateRHSValue(
            center, 
            filteringFluxX, filteringFluxY,  
            QVis, QRes, Q9Wave, 
            NX, NY, DX, DY, 
            cPsi, tauPsi, 
            index, order
        );

        MHDFloat rho  = 1.0 / 3.0 * UPast[index].rho                  + 2.0 / 3.0 * (U2[index].rho               + DT * (RHS.v0 + source[index].s0)); 
        MHDFloat rhoU = 1.0 / 3.0 * UPast[index].rho * UPast[index].u + 2.0 / 3.0 * (U2[index].rho * U2[index].u + DT * (RHS.v1 + source[index].s1)); 
        MHDFloat rhoV = 1.0 / 3.0 * UPast[index].rho * UPast[index].v + 2.0 / 3.0 * (U2[index].rho * U2[index].v + DT * (RHS.v2 + source[index].s2)); 
        MHDFloat rhoW = 1.0 / 3.0 * UPast[index].rho * UPast[index].w + 2.0 / 3.0 * (U2[index].rho * U2[index].w + DT * (RHS.v3 + source[index].s3)); 
        MHDFloat bX   = 1.0 / 3.0 * UPast[index].bX                   + 2.0 / 3.0 * (U2[index].bX                + DT * (RHS.v4 + source[index].s4)); 
        MHDFloat bY   = 1.0 / 3.0 * UPast[index].bY                   + 2.0 / 3.0 * (U2[index].bY                + DT * (RHS.v5 + source[index].s5)); 
        MHDFloat bZ   = 1.0 / 3.0 * UPast[index].bZ                   + 2.0 / 3.0 * (U2[index].bZ                + DT * (RHS.v6 + source[index].s6)); 
        MHDFloat pXX  = 1.0 / 3.0 * UPast[index].pXX                  + 2.0 / 3.0 * (U2[index].pXX               + DT * (RHS.v7 + source[index].s7)); 
        MHDFloat pYY  = 1.0 / 3.0 * UPast[index].pYY                  + 2.0 / 3.0 * (U2[index].pYY               + DT * (RHS.v8 + source[index].s8)); 
        MHDFloat pZZ  = 1.0 / 3.0 * UPast[index].pZZ                  + 2.0 / 3.0 * (U2[index].pZZ               + DT * (RHS.v9 + source[index].s9)); 
        MHDFloat pXY  = 1.0 / 3.0 * UPast[index].pXY                  + 2.0 / 3.0 * (U2[index].pXY               + DT * (RHS.v10 + source[index].s10)); 
        MHDFloat pXZ  = 1.0 / 3.0 * UPast[index].pXZ                  + 2.0 / 3.0 * (U2[index].pXZ               + DT * (RHS.v11 + source[index].s11)); 
        MHDFloat pYZ  = 1.0 / 3.0 * UPast[index].pYZ                  + 2.0 / 3.0 * (U2[index].pYZ               + DT * (RHS.v12 + source[index].s12)); 
        MHDFloat psi  = 1.0 / 3.0 * UPast[index].psi                  + 2.0 / 3.0 * (U2[index].psi               + DT * (RHS.v13 + source[index].s13)); 

        MHDFloat dpXX = 0.0, dpYY = 0.0, dpZZ = 0.0, dpXY = 0.0, dpXZ = 0.0, dpYZ = 0.0; 
        applyIsotropization(
            dpXX, dpYY, dpZZ,
            dpXY, dpXZ, dpYZ,
            pXX, pYY, pZZ,
            pXY, pXZ, pYZ,
            bX, bY, bZ,
            NUG_COEF, DT, 
            EPS, 
            ACTIVATE_ISOTROPIC_EFFECT, 
            ACTIVATE_GYROTROPIC_EFFECT
        ); 
        pXX += dpXX; 
        pYY += dpYY; 
        pZZ += dpZZ; 
        pXY += dpXY; 
        pXZ += dpXZ; 
        pYZ += dpYZ; 

        U[index].rho = rho; 
        U[index].u   = rhoU / rho; 
        U[index].v   = rhoV / rho; 
        U[index].w   = rhoW / rho; 
        U[index].bX  = bX; 
        U[index].bY  = bY; 
        U[index].bZ  = bZ; 
        U[index].pXX = pXX; 
        U[index].pYY = pYY; 
        U[index].pZZ = pZZ; 
        U[index].pXY = pXY; 
        U[index].pXZ = pXZ; 
        U[index].pYZ = pYZ; 
        U[index].psi = psi; 
    }
}


void SSPRK3::push(
    const MHDFloat DT
) 
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (NY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    mHDConstParameter.CH = 0.1 * min(DX, DY) / DT; 
    mHDConstParameter.CP = sqrt(mHDConstParameter.CR * mHDConstParameter.CH);

    // STEP1
    
    sourceTermCalculator.calculateSourceTerm(UPast); 
    MHDFloat maxSpeed = min(DX, DY) / DT; 
    calculateFilteringFlux(UPast, maxSpeed); 
    firstStep_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U1.data()), 
        thrust::raw_pointer_cast(UPast.data()), 
        thrust::raw_pointer_cast(reconstructor->getCenterMHDValueRef().data()), 
        thrust::raw_pointer_cast(filteringFluxX.data()), 
        thrust::raw_pointer_cast(filteringFluxY.data()),   
        thrust::raw_pointer_cast(sourceTermCalculator.getSourceRef().data()), 
        NX, NY,
        mHDGridParameter.BUFFER,   
        DX, DY,  
        DT, 
        mHDConstParameter.CH, mHDConstParameter.CP, 
        mHDConstParameter.NUG_COEF, mHDConstParameter.EPS,  
        order, 
        mHDConstParameter.ACTIVATE_ISOTROPIC_EFFECT, 
        mHDConstParameter.ACTIVATE_GYROTROPIC_EFFECT 
    );
    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess) {
        printf("Kernel launch failed at firstStep_kernel: %s\n", cudaGetErrorString(err1));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err1 = cudaDeviceSynchronize();
    if (err1 != cudaSuccess) {
        printf("Kernel execution failed at firstStep_kernel: %s\n", cudaGetErrorString(err1));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    boundary->applyUForAllDirection(U1);

    // STEP 2

    sourceTermCalculator.calculateSourceTerm(U1); 
    calculateFilteringFlux(U1, maxSpeed); 
    secondStep_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U2.data()), 
        thrust::raw_pointer_cast(U1.data()), 
        thrust::raw_pointer_cast(UPast.data()), 
        thrust::raw_pointer_cast(reconstructor->getCenterMHDValueRef().data()), 
        thrust::raw_pointer_cast(filteringFluxX.data()), 
        thrust::raw_pointer_cast(filteringFluxY.data()),   
        thrust::raw_pointer_cast(sourceTermCalculator.getSourceRef().data()), 
        NX, NY,
        mHDGridParameter.BUFFER,   
        DX, DY,  
        DT, 
        mHDConstParameter.CH, mHDConstParameter.CP, 
        mHDConstParameter.NUG_COEF, mHDConstParameter.EPS,  
        order, 
        mHDConstParameter.ACTIVATE_ISOTROPIC_EFFECT, 
        mHDConstParameter.ACTIVATE_GYROTROPIC_EFFECT 
    );
    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess) {
        printf("Kernel launch failed at secondStep_kernel: %s\n", cudaGetErrorString(err2));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err2 = cudaDeviceSynchronize();
    if (err2 != cudaSuccess) {
        printf("Kernel execution failed at secondStep_kernel: %s\n", cudaGetErrorString(err2));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    boundary->applyUForAllDirection(U2);

    // STEP3 

    sourceTermCalculator.calculateSourceTerm(U2); 
    calculateFilteringFlux(U2, maxSpeed); 
    thirdStep_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(U2.data()), 
        thrust::raw_pointer_cast(UPast.data()), 
        thrust::raw_pointer_cast(reconstructor->getCenterMHDValueRef().data()), 
        thrust::raw_pointer_cast(filteringFluxX.data()), 
        thrust::raw_pointer_cast(filteringFluxY.data()),   
        thrust::raw_pointer_cast(sourceTermCalculator.getSourceRef().data()), 
        NX, NY,
        mHDGridParameter.BUFFER,   
        DX, DY,  
        DT, 
        mHDConstParameter.CH, mHDConstParameter.CP, 
        mHDConstParameter.NUG_COEF, mHDConstParameter.EPS,  
        order, 
        mHDConstParameter.ACTIVATE_ISOTROPIC_EFFECT, 
        mHDConstParameter.ACTIVATE_GYROTROPIC_EFFECT 
    );
    cudaError_t err3 = cudaGetLastError();
    if (err3 != cudaSuccess) {
        printf("Kernel launch failed at thirdStep_kernel: %s\n", cudaGetErrorString(err3));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err3 = cudaDeviceSynchronize();
    if (err3 != cudaSuccess) {
        printf("Kernel execution failed at thirdStep_kernel: %s\n", cudaGetErrorString(err3));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    boundary->applyUForAllDirection(U);

}


void SSPRK3::push(
    const thrust::device_vector<MHDValue>& coarseUPast, 
    const thrust::device_vector<MHDValue>& coarseUNext, 
    const MHDFloat DT, 
    const MHDInt substep
) 
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (NY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    mHDConstParameter.CH = 0.1 * min(DX, DY) / DT; 
    mHDConstParameter.CP = sqrt(mHDConstParameter.CR * mHDConstParameter.CH);

    // STEP1
    
    sourceTermCalculator.calculateSourceTerm(UPast); 
    MHDFloat maxSpeed = min(DX, DY) / DT; 
    calculateFilteringFlux(UPast, maxSpeed); 
    firstStep_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U1.data()), 
        thrust::raw_pointer_cast(UPast.data()), 
        thrust::raw_pointer_cast(reconstructor->getCenterMHDValueRef().data()), 
        thrust::raw_pointer_cast(filteringFluxX.data()), 
        thrust::raw_pointer_cast(filteringFluxY.data()),   
        thrust::raw_pointer_cast(sourceTermCalculator.getSourceRef().data()), 
        NX, NY,
        mHDGridParameter.BUFFER,   
        DX, DY,  
        DT, 
        mHDConstParameter.CH, mHDConstParameter.CP, 
        mHDConstParameter.NUG_COEF, mHDConstParameter.EPS,  
        order, 
        mHDConstParameter.ACTIVATE_ISOTROPIC_EFFECT, 
        mHDConstParameter.ACTIVATE_GYROTROPIC_EFFECT 
    );
    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess) {
        printf("Kernel launch failed at firstStep_kernel: %s\n", cudaGetErrorString(err1));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err1 = cudaDeviceSynchronize();
    if (err1 != cudaSuccess) {
        printf("Kernel execution failed at firstStep_kernel: %s\n", cudaGetErrorString(err1));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    const MHDFloat timeRatio1 = 2.0 * (substep + 1.0) / 4.0;  
    smrBoundary->applyUForAllDirection(
        coarseUPast, coarseUNext, 
        timeRatio1, 
        U1
    );

    // STEP 2

    sourceTermCalculator.calculateSourceTerm(U1); 
    calculateFilteringFlux(U1, maxSpeed); 
    secondStep_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U2.data()), 
        thrust::raw_pointer_cast(U1.data()), 
        thrust::raw_pointer_cast(UPast.data()), 
        thrust::raw_pointer_cast(reconstructor->getCenterMHDValueRef().data()), 
        thrust::raw_pointer_cast(filteringFluxX.data()), 
        thrust::raw_pointer_cast(filteringFluxY.data()),   
        thrust::raw_pointer_cast(sourceTermCalculator.getSourceRef().data()), 
        NX, NY,
        mHDGridParameter.BUFFER,   
        DX, DY,  
        DT, 
        mHDConstParameter.CH, mHDConstParameter.CP, 
        mHDConstParameter.NUG_COEF, mHDConstParameter.EPS,  
        order, 
        mHDConstParameter.ACTIVATE_ISOTROPIC_EFFECT, 
        mHDConstParameter.ACTIVATE_GYROTROPIC_EFFECT 
    );
    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess) {
        printf("Kernel launch failed at secondStep_kernel: %s\n", cudaGetErrorString(err2));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err2 = cudaDeviceSynchronize();
    if (err2 != cudaSuccess) {
        printf("Kernel execution failed at secondStep_kernel: %s\n", cudaGetErrorString(err2));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    const MHDFloat timeRatio2 = (2.0 * substep + 1.0) / 4.0;  
    smrBoundary->applyUForAllDirection(
        coarseUPast, coarseUNext, 
        timeRatio2, 
        U2
    );

    // STEP3 

    sourceTermCalculator.calculateSourceTerm(U2); 
    calculateFilteringFlux(U2, maxSpeed); 
    thirdStep_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(U2.data()), 
        thrust::raw_pointer_cast(UPast.data()), 
        thrust::raw_pointer_cast(reconstructor->getCenterMHDValueRef().data()), 
        thrust::raw_pointer_cast(filteringFluxX.data()), 
        thrust::raw_pointer_cast(filteringFluxY.data()),   
        thrust::raw_pointer_cast(sourceTermCalculator.getSourceRef().data()), 
        NX, NY,
        mHDGridParameter.BUFFER,   
        DX, DY,  
        DT, 
        mHDConstParameter.CH, mHDConstParameter.CP, 
        mHDConstParameter.NUG_COEF, mHDConstParameter.EPS,  
        order, 
        mHDConstParameter.ACTIVATE_ISOTROPIC_EFFECT, 
        mHDConstParameter.ACTIVATE_GYROTROPIC_EFFECT 
    );
    cudaError_t err3 = cudaGetLastError();
    if (err3 != cudaSuccess) {
        printf("Kernel launch failed at thirdStep_kernel: %s\n", cudaGetErrorString(err3));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err3 = cudaDeviceSynchronize();
    if (err3 != cudaSuccess) {
        printf("Kernel execution failed at thirdStep_kernel: %s\n", cudaGetErrorString(err3));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }

    const MHDFloat timeRatio3 = 2.0 * (substep + 1.0) / 4.0;  
    smrBoundary->applyUForAllDirection(
        coarseUPast, coarseUNext, 
        timeRatio3, 
        U
    );
}


/*
__global__ static void calculateDtVector_kernel(
    const MHDValue* U, 
    const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
    const MHDFloat DX, const MHDFloat DY, 
    const MHDFloat EPS, 
    MHDFloat* dtVector 
)
{
    MHDUnsignedInt i = blockIdx.x * blockDim.x + threadIdx.x;
    MHDUnsignedInt j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < NX && j < NY) {
        MHDUnsignedLongLong index = getIndex<MHDUnsignedLongLong>(i, j, NX, NY);

        if (i == 0 || j == 0) {
            dtVector[index] = 1e10;
            return;
        }

        const MHDFloat& rho = U[index].rho;
        const MHDFloat& u   = U[index].u;
        const MHDFloat& v   = U[index].v;
        const MHDFloat& w   = U[index].w;
        const MHDFloat& bX  = U[index].bX;
        const MHDFloat& bY  = U[index].bY;
        const MHDFloat& bZ  = U[index].bZ;
        const MHDFloat& pXX = U[index].pXX;
        const MHDFloat& pYY = U[index].pYY;
        const MHDFloat& pZZ = U[index].pZZ;
        const MHDFloat& pXY = U[index].pXY;
        const MHDFloat& pXZ = U[index].pXZ;
        const MHDFloat& pYZ = U[index].pYZ;

        MHDFloat b, c; 

        b = (4.0 * pXX + bX * bX + bY * bY + bZ * bZ) / (2.0 * rho);
        c = ((3.0 * pXX + bY * bY + bZ * bZ) * (pXX + bX * bX) 
          + (2.0 * pXY - bX * bY) * bX * bY
          + (2.0 * pXZ - bX * bZ) * bX * bZ)
          / (rho * rho); 
        MHDFloat maxSpeedX = abs(u) + sqrt(b + sqrt(max(b * b - c, 0.0)));

        b = (4.0 * pYY + bX * bX + bY * bY + bZ * bZ) / (2.0 * rho);
        c = ((3.0 * pYY + bZ * bZ + bX * bX) * (pYY + bY * bY) 
          + (2.0 * pYZ - bY * bZ) * bY * bZ
          + (2.0 * pXY - bX * bY) * bX * bY)
          / (rho * rho); 
        MHDFloat maxSpeedY = abs(v) + sqrt(b + sqrt(max(b * b - c, 0.0)));

        b = (4.0 * pZZ + bX * bX + bY * bY + bZ * bZ) / (2.0 * rho);
        c = ((3.0 * pZZ + bX * bX + bY * bY) * (pZZ + bZ * bZ) 
          + (2.0 * pXZ - bX * bZ) * bX * bZ
          + (2.0 * pYZ - bY * bZ) * bY * bZ)
          / (rho * rho); 
        MHDFloat maxSpeedZ = abs(w) + sqrt(b + sqrt(max(b * b - c, 0.0)));

        dtVector[index] = static_cast<MHDFloat>(1.0) / (maxSpeedX / DX + maxSpeedY / DY + EPS);
    }
}


MHDFloat SSPRK3::calculateAndGetDt(
    const thrust::device_vector<MHDValue>& U 
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (NY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    calculateDtVector_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        NX, NY,   
        DX, DY,  
        mHDConstParameter.EPS, 
        thrust::raw_pointer_cast(dtVector.data())
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed at calculateDtVector_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed at calculateDtVector_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }

    thrust::device_vector<MHDFloat>::iterator dtMinIt = thrust::min_element(dtVector.begin(), dtVector.end());
    
    MHDFloat dtMin = (*dtMinIt) * mHDConstParameter.CFL;

    return dtMin;
}
*/


thrust::device_vector<MHDValue>& SSPRK3::getURef()
{
    return U; 
}


thrust::device_vector<MHDValue>& SSPRK3::getUPastRef()
{
    return UPast;
}


Boundary& SSPRK3::getBoundaryRef()
{
    return *boundary;
}


SMRBoundary& SSPRK3::getSMRBoundaryRef()
{
    return *smrBoundary;
}

