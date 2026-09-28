#include "interface2D.hpp"


auto smoothStep = [](PICFloat dist, InterfaceFloat delta) -> InterfaceFloat {
    if (dist <= 0.0) return 0.0;
    if (dist >= delta) return 1.0;
    return 0.5 * (1.0 - cos(M_PI * dist / delta));
};


Interface2D::Interface2D(
    nlohmann::json& constJSON, 
    nlohmann::json& gridJSON, 
    const MHDUnsignedInt level, 
    MHDConstParameter& mHDConstParameter, 
    const MHDGridParameter& mHDGridParameter, 
    PICConstParameter& pICConstParameter, 
    const PICGridParameter& pICGridParameter
)
  : interfaceConstParameter(constJSON), 
    interfaceGridParameter(gridJSON), 
    mHDConstParameter(mHDConstParameter),
    mHDGridParameter(mHDGridParameter),
    pICConstParameter(pICConstParameter),
    pICGridParameter(pICGridParameter),

    GRID_SIZE_RATIO(interfaceGridParameter.GRID_SIZE_RATIO), 
    START_INDEX_IN_MHD_X(interfaceGridParameter.START_INDEX_IN_MHD_X), 
    START_INDEX_IN_MHD_Y(interfaceGridParameter.START_INDEX_IN_MHD_Y), 
    NX_MHD(mHDGridParameter.NX[level]), 
    NY_MHD(mHDGridParameter.NY[level]), 
    DX_MHD(mHDGridParameter.DX[level]), 
    DY_MHD(mHDGridParameter.DY[level]), 
    NX_PIC(pICGridParameter.NX), 
    NY_PIC(pICGridParameter.NY),
    DX_PIC(pICGridParameter.DX), 
    DY_PIC(pICGridParameter.DY), 

    restartParticlesIndexIon(0), 
    restartParticlesIndexElectron(0), 

    reloadParticlesDataIon     (NX_PIC * NY_PIC), 
    reloadParticlesDataElectron(NX_PIC * NY_PIC), 
    
    B_PICtoMHD                   (((NX_PIC - pICGridParameter.BUFFER) / GRID_SIZE_RATIO) * ((NY_PIC - pICGridParameter.BUFFER) / GRID_SIZE_RATIO)), 
    zerothMomentIon_PICtoMHD     (((NX_PIC - pICGridParameter.BUFFER) / GRID_SIZE_RATIO) * ((NY_PIC - pICGridParameter.BUFFER) / GRID_SIZE_RATIO)), 
    zerothMomentElectron_PICtoMHD(((NX_PIC - pICGridParameter.BUFFER) / GRID_SIZE_RATIO) * ((NY_PIC - pICGridParameter.BUFFER) / GRID_SIZE_RATIO)), 
    firstMomentIon_PICtoMHD      (((NX_PIC - pICGridParameter.BUFFER) / GRID_SIZE_RATIO) * ((NY_PIC - pICGridParameter.BUFFER) / GRID_SIZE_RATIO)), 
    firstMomentElectron_PICtoMHD (((NX_PIC - pICGridParameter.BUFFER) / GRID_SIZE_RATIO) * ((NY_PIC - pICGridParameter.BUFFER) / GRID_SIZE_RATIO)), 
    secondMomentIon_PICtoMHD     (((NX_PIC - pICGridParameter.BUFFER) / GRID_SIZE_RATIO) * ((NY_PIC - pICGridParameter.BUFFER) / GRID_SIZE_RATIO)), 
    secondMomentElectron_PICtoMHD(((NX_PIC - pICGridParameter.BUFFER) / GRID_SIZE_RATIO) * ((NY_PIC - pICGridParameter.BUFFER) / GRID_SIZE_RATIO)), 

    timeInterpolatedU(NX_MHD * NY_MHD), 

    interlockingFunction(NX_PIC * NY_PIC) 
{
    thrust::host_vector<InterfaceFloat> host_interlockingFunction(NX_PIC * NY_PIC, 0.0);
    
    for (PICUnsignedInt i = pICGridParameter.BUFFER; i < NX_PIC - pICGridParameter.BUFFER; i++) {
        for (PICUnsignedInt j = pICGridParameter.BUFFER; j < NY_PIC - pICGridParameter.BUFFER; j++) {
            PICUnsignedLongLong index = getIndex<PICUnsignedLongLong>(i, j, NX_PIC, NY_PIC); 

            PICFloat x1 = pICGridParameter.XMIN + (pICGridParameter.BUFFER + 0.5) * DX_PIC; 
            PICFloat x2 = pICGridParameter.XMIN + (NX_PIC - pICGridParameter.BUFFER - 0.5) * DX_PIC; 
            PICFloat y1 = pICGridParameter.YMIN + (pICGridParameter.BUFFER + 0.5) * DY_PIC; 
            PICFloat y2 = pICGridParameter.YMIN + (NY_PIC - pICGridParameter.BUFFER - 0.5) * DY_PIC; 
            PICFloat x = pICGridParameter.XMIN + (i + 0.5) * DX_PIC; 
            PICFloat y = pICGridParameter.YMIN + (j + 0.5) * DY_PIC; 

            InterfaceFloat delta = interfaceConstParameter.DELTA_INTERLOCKING_FUNCTION; 

            InterfaceFloat wx = smoothStep(x - x1, delta) * smoothStep(x2 - x, delta);
            InterfaceFloat wy = smoothStep(y - y1, delta) * smoothStep(y2 - y, delta);

            host_interlockingFunction[index] = wx * wy;

            //PICUnsignedInt CASTED_DELTA = static_cast<PICUnsignedInt>(interfaceConstParameter.DELTA_INTERLOCKING_FUNCTION); 
            //if (pICGridParameter.BUFFER <= i && i < NX_PIC - pICGridParameter.BUFFER - 1 && pICGridParameter.BUFFER <= j && j < NY_PIC - pICGridParameter.BUFFER - 1) {
            //    host_interlockingFunction[index] = 1.0; 
            //} else {
            //    host_interlockingFunction[index] = 0.0;
            //}
        }
    }

    interlockingFunction = host_interlockingFunction; 
}


InterfaceGridParameter& Interface2D::getInterfaceGridParameterRef()
{
    return interfaceGridParameter;
}


InterfaceConstParameter& Interface2D::getInterfaceConstParameterRef()
{
    return interfaceConstParameter;
}

