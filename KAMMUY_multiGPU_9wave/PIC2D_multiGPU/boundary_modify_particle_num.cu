#include "boundary.hpp"


void BoundaryPIC::modifySendNumParticles()
{
    modifySendNumParticlesSpecies(
        mPIInfo.numForSendParticlesIonCornerLeftDown, 
        mPIInfo.numForSendParticlesIonCornerRightDown, 
        mPIInfo.numForSendParticlesIonCornerLeftUp, 
        mPIInfo.numForSendParticlesIonCornerRightUp, 
        mPIInfo.numForRecvParticlesIonCornerLeftDown, 
        mPIInfo.numForRecvParticlesIonCornerRightDown, 
        mPIInfo.numForRecvParticlesIonCornerLeftUp, 
        mPIInfo.numForRecvParticlesIonCornerRightUp, 
        mPIInfo.numForSendParticlesIonDown, 
        mPIInfo.numForSendParticlesIonUp 
    ); 
    modifySendNumParticlesSpecies(
        mPIInfo.numForSendParticlesElectronCornerLeftDown, 
        mPIInfo.numForSendParticlesElectronCornerRightDown, 
        mPIInfo.numForSendParticlesElectronCornerLeftUp, 
        mPIInfo.numForSendParticlesElectronCornerRightUp, 
        mPIInfo.numForRecvParticlesElectronCornerLeftDown, 
        mPIInfo.numForRecvParticlesElectronCornerRightDown, 
        mPIInfo.numForRecvParticlesElectronCornerLeftUp, 
        mPIInfo.numForRecvParticlesElectronCornerRightUp, 
        mPIInfo.numForSendParticlesElectronDown, 
        mPIInfo.numForSendParticlesElectronUp 
    ); 

    MPI_Barrier(MPI_COMM_WORLD);
}


void BoundaryPIC::modifySendNumParticlesSpecies(
    const unsigned int& numForSendParticlesSpeciesCornerLeftDown, 
    const unsigned int& numForSendParticlesSpeciesCornerRightDown, 
    const unsigned int& numForSendParticlesSpeciesCornerLeftUp, 
    const unsigned int& numForSendParticlesSpeciesCornerRightUp, 
    unsigned int& numForRecvParticlesSpeciesCornerLeftDown, 
    unsigned int& numForRecvParticlesSpeciesCornerRightDown, 
    unsigned int& numForRecvParticlesSpeciesCornerLeftUp, 
    unsigned int& numForRecvParticlesSpeciesCornerRightUp, 
    unsigned int& numForSendParticlesSpeciesDown, 
    unsigned int& numForSendParticlesSpeciesUp
)
{
    sendrecv_numParticle_corner(
        numForSendParticlesSpeciesCornerLeftDown, 
        numForSendParticlesSpeciesCornerRightDown, 
        numForSendParticlesSpeciesCornerLeftUp, 
        numForSendParticlesSpeciesCornerRightUp, 
        numForRecvParticlesSpeciesCornerLeftDown, 
        numForRecvParticlesSpeciesCornerRightDown, 
        numForRecvParticlesSpeciesCornerLeftUp, 
        numForRecvParticlesSpeciesCornerRightUp, 
        mPIInfo
    );

    numForSendParticlesSpeciesDown += numForRecvParticlesSpeciesCornerLeftDown
                                    + numForRecvParticlesSpeciesCornerRightDown;
    numForSendParticlesSpeciesUp   += numForRecvParticlesSpeciesCornerLeftUp
                                    + numForRecvParticlesSpeciesCornerRightUp;

}



