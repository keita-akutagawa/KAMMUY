#include "const.hpp"


using namespace PIC2DConst;

void initializeDeviceConstants_PIC()
{
    cudaMemcpyToSymbol(device_c_PIC, &c_PIC, sizeof(double));
    cudaMemcpyToSymbol(device_epsilon0_PIC, &epsilon0_PIC, sizeof(double));
    cudaMemcpyToSymbol(device_mu0_PIC, &mu0_PIC, sizeof(double));
    cudaMemcpyToSymbol(device_dOfLangdonMarderTypeCorrection_PIC, &dOfLangdonMarderTypeCorrection_PIC, sizeof(double));
    cudaMemcpyToSymbol(device_EPS_PIC, &EPS_PIC, sizeof(double));

    cudaMemcpyToSymbol(device_nx_PIC, &nx_PIC, sizeof(int));
    cudaMemcpyToSymbol(device_dx_PIC, &dx_PIC, sizeof(double));
    cudaMemcpyToSymbol(device_xmin_PIC, &xmin_PIC, sizeof(double));
    cudaMemcpyToSymbol(device_xmax_PIC, &xmax_PIC, sizeof(double));

    cudaMemcpyToSymbol(device_ny_PIC, &ny_PIC, sizeof(int));
    cudaMemcpyToSymbol(device_dy_PIC, &dy_PIC, sizeof(double));
    cudaMemcpyToSymbol(device_ymin_PIC, &ymin_PIC, sizeof(double));
    cudaMemcpyToSymbol(device_ymax_PIC, &ymax_PIC, sizeof(double));

    cudaMemcpyToSymbol(device_dt_PIC, &dt_PIC, sizeof(double));

    cudaMemcpyToSymbol(device_numberDensityIon_PIC, &numberDensityIon_PIC, sizeof(int));
    cudaMemcpyToSymbol(device_numberDensityElectron_PIC, &numberDensityElectron_PIC, sizeof(int));

    cudaMemcpyToSymbol(device_totalNumIon_PIC, &totalNumIon_PIC, sizeof(unsigned long long));
    cudaMemcpyToSymbol(device_totalNumElectron_PIC, &totalNumElectron_PIC, sizeof(unsigned long long));
    cudaMemcpyToSymbol(device_totalNumParticles_PIC, &totalNumParticles_PIC, sizeof(unsigned long long));
    cudaMemcpyToSymbol(device_existNumIon_PIC, &existNumIon_PIC, sizeof(unsigned long long));
    cudaMemcpyToSymbol(device_existNumElectron_PIC, &existNumElectron_PIC, sizeof(unsigned long long));

    cudaMemcpyToSymbol(device_B0_PIC, &B0_PIC, sizeof(double));

    cudaMemcpyToSymbol(device_mRatio_PIC, &mRatio_PIC, sizeof(double));
    cudaMemcpyToSymbol(device_mIon_PIC, &mIon_PIC, sizeof(double));
    cudaMemcpyToSymbol(device_mElectron_PIC, &mElectron_PIC, sizeof(double));

    cudaMemcpyToSymbol(device_tRatio_PIC, &tRatio_PIC, sizeof(double));
    cudaMemcpyToSymbol(device_tIon_PIC, &tIon_PIC, sizeof(double));
    cudaMemcpyToSymbol(device_tElectron_PIC, &tElectron_PIC, sizeof(double));

    cudaMemcpyToSymbol(device_qRatio_PIC, &qRatio_PIC, sizeof(double));
    cudaMemcpyToSymbol(device_qIon_PIC, &qIon_PIC, sizeof(double));
    cudaMemcpyToSymbol(device_qElectron_PIC, &qElectron_PIC, sizeof(double));

    cudaMemcpyToSymbol(device_omegaPe_PIC, &omegaPe_PIC, sizeof(double));
    cudaMemcpyToSymbol(device_omegaPi_PIC, &omegaPi_PIC, sizeof(double));
    cudaMemcpyToSymbol(device_omegaCe_PIC, &omegaCe_PIC, sizeof(double));
    cudaMemcpyToSymbol(device_omegaCi_PIC, &omegaCi_PIC, sizeof(double));

    cudaMemcpyToSymbol(device_debyeLength_PIC, &debyeLength_PIC, sizeof(double));
    cudaMemcpyToSymbol(device_ionInertialLength_PIC, &ionInertialLength_PIC, sizeof(double));

    cudaMemcpyToSymbol(device_vThIon_PIC, &vThIon_PIC, sizeof(double));
    cudaMemcpyToSymbol(device_vThElectron_PIC, &vThElectron_PIC, sizeof(double));
    cudaMemcpyToSymbol(device_bulkVxIon_PIC, &bulkVxIon_PIC, sizeof(double));
    cudaMemcpyToSymbol(device_bulkVyIon_PIC, &bulkVyIon_PIC, sizeof(double));
    cudaMemcpyToSymbol(device_bulkVzIon_PIC, &bulkVzIon_PIC, sizeof(double));
    cudaMemcpyToSymbol(device_bulkVxElectron_PIC, &bulkVxElectron_PIC, sizeof(double));
    cudaMemcpyToSymbol(device_bulkVyElectron_PIC, &bulkVyElectron_PIC, sizeof(double));
    cudaMemcpyToSymbol(device_bulkVzElectron_PIC, &bulkVzElectron_PIC, sizeof(double));

    cudaMemcpyToSymbol(device_totalStep_PIC, &totalStep_PIC, sizeof(int));
    cudaMemcpyToSymbol(device_totalTime_PIC, &totalTime_PIC, sizeof(double));
}

