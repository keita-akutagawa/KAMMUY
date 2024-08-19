#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>

#include "../../IdealMHD2D_gpu/IdealMHD2D.hpp"
#include "../../PIC2D_gpu/PIC2D.hpp"
#include "../../Interface2D/interface.hpp"
#include "../../PIC2D_gpu/boundary.hpp"
#include "../../IdealMHD2D_gpu/boundary.hpp"
#include "../../Interface2D/remove_noise.hpp"

#include "../../IdealMHD2D_gpu/const.hpp"
#include "../../PIC2D_gpu/const.hpp"
#include "../../Interface2D/const.hpp"


using namespace IdealMHD2DConst;
using namespace PIC2DConst;
using namespace Interface2DConst;


std::string directoryname = "results_current_sheet";
std::string filenameWithoutStep = "current_sheet";
std::ofstream logfile("results_current_sheet/log_current_sheet.txt");

const int IdealMHD2DConst::totalStep_MHD = 100000;
const int PIC2DConst::totalStep_PIC = -1;
const int recordStep = 100;
const bool isParticleRecord = false;
const int particleRecordStep = PIC2DConst::totalStep_PIC;

double PIC2DConst::totalTime_PIC = 0.0;
double IdealMHD2DConst::totalTime_MHD = 0.0;

const double Interface2DConst::EPS = 1.0e-20;
const double Interface2DConst::PI = 3.14159265358979;
const double PIC2DConst::EPS_PIC = 1.0e-20;
const double IdealMHD2DConst::EPS_MHD = 1.0e-20;
const double IdealMHD2DConst::PI_MHD = 3.14159265358979;

const int PIC2DConst::nx_PIC = 1000;
const double PIC2DConst::dx_PIC = 1.0;
const double PIC2DConst::xmin_PIC = 0.0 * dx_PIC; 
const double PIC2DConst::xmax_PIC = nx_PIC * dx_PIC - 0.0 * dx_PIC;

const int PIC2DConst::ny_PIC = 1000;
const double PIC2DConst::dy_PIC = 1.0;
const double PIC2DConst::ymin_PIC = 0.0 * dy_PIC; 
const double PIC2DConst::ymax_PIC = ny_PIC * dy_PIC - 0.0 * dy_PIC;


const int IdealMHD2DConst::nx_MHD = PIC2DConst::nx_PIC;
const double IdealMHD2DConst::dx_MHD = 1.0;
const double IdealMHD2DConst::xmin_MHD = 0.0 * dx_MHD;
const double IdealMHD2DConst::xmax_MHD = nx_MHD * dx_MHD - 0.0 * dx_MHD;

const int IdealMHD2DConst::ny_MHD = 5000;
const double IdealMHD2DConst::dy_MHD = 1.0;
const double IdealMHD2DConst::ymin_MHD = 0.0 * dy_MHD;
const double IdealMHD2DConst::ymax_MHD = ny_MHD * dy_MHD - 0.0 * dy_MHD;


// Interface

const int Interface2DConst::windowSizeForConvolution = 5;

const int interfaceLength = 50;
const int indexOfInterfaceStartInPIC_Lower = 0;
const int indexOfInterfaceStartInMHD_Lower = IdealMHD2DConst::ny_MHD - interfaceLength;
const int indexOfInterfaceStartInPIC_Upper = PIC2DConst::ny_PIC - interfaceLength;
const int indexOfInterfaceStartInMHD_Upper = 0;
const int nx_Interface = PIC2DConst::nx_PIC;
const int ny_Interface = interfaceLength + 5 * Interface2DConst::windowSizeForConvolution;

thrust::host_vector<double> host_interlockingFunctionY_Lower(interfaceLength, 0.0);
thrust::host_vector<double> host_interlockingFunctionYHalf_Lower(interfaceLength, 0.0);
thrust::host_vector<double> host_interlockingFunctionY_Upper(interfaceLength, 0.0);
thrust::host_vector<double> host_interlockingFunctionYHalf_Upper(interfaceLength, 0.0);

const unsigned long long Interface2DConst::reloadParticlesTotalNumIon = PIC2DConst::numberDensityIon_PIC * PIC2DConst::nx_PIC * (interfaceLength + 0);
const unsigned long long Interface2DConst::reloadParticlesTotalNumElectron = PIC2DConst::numberDensityElectron_PIC * PIC2DConst::nx_PIC * (interfaceLength + 0);

// PIC

const double PIC2DConst::c_PIC = 1.0;
const double PIC2DConst::epsilon0_PIC = 1.0;
const double PIC2DConst::mu0_PIC = 1.0;
const double PIC2DConst::dOfLangdonMarderTypeCorrection_PIC = 0.01;

const int PIC2DConst::numberDensityIon_PIC = 100;
const int PIC2DConst::numberDensityElectron_PIC = 100;

const double PIC2DConst::b0_PIC = sqrt(static_cast<double>(PIC2DConst::numberDensityElectron_PIC)) / 1.0;

const double PIC2DConst::mRatio_PIC = 25.0;
const double PIC2DConst::mElectron_PIC = 1.0;
const double PIC2DConst::mIon_PIC = mRatio_PIC * mElectron_PIC;

const float tRatio = 1.0f;
const float tElectron = (B0 * B0 / 2.0 / mu0) / (numberDensityIon + numberDensityElectron * tRatio);
const float tIon = tRatio * tElectron;

const double PIC2DConst::qRatio_PIC = -1.0;
const double PIC2DConst::qElectron_PIC = -1.0 * sqrt(epsilon0_PIC * tElectron_PIC / static_cast<double>(numberDensityElectron_PIC));
const double PIC2DConst::qIon_PIC = qRatio_PIC * qElectron_PIC;

const double PIC2DConst::omegaPe_PIC = sqrt(static_cast<double>(numberDensityElectron_PIC) * pow(qElectron_PIC, 2) / mElectron_PIC / epsilon0_PIC);
const double PIC2DConst::omegaPi_PIC = sqrt(static_cast<double>(numberDensityIon_PIC) * pow(qIon_PIC, 2) / mIon_PIC / epsilon0_PIC);
const double PIC2DConst::omegaCe_PIC = abs(qElectron_PIC * b0_PIC / mElectron_PIC);
const double PIC2DConst::omegaCi_PIC = qIon_PIC * b0_PIC / mIon_PIC;

const double PIC2DConst::debyeLength_PIC = sqrt(epsilon0_PIC * tElectron_PIC / static_cast<double>(numberDensityElectron_PIC) / pow(qElectron_PIC, 2));
const double PIC2DConst::ionInertialLength_PIC = c_PIC / omegaPi_PIC;

double PIC2DConst::dt_PIC = 0.0;

const unsigned long long harrisNumIon = round(nx * numberDensityIon * 2.0f * sheatThickness);
const unsigned long long backgroundNumIon = round(0.2f * nx * ny * numberDensityIon);
const unsigned long long harrisNumElectron = round(nx * numberDensityElectron * 2.0f * sheatThickness);
const unsigned long long backgroundNumElectron = round(0.2f * nx * ny * numberDensityElectron);
const unsigned long long PIC2DConst::existNumIon = harrisNumIon + backgroundNumIon;
const unsigned long long PIC2DConst::existNumElectron = harrisNumElectron + backgroundNumElectron;
const unsigned long long PIC2DConst::totalNumIon_PIC = existNumIon_PIC + Interface2DConst::reloadParticlesTotalNumIon;
const unsigned long long PIC2DConst::totalNumElectron_PIC = existNumElectron_PIC + Interface2DConst::reloadParticlesTotalNumElectron;
const unsigned long long PIC2DConst::totalNumParticles_PIC = totalNumIon_PIC + totalNumElectron_PIC;

const double PIC2DConst::vThIon_PIC = sqrt(2.0 * tIon_PIC / mIon_PIC);
const double PIC2DConst::vThElectron_PIC = sqrt(2.0 * tElectron_PIC / mElectron_PIC);
const double PIC2DConst::bulkVxIon_PIC = 0.0;
const double PIC2DConst::bulkVyIon_PIC = 0.0;
const double PIC2DConst::bulkVzIon_PIC = 0.0;
const double PIC2DConst::bulkVxElectron_PIC = 0.0;
const double PIC2DConst::bulkVyElectron_PIC = 0.0;
const double PIC2DConst::bulkVzElectron_PIC = 0.0;

const float vThIonB_PIC = sqrt(2.0 * tIon * 0.2 / mIon);
const float vThElectronB_PIC = sqrt(2.0 * tElectron * 0.2 / mElectron);
const float bulkVxElectronB_PIC = 0.0;
const float bulkVyElectronB_PIC = 0.0;
const float bulkVzElectronB_PIC = 0.0;
const float bulkVxIonB_PIC = 0.0;
const float bulkVyIonB_PIC = 0.0;
const float bulkVzIonB_PIC = 0.0;


// MHD

const double IdealMHD2DConst::b0_MHD = b0_PIC;

const double IdealMHD2DConst::rho0_MHD = mIon_PIC * numberDensityIon_PIC + mElectron_PIC * numberDensityElectron_PIC;
const double IdealMHD2DConst::u0_MHD = 0.0;
const double IdealMHD2DConst::v0_MHD = 0.0;
const double IdealMHD2DConst::w0_MHD = 0.0;
const double IdealMHD2DConst::bX0_MHD = 0.0;
const double IdealMHD2DConst::bY0_MHD = 0.0;
const double IdealMHD2DConst::bZ0_MHD = 0.0;
const double IdealMHD2DConst::p0_MHD = numberDensityIon_PIC * tIon_PIC + numberDensityElectron_PIC * tElectron_PIC;
const double IdealMHD2DConst::e0_MHD = p0_MHD / (gamma_MHD - 1.0)
                                     + 0.5 * rho0_MHD * (u0_MHD * u0_MHD + v0_MHD * v0_MHD + w0_MHD * w0_MHD)
                                     + 0.5 * (bX0_MHD * bX0_MHD + bY0_MHD * bY0_MHD + bZ0_MHD * bZ0_MHD);

const double IdealMHD2DConst::CFL_MHD = 0.7;
const double IdealMHD2DConst::gamma_MHD = 5.0 / 3.0;

double IdealMHD2DConst::dt_MHD = 0.0;

////////// device //////////

// PIC

__constant__ int PIC2DConst::device_totalStep_PIC;
__device__ double PIC2DConst::device_totalTime_PIC;

__constant__ double PIC2DConst::device_c_PIC;
__constant__ double PIC2DConst::device_epsilon0_PIC;
__constant__ double PIC2DConst::device_mu0_PIC;
__constant__ double PIC2DConst::device_dOfLangdonMarderTypeCorrection_PIC;
__constant__ double PIC2DConst::device_EPS_PIC;

__constant__ int PIC2DConst::device_numberDensityIon_PIC;
__constant__ int PIC2DConst::device_numberDensityElectron_PIC;

__constant__ double PIC2DConst::device_b0_PIC;

__constant__ double PIC2DConst::device_mRatio_PIC;
__constant__ double PIC2DConst::device_mElectron_PIC;
__constant__ double PIC2DConst::device_mIon_PIC;

__constant__ double PIC2DConst::device_tRatio_PIC;
__constant__ double PIC2DConst::device_tElectron_PIC;
__constant__ double PIC2DConst::device_tIon_PIC;

__constant__ double PIC2DConst::device_qRatio_PIC;
__constant__ double PIC2DConst::device_qElectron_PIC;
__constant__ double PIC2DConst::device_qIon_PIC;

__constant__ double PIC2DConst::device_omegaPe_PIC;
__constant__ double PIC2DConst::device_omegaPi_PIC;
__constant__ double PIC2DConst::device_omegaCe_PIC;
__constant__ double PIC2DConst::device_omegaCi_PIC;

__constant__ double PIC2DConst::device_debyeLength_PIC;
__constant__ double PIC2DConst::device_ionInertialLength_PIC;

__constant__ int PIC2DConst::device_nx_PIC;
__constant__ double PIC2DConst::device_dx_PIC;
__constant__ double PIC2DConst::device_xmin_PIC;
__constant__ double PIC2DConst::device_xmax_PIC;

__constant__ int PIC2DConst::device_ny_PIC;
__constant__ double PIC2DConst::device_dy_PIC;
__constant__ double PIC2DConst::device_ymin_PIC;
__constant__ double PIC2DConst::device_ymax_PIC;

__device__ double PIC2DConst::device_dt_PIC;

__constant__ unsigned long long PIC2DConst::device_totalNumIon_PIC;
__constant__ unsigned long long PIC2DConst::device_totalNumElectron_PIC;
__constant__ unsigned long long PIC2DConst::device_totalNumParticles_PIC;

__device__ unsigned long long PIC2DConst::device_existNumIon_PIC;
__device__ unsigned long long PIC2DConst::device_existNumElectron_PIC;

__constant__ double PIC2DConst::device_vThIon_PIC;
__constant__ double PIC2DConst::device_vThElectron_PIC;
__constant__ double PIC2DConst::device_bulkVxElectron_PIC;
__constant__ double PIC2DConst::device_bulkVyElectron_PIC;
__constant__ double PIC2DConst::device_bulkVzElectron_PIC;
__constant__ double PIC2DConst::device_bulkVxIon_PIC;
__constant__ double PIC2DConst::device_bulkVyIon_PIC;
__constant__ double PIC2DConst::device_bulkVzIon_PIC;



// MHD

__constant__ double IdealMHD2DConst::device_EPS_MHD;
__constant__ double IdealMHD2DConst::device_PI_MHD;

__constant__ double IdealMHD2DConst::device_b0_MHD;

__constant__ double IdealMHD2DConst::device_rho0_MHD;
__constant__ double IdealMHD2DConst::device_u0_MHD;
__constant__ double IdealMHD2DConst::device_v0_MHD;
__constant__ double IdealMHD2DConst::device_w0_MHD;
__constant__ double IdealMHD2DConst::device_bX0_MHD;
__constant__ double IdealMHD2DConst::device_bY0_MHD;
__constant__ double IdealMHD2DConst::device_bZ0_MHD;
__constant__ double IdealMHD2DConst::device_p0_MHD;
__constant__ double IdealMHD2DConst::device_e0_MHD;

__constant__ double IdealMHD2DConst::device_dx_MHD;
__constant__ double IdealMHD2DConst::device_xmin_MHD;
__constant__ double IdealMHD2DConst::device_xmax_MHD;
__constant__ int IdealMHD2DConst::device_nx_MHD;

__constant__ double IdealMHD2DConst::device_dy_MHD;
__constant__ double IdealMHD2DConst::device_ymin_MHD;
__constant__ double IdealMHD2DConst::device_ymax_MHD;
__constant__ int IdealMHD2DConst::device_ny_MHD;

__constant__ double IdealMHD2DConst::device_CFL_MHD;
__constant__ double IdealMHD2DConst::device_gamma_MHD;

__device__ double IdealMHD2DConst::device_dt_MHD;


// Interface

__constant__ double Interface2DConst::device_EPS;
__constant__ double Interface2DConst::device_PI;

__constant__ int Interface2DConst::device_windowSizeForConvolution;

__constant__ unsigned long long Interface2DConst::device_reloadParticlesTotalNumIon;
__constant__ unsigned long long Interface2DConst::device_reloadParticlesTotalNumElectron;
