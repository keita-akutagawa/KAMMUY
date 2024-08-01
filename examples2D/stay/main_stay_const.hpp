#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>

#include "../../IdealMHD2D_gpu/const.hpp"
#include "../../PIC2D_gpu_single/const.hpp"
#include "../../Interface2D/const.hpp"


using namespace IdealMHD2DConst;
using namespace PIC2DConst;
using namespace Interface2DConst;


std::string directoryname = "results_stay";
std::string filenameWithoutStep = "stay";
std::ofstream logfile("results_stay/log_stay.txt");

const int PIC2DConst::totalStep_PIC = 100 * 100;
const int fieldRecordStep = 100;
const bool isParticleRecord = false;
const int particleRecordStep = PIC2DConst::totalStep_PIC;
float PIC2DConst::totalTime_PIC = 0.0f;

const int IdealMHD2DConst::totalStep_MHD = 0;
double IdealMHD2DConst::totalTime_MHD = 0.0;

const int interfaceLength = 50;
const int indexOfInterfaceStartInPIC = 0;
const int indexOfInterfaceStartInMHD = 100;


// PIC

const float PIC2DConst::c_PIC = 1.0f;
const float PIC2DConst::epsilon0_PIC = 1.0f;
const float PIC2DConst::mu0_PIC = 1.0f;
const float PIC2DConst::dOfLangdonMarderTypeCorrection_PIC = 0.001f;
const float PIC2DConst::EPS_PIC = 1e-20f;

const int PIC2DConst::numberDensityIon_PIC = 100;
const int PIC2DConst::numberDensityElectron_PIC = 100;

const float PIC2DConst::B0_PIC = sqrt(static_cast<float>(PIC2DConst::numberDensityElectron_PIC)) / 1.0;

const float PIC2DConst::mRatio_PIC = 100.0f;
const float PIC2DConst::mElectron_PIC = 1.0f;
const float PIC2DConst::mIon_PIC = mRatio_PIC * mElectron_PIC;

const float PIC2DConst::tRatio_PIC = 1.0f;
const float PIC2DConst::tElectron_PIC = 0.5f * mIon_PIC * pow(0.1f * c_PIC, 2);
const float PIC2DConst::tIon_PIC = tRatio_PIC * tElectron_PIC;

const float PIC2DConst::qRatio_PIC = -1.0f;
const float PIC2DConst::qElectron_PIC = -1.0f * sqrt(epsilon0_PIC * tElectron_PIC / static_cast<float>(numberDensityElectron_PIC));
const float PIC2DConst::qIon_PIC = qRatio_PIC * qElectron_PIC;

const float PIC2DConst::omegaPe_PIC = sqrt(static_cast<float>(numberDensityElectron_PIC) * pow(qElectron_PIC, 2) / mElectron_PIC / epsilon0_PIC);
const float PIC2DConst::omegaPi_PIC = sqrt(static_cast<float>(numberDensityIon_PIC) * pow(qIon_PIC, 2) / mIon_PIC / epsilon0_PIC);
const float PIC2DConst::omegaCe_PIC = abs(qElectron_PIC * B0_PIC / mElectron_PIC);
const float PIC2DConst::omegaCi_PIC = qIon_PIC * B0_PIC / mIon_PIC;

const float PIC2DConst::debyeLength_PIC = sqrt(epsilon0_PIC * tElectron_PIC / static_cast<float>(numberDensityElectron_PIC) / pow(qElectron_PIC, 2));
const float PIC2DConst::ionInertialLength_PIC = c_PIC / omegaPi_PIC;

const int PIC2DConst::nx_PIC = 100;
const float PIC2DConst::dx_PIC = 1.0f;
const float PIC2DConst::xmin_PIC = 0.0f * dx_PIC; 
const float PIC2DConst::xmax_PIC = nx_PIC * dx_PIC - 0.0f * dx_PIC;

const int PIC2DConst::ny_PIC = 200;
const float PIC2DConst::dy_PIC = 1.0f;
const float PIC2DConst::ymin_PIC = 1.0f * dy_PIC; 
const float PIC2DConst::ymax_PIC = ny_PIC * dy_PIC - 1.5f * dy_PIC;

float PIC2DConst::dt_PIC = 0.0f;

const unsigned long long PIC2DConst::totalNumIon_PIC = static_cast<unsigned long long>(nx_PIC * ny_PIC * numberDensityIon_PIC);
const unsigned long long PIC2DConst::totalNumElectron_PIC = static_cast<unsigned long long>(nx_PIC * ny_PIC * numberDensityElectron_PIC);
const unsigned long long PIC2DConst::totalNumParticles_PIC = totalNumIon_PIC + totalNumElectron_PIC;
unsigned long long PIC2DConst::existNumIon_PIC = totalNumIon_PIC;
unsigned long long PIC2DConst::existNumElectron_PIC = totalNumElectron_PIC;

const float PIC2DConst::vThIon_PIC = sqrt(2.0f * tIon_PIC / mIon_PIC);
const float PIC2DConst::vThElectron_PIC = sqrt(2.0f * tElectron_PIC / mElectron_PIC);
const float PIC2DConst::bulkVxElectron_PIC = 0.0f;
const float PIC2DConst::bulkVyElectron_PIC = 0.0f;
const float PIC2DConst::bulkVzElectron_PIC = 0.0f;
const float PIC2DConst::bulkVxIon_PIC = 0.0f;
const float PIC2DConst::bulkVyIon_PIC = 0.0f;
const float PIC2DConst::bulkVzIon_PIC = 0.0f;


// MHD

const double IdealMHD2DConst::EPS_MHD = 1e-40;
const double IdealMHD2DConst::PI_MHD = 3.14159265358979;

const double IdealMHD2DConst::rho0_MHD = mIon_PIC * numberDensityIon_PIC + mElectron_PIC * numberDensityElectron_PIC;
const double IdealMHD2DConst::u0_MHD = (mIon_PIC * bulkVxIon_PIC + mElectron_PIC * bulkVxElectron_PIC) / rho0_MHD;
const double IdealMHD2DConst::v0_MHD = (mIon_PIC * bulkVyIon_PIC + mElectron_PIC * bulkVyElectron_PIC) / rho0_MHD;
const double IdealMHD2DConst::w0_MHD = (mIon_PIC * bulkVzIon_PIC + mElectron_PIC * bulkVzElectron_PIC) / rho0_MHD;
const double IdealMHD2DConst::bX0_MHD = 0.0;
const double IdealMHD2DConst::bY0_MHD = 0.0;
const double IdealMHD2DConst::bZ0_MHD = 0.0;
const double IdealMHD2DConst::p0_MHD = numberDensityIon_PIC * tIon_PIC + numberDensityElectron_PIC * tElectron_PIC;
const double IdealMHD2DConst::e0_MHD = p0_MHD / (gamma_MHD - 1.0)
                                     + 0.5 * rho0_MHD * (u0_MHD * u0_MHD + v0_MHD * v0_MHD + w0_MHD * w0_MHD)
                                     + 0.5 * (bX0_MHD * bX0_MHD + bY0_MHD * bY0_MHD + bZ0_MHD * bZ0_MHD);

const int IdealMHD2DConst::nx_MHD = PIC2DConst::nx_PIC;
const double IdealMHD2DConst::dx_MHD = 1.0;
const double IdealMHD2DConst::xmin_MHD = 0.0;
const double IdealMHD2DConst::xmax_MHD = nx_MHD * dx_MHD;

const int IdealMHD2DConst::ny_MHD = 200;
const double IdealMHD2DConst::dy_MHD = 1.0;
const double IdealMHD2DConst::ymin_MHD = 0.0;
const double IdealMHD2DConst::ymax_MHD = ny_MHD * dy_MHD;

const double IdealMHD2DConst::CFL_MHD = 0.7;
const double IdealMHD2DConst::gamma_MHD = 5.0 / 3.0;

double IdealMHD2DConst::dt_MHD = 0.0;


// Interface

const float Interface2DConst::PI = 3.14159265358979f;

const int Interface2DConst::windowSizeForRemoveNoiseByConvolution = 5;

const unsigned long long Interface2DConst::reloadParticlesTotalNumIon = PIC2DConst::numberDensityIon_PIC * PIC2DConst::nx_PIC * (interfaceLength + 50);
const unsigned long long Interface2DConst::reloadParticlesTotalNumElectron = PIC2DConst::numberDensityElectron_PIC * PIC2DConst::nx_PIC * (interfaceLength + 50);

////////// device //////////

// PIC

__constant__ int PIC2DConst::device_totalStep_PIC;
__device__ float PIC2DConst::device_totalTime_PIC;

__constant__ float PIC2DConst::device_c_PIC;
__constant__ float PIC2DConst::device_epsilon0_PIC;
__constant__ float PIC2DConst::device_mu0_PIC;
__constant__ float PIC2DConst::device_dOfLangdonMarderTypeCorrection_PIC;
__constant__ float PIC2DConst::device_EPS_PIC;

__constant__ int PIC2DConst::device_numberDensityIon_PIC;
__constant__ int PIC2DConst::device_numberDensityElectron_PIC;

__constant__ float PIC2DConst::device_B0_PIC;

__constant__ float PIC2DConst::device_mRatio_PIC;
__constant__ float PIC2DConst::device_mElectron_PIC;
__constant__ float PIC2DConst::device_mIon_PIC;

__constant__ float PIC2DConst::device_tRatio_PIC;
__constant__ float PIC2DConst::device_tElectron_PIC;
__constant__ float PIC2DConst::device_tIon_PIC;

__constant__ float PIC2DConst::device_qRatio_PIC;
__constant__ float PIC2DConst::device_qElectron_PIC;
__constant__ float PIC2DConst::device_qIon_PIC;

__constant__ float PIC2DConst::device_omegaPe_PIC;
__constant__ float PIC2DConst::device_omegaPi_PIC;
__constant__ float PIC2DConst::device_omegaCe_PIC;
__constant__ float PIC2DConst::device_omegaCi_PIC;

__constant__ float PIC2DConst::device_debyeLength_PIC;
__constant__ float PIC2DConst::device_ionInertialLength_PIC;

__constant__ int PIC2DConst::device_nx_PIC;
__constant__ float PIC2DConst::device_dx_PIC;
__constant__ float PIC2DConst::device_xmin_PIC;
__constant__ float PIC2DConst::device_xmax_PIC;

__constant__ int PIC2DConst::device_ny_PIC;
__constant__ float PIC2DConst::device_dy_PIC;
__constant__ float PIC2DConst::device_ymin_PIC;
__constant__ float PIC2DConst::device_ymax_PIC;

__device__ float PIC2DConst::device_dt_PIC;

__constant__ unsigned long long PIC2DConst::device_totalNumIon_PIC;
__constant__ unsigned long long PIC2DConst::device_totalNumElectron_PIC;
__constant__ unsigned long long PIC2DConst::device_totalNumParticles_PIC;

__device__ unsigned long long PIC2DConst::device_existNumIon_PIC;
__device__ unsigned long long PIC2DConst::device_existNumElectron_PIC;

__constant__ float PIC2DConst::device_vThIon_PIC;
__constant__ float PIC2DConst::device_vThElectron_PIC;
__constant__ float PIC2DConst::device_bulkVxElectron_PIC;
__constant__ float PIC2DConst::device_bulkVyElectron_PIC;
__constant__ float PIC2DConst::device_bulkVzElectron_PIC;
__constant__ float PIC2DConst::device_bulkVxIon_PIC;
__constant__ float PIC2DConst::device_bulkVyIon_PIC;
__constant__ float PIC2DConst::device_bulkVzIon_PIC;



// MHD

__constant__ double IdealMHD2DConst::device_EPS_MHD;
__constant__ double IdealMHD2DConst::device_PI_MHD;

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

__constant__ float Interface2DConst::device_PI;

__constant__ int Interface2DConst::device_windowSizeForRemoveNoiseByConvolution;

__constant__ unsigned long long Interface2DConst::device_reloadParticlesTotalNumIon;
__constant__ unsigned long long Interface2DConst::device_reloadParticlesTotalNumElectron;