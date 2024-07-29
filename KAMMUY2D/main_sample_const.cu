#include "../IdealMHD2D_gpu/const.hpp"
#include "../PIC2D_gpu_single/const.hpp"
#include "../Interface2D/const.hpp"
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>


std::string directoryname = "results_sample";
std::string filenameWithoutStep = "sample";
std::ofstream logfile("results_sample/log_sample.txt");

const int PIC2DConst::totalStep = 100 * 100;
const int fieldRecordStep = 100;
const bool isParticleRecord = false;
const int particleRecordStep = PIC2DConst::totalStep;
float PIC2DConst::totalTime = 0.0f;

const int IdealMHD2DConst::totalStep = 0;
double IdealMHD2DConst::totalTime = 0.0;

const int interfaceLength = 50;
const int indexOfInterfaceStartInPIC = 0;
const int indexOfInterfaceStartInMHD = 100;


// PIC

const float PIC2DConst::c = 1.0f;
const float PIC2DConst::epsilon0 = 1.0f;
const float PIC2DConst::mu0 = 1.0f;
const float PIC2DConst::dOfLangdonMarderTypeCorrection = 0.001f;

const int PIC2DConst::numberDensityIon = 100;
const int PIC2DConst::numberDensityElectron = 100;

const float PIC2DConst::B0 = sqrt(static_cast<float>(numberDensityElectron)) / 1.0;

const float PIC2DConst::mRatio = 25.0f;
const float PIC2DConst::mElectron = 1.0f;
const float PIC2DConst::mIon = mRatio * mElectron;

const float PIC2DConst::tRatio = 1.0f;
const float PIC2DConst::tElectron = (B0 * B0 / 2.0 / mu0) / (numberDensityIon + numberDensityElectron * tRatio);
const float PIC2DConst::tIon = tRatio * tElectron;

const float PIC2DConst::qRatio = -1.0f;
const float PIC2DConst::qElectron = -1.0f * sqrt(epsilon0 * tElectron / static_cast<float>(numberDensityElectron)) / 1.0f;
const float PIC2DConst::qIon = qRatio * qElectron;

const float PIC2DConst::omegaPe = sqrt(static_cast<float>(numberDensityElectron) * pow(qElectron, 2) / mElectron / epsilon0);
const float PIC2DConst::omegaPi = sqrt(static_cast<float>(numberDensityIon) * pow(qIon, 2) / mIon / epsilon0);
const float PIC2DConst::omegaCe = abs(qElectron * B0 / mElectron);
const float PIC2DConst::omegaCi = qIon * B0 / mIon;

const float PIC2DConst::debyeLength = sqrt(epsilon0 * tElectron / static_cast<float>(numberDensityElectron) / pow(qElectron, 2));
//追加
const float PIC2DConst::ionInertialLength = c / omegaPi;

const int PIC2DConst::nx = int(100.0f * ionInertialLength);
const float PIC2DConst::dx = 1.0f;
const float PIC2DConst::xmin = 0.0f * dx; 
const float PIC2DConst::xmax = nx * dx - 0.0f * dx;

const int PIC2DConst::ny = int(20.0f * ionInertialLength);
const float PIC2DConst::dy = 1.0f;
const float PIC2DConst::ymin = 1.0f * dy; 
const float PIC2DConst::ymax = ny * dy - 1.5f * dy;

const float PIC2DConst::dt = 0.5f;

const unsigned long long PIC2DConst::totalNumIon = static_cast<unsigned long long>(nx * ny * numberDensityIon);
const unsigned long long PIC2DConst::totalNumElectron = static_cast<unsigned long long>(nx * ny * numberDensityElectron);
const unsigned long long PIC2DConst::totalNumParticles = totalNumIon + totalNumElectron;

const float PIC2DConst::vThIon = sqrt(2.0f * tIon / mIon);
const float PIC2DConst::vThElectron = sqrt(2.0f * tElectron / mElectron);
const float PIC2DConst::bulkVxElectron = 0.0f;
const float PIC2DConst::bulkVyElectron = 0.0f;
const float PIC2DConst::bulkVzElectron = 0.0f;
const float PIC2DConst::bulkVxIon = 0.0f;
const float PIC2DConst::bulkVyIon = 0.0f;
const float PIC2DConst::bulkVzIon = 0.0f;


// MHD

const double IdealMHD2DConst::EPS = 1e-40;
const double IdealMHD2DConst::PI = 3.14159265358979;

const double IdealMHD2DConst::dx = 1.0;
const double IdealMHD2DConst::xmin = 0.0;
const double IdealMHD2DConst::xmax = 100.0;
const int IdealMHD2DConst::nx = int((xmax - xmin) / dx);

const double IdealMHD2DConst::dy = 1.0;
const double IdealMHD2DConst::ymin = 0.0;
const double IdealMHD2DConst::ymax = 100.0;
const int IdealMHD2DConst::ny = int((ymax - ymin) / dy);

const double IdealMHD2DConst::CFL = 0.7;
const double IdealMHD2DConst::gamma_mhd = 5.0 / 3.0;

double IdealMHD2DConst::dt = 0.0;


// Interface

const float Interface2DConst::PI = 3.14159265358979f;

const int Interface2DConst::windowSizeForRemoveNoiseByConvolution = 5;

const unsigned long long Interface2DConst::reloadParticlesTotalNumIon = PIC2DConst::numberDensityIon * PIC2DConst::nx * (interfaceLength + 50);
const unsigned long long Interface2DConst::reloadParticlesTotalNumElectron = PIC2DConst::numberDensityElectron * PIC2DConst::nx * (interfaceLength + 50);

////////// device //////////

// PIC

__constant__ int PIC2DConst::device_totalStep;
__constant__ int PIC2DConst::device_recordStep;
__device__ float PIC2DConst::device_totalTime;

__constant__ float PIC2DConst::device_c;
__constant__ float PIC2DConst::device_epsilon0;
__constant__ float PIC2DConst::device_mu0;
__constant__ float PIC2DConst::device_dOfLangdonMarderTypeCorrection;

__constant__ int PIC2DConst::device_numberDensityIon;
__constant__ int PIC2DConst::device_numberDensityElectron;

__constant__ float PIC2DConst::device_B0;

__constant__ float PIC2DConst::device_mRatio;
__constant__ float PIC2DConst::device_mElectron;
__constant__ float PIC2DConst::device_mIon;

__constant__ float PIC2DConst::device_tRatio;
__constant__ float PIC2DConst::device_tElectron;
__constant__ float PIC2DConst::device_tIon;

__constant__ float PIC2DConst::device_qRatio;
__constant__ float PIC2DConst::device_qElectron;
__constant__ float PIC2DConst::device_qIon;

__constant__ float PIC2DConst::device_omegaPe;
__constant__ float PIC2DConst::device_omegaPi;
__constant__ float PIC2DConst::device_omegaCe;
__constant__ float PIC2DConst::device_omegaCi;

__constant__ float PIC2DConst::device_debyeLength;
__constant__ float PIC2DConst::device_ionInertialLength;

__constant__ int PIC2DConst::device_nx;
__constant__ float PIC2DConst::device_dx;
__constant__ float PIC2DConst::device_xmin;
__constant__ float PIC2DConst::device_xmax;

__constant__ int PIC2DConst::device_ny;
__constant__ float PIC2DConst::device_dy;
__constant__ float PIC2DConst::device_ymin;
__constant__ float PIC2DConst::device_ymax;

__constant__ float PIC2DConst::device_dt;

__constant__ unsigned long long PIC2DConst::device_totalNumIon;
__constant__ unsigned long long PIC2DConst::device_totalNumElectron;
__constant__ unsigned long long PIC2DConst::device_totalNumParticles;

__constant__ float PIC2DConst::device_vThIon;
__constant__ float PIC2DConst::device_vThElectron;
__constant__ float PIC2DConst::device_bulkVxElectron;
__constant__ float PIC2DConst::device_bulkVyElectron;
__constant__ float PIC2DConst::device_bulkVzElectron;
__constant__ float PIC2DConst::device_bulkVxIon;
__constant__ float PIC2DConst::device_bulkVyIon;
__constant__ float PIC2DConst::device_bulkVzIon;



// MHD

__constant__ double IdealMHD2DConst::device_EPS;
__constant__ double IdealMHD2DConst::device_PI;

__constant__ double IdealMHD2DConst::device_dx;
__constant__ double IdealMHD2DConst::device_xmin;
__constant__ double IdealMHD2DConst::device_xmax;
__constant__ int IdealMHD2DConst::device_nx;

__constant__ double IdealMHD2DConst::device_dy;
__constant__ double IdealMHD2DConst::device_ymin;
__constant__ double IdealMHD2DConst::device_ymax;
__constant__ int IdealMHD2DConst::device_ny;

__constant__ double IdealMHD2DConst::device_CFL;
__constant__ double IdealMHD2DConst::device_gamma_mhd;

__device__ double IdealMHD2DConst::device_dt;


// Interface

__constant__ float Interface2DConst::device_PI;

__constant__ int Interface2DConst::device_windowSizeForRemoveNoiseByConvolution;

__constant__ unsigned long long Interface2DConst::device_reloadParticlesTotalNumIon;
__constant__ unsigned long long Interface2DConst::device_reloadParticlesTotalNumElectron;
