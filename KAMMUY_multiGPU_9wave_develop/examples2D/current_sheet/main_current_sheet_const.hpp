#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

#include "../../IdealMHD2D_multiGPU/idealMHD2D.hpp"
#include "../../PIC2D_multiGPU/pic2D.hpp"
#include "../../Interface2D_multiGPU/interface.hpp"
#include "../../PIC2D_multiGPU/boundary.hpp"
#include "../../IdealMHD2D_multiGPU/boundary.hpp"
#include "../../Interface2D_multiGPU/remove_noise.hpp"

#include "../../IdealMHD2D_multiGPU/const.hpp"
#include "../../PIC2D_multiGPU/const.hpp"
#include "../../Interface2D_multiGPU/const.hpp"


std::string directoryName = "/cfca-work/akutagawakt/KAMMUY_multiGPU_9wave_develop/results_current_sheet_test";
std::string filenameWithoutStep = "current_sheet";
std::ofstream logfile(    directoryName + "/log_current_sheet.txt"       );
std::ofstream mpifile_MHD(directoryName + "/mpilog_mhd_current_sheet.txt");
std::ofstream mpifile_PIC(directoryName + "/mpilog_pic_current_sheet.txt");
std::ofstream mpifile_Interface(directoryName + "/mpilog_interface_current_sheet.txt");


const int buffer = 3; 


const float betaUpstream = 0.5f * 0.5f;
const float triggerRatio = 0.1f;
__constant__ float device_betaUpstream;
__constant__ float device_sheatThickness;
__constant__ float device_triggerRatio;

const int IdealMHD2DConst::totalStep = 100000;
const int PIC2DConst::totalStep = -1;
const int recordStep = 100;
const bool isParticleRecord = true;
const int particleRecordStep = recordStep;

float PIC2DConst::totalTime = 0.0f;
double IdealMHD2DConst::totalTime = 0.0;

const double Interface2DConst::EPS = 1.0e-20;
const double Interface2DConst::PI = 3.14159265358979;
const float PIC2DConst::EPS = 1.0e-10f;
const double IdealMHD2DConst::EPS = 1.0e-20;
const double IdealMHD2DConst::PI = 3.14159265358979;

const int PIC2DConst::nx = 2000;
const float PIC2DConst::dx = 1.0;
const float PIC2DConst::xmin = 0.0 * PIC2DConst::dx; 
const float PIC2DConst::xmax = PIC2DConst::nx * PIC2DConst::dx - 0.0 * PIC2DConst::dx;

const int PIC2DConst::ny = 100;
const float PIC2DConst::dy = 1.0;
const float PIC2DConst::ymin = buffer * PIC2DConst::dy; 
const float PIC2DConst::ymax = PIC2DConst::ny * PIC2DConst::dy + buffer * IdealMHD2DConst::dy;


const int IdealMHD2DConst::nx = PIC2DConst::nx;
const double IdealMHD2DConst::dx = 1.0;
const double IdealMHD2DConst::xmin = 0.0 * IdealMHD2DConst::dx;
const double IdealMHD2DConst::xmax = IdealMHD2DConst::nx * IdealMHD2DConst::dx - 0.0 * IdealMHD2DConst::dx;

const int IdealMHD2DConst::ny = 500;
const double IdealMHD2DConst::dy = 1.0;
const double IdealMHD2DConst::ymin = 0.0 * IdealMHD2DConst::dy;
const double IdealMHD2DConst::ymax = IdealMHD2DConst::ny * IdealMHD2DConst::dy - 0.0 * IdealMHD2DConst::dy;


// Interface

const int Interface2DConst::convolutionCount = 3;

const int Interface2DConst::interfaceLength = 20;
const int indexOfInterfaceStartInPIC_lower = 0;
const int indexOfInterfaceStartInMHD_lower = IdealMHD2DConst::ny + 2 * buffer - Interface2DConst::interfaceLength;
const int indexOfInterfaceStartInPIC_upper = PIC2DConst::ny + 2 * buffer - Interface2DConst::interfaceLength;
const int indexOfInterfaceStartInMHD_upper = 0;

const int convolutionSizeX = PIC2DConst::nx + 2 * buffer; 
const int convolutionSizeY = IdealMHD2DConst::ny; 
const int indexOfConvolutionStartInPIC_lowerInterface = 0;
const int indexOfConvolutionStartInMHD_lowerInterface = IdealMHD2DConst::ny + 2 * buffer - convolutionSizeY; 
const int indexOfConvolutionStartInPIC_upperInterface = PIC2DConst::ny + 2 * buffer - convolutionSizeY; 
const int indexOfConvolutionStartInMHD_upperInterface = 0;

const int Interface2DConst::nx = PIC2DConst::nx;
const int Interface2DConst::ny = Interface2DConst::interfaceLength; 

thrust::host_vector<double> host_interlockingFunctionY_lower(Interface2DConst::interfaceLength, 0.0);
thrust::host_vector<double> host_interlockingFunctionY_upper(Interface2DConst::interfaceLength, 0.0);

const unsigned long long Interface2DConst::reloadParticlesTotalNum = 1000000;//PIC2DConst::numberDensityIon * PIC2DConst::nx * (Interface2DConst::interfaceLength * 2 + 0);

// PIC

const float PIC2DConst::c = 1.0f;
const float PIC2DConst::epsilon0 = 1.0f;
const float PIC2DConst::mu0 = 1.0f;
const float PIC2DConst::dOfLangdonMarderTypeCorrection = 0.001f;

const int PIC2DConst::numberDensityIon = 50;
const int PIC2DConst::numberDensityElectron = 50;

const float PIC2DConst::B0 = sqrt(static_cast<double>(PIC2DConst::numberDensityElectron)) / 1.0f;

const float PIC2DConst::mRatio = 25.0f;
const float PIC2DConst::mElectron = 1.0f;
const float PIC2DConst::mIon = PIC2DConst::mRatio * PIC2DConst::mElectron;

const float PIC2DConst::tRatio = 1.0f;
const float PIC2DConst::tElectron = (PIC2DConst::B0 * PIC2DConst::B0 / 2.0f / PIC2DConst::mu0) / (PIC2DConst::numberDensityIon + PIC2DConst::numberDensityElectron * PIC2DConst::tRatio);
const float PIC2DConst::tIon = tRatio * tElectron;

const float PIC2DConst::qRatio = -1.0f;
const float PIC2DConst::qElectron = -1.0f * sqrt(PIC2DConst::epsilon0 * PIC2DConst::tElectron / static_cast<double>(PIC2DConst::numberDensityElectron));
const float PIC2DConst::qIon = PIC2DConst::qRatio * PIC2DConst::qElectron;

const float PIC2DConst::omegaPe = sqrt(static_cast<float>(PIC2DConst::numberDensityElectron) * pow(PIC2DConst::qElectron, 2) / PIC2DConst::mElectron / PIC2DConst::epsilon0);
const float PIC2DConst::omegaPi = sqrt(static_cast<float>(PIC2DConst::numberDensityIon) * pow(PIC2DConst::qIon, 2) / PIC2DConst::mIon / PIC2DConst::epsilon0);
const float PIC2DConst::omegaCe = abs(PIC2DConst::qElectron * PIC2DConst::B0 / PIC2DConst::mElectron);
const float PIC2DConst::omegaCi = PIC2DConst::qIon * PIC2DConst::B0 / PIC2DConst::mIon;

const float PIC2DConst::debyeLength = sqrt(PIC2DConst::epsilon0 * PIC2DConst::tElectron / static_cast<double>(PIC2DConst::numberDensityElectron) / pow(PIC2DConst::qElectron, 2));
const float PIC2DConst::ionInertialLength = PIC2DConst::c / PIC2DConst::omegaPi;

const float sheatThickness = 0.5f * PIC2DConst::ionInertialLength;

float PIC2DConst::dt = 0.0f;

const unsigned long long harrisNumIon = round(PIC2DConst::nx * PIC2DConst::numberDensityIon * 2.0f * sheatThickness);
const unsigned long long backgroundNumIon = round(sqrt(betaUpstream) * PIC2DConst::nx * PIC2DConst::ny * PIC2DConst::numberDensityIon);
const unsigned long long harrisNumElectron = harrisNumIon;
const unsigned long long backgroundNumElectron = backgroundNumIon;
const unsigned long long PIC2DConst::totalNumIon = harrisNumIon + backgroundNumIon;
const unsigned long long PIC2DConst::totalNumElectron = harrisNumElectron + backgroundNumElectron;
const unsigned long long PIC2DConst::totalNumParticles = PIC2DConst::totalNumIon + PIC2DConst::totalNumElectron;

const float PIC2DConst::vThIon = sqrt(2.0f * tIon / mIon);
const float PIC2DConst::vThElectron = sqrt(2.0f * tElectron / mElectron);
const float PIC2DConst::bulkVxIon = 0.0f;
const float PIC2DConst::bulkVyIon = 0.0f;
const float PIC2DConst::bulkVzIon = -PIC2DConst::c * PIC2DConst::debyeLength / sheatThickness * sqrt(2.0f / (1.0f + 1.0f / PIC2DConst::tRatio));
const float PIC2DConst::bulkVxElectron = -PIC2DConst::bulkVxElectron / PIC2DConst::tRatio;
const float PIC2DConst::bulkVyElectron = -PIC2DConst::bulkVyElectron / PIC2DConst::tRatio;
const float PIC2DConst::bulkVzElectron = -PIC2DConst::bulkVzElectron / PIC2DConst::tRatio;

const float vThIonBackground = sqrt(sqrt(betaUpstream) * 2.0f * PIC2DConst::tIon / PIC2DConst::mIon);
const float vThElectronBackground = sqrt(sqrt(betaUpstream) * 2.0f * PIC2DConst::tElectron / PIC2DConst::mElectron);
const float bulkVxElectronBackground = 0.0f;
const float bulkVyElectronBackground = 0.0f;
const float bulkVzElectronBackground = 0.0f;
const float bulkVxIonBackground = 0.0f;
const float bulkVyIonBackground = 0.0f;
const float bulkVzIonBackground = 0.0f;


// MHD

const double IdealMHD2DConst::B0 = PIC2DConst::B0;
const double IdealMHD2DConst::rho0 = PIC2DConst::mIon * PIC2DConst::numberDensityIon + PIC2DConst::mElectron * PIC2DConst::numberDensityElectron;
const double IdealMHD2DConst::p0 = PIC2DConst::numberDensityIon * PIC2DConst::tIon + PIC2DConst::numberDensityElectron * PIC2DConst::tElectron;

const double IdealMHD2DConst::CFL = 0.7;
const double IdealMHD2DConst::gamma = 5.0 / 3.0;

double IdealMHD2DConst::dt = 0.0;

double IdealMHD2DConst::ch = 0.0; 
double IdealMHD2DConst::cp = 0.0; 
double IdealMHD2DConst::cr = 0.18; 

////////// device //////////

// PIC

__constant__ int PIC2DConst::device_totalStep;
__device__ float PIC2DConst::device_totalTime;

__constant__ float PIC2DConst::device_c;
__constant__ float PIC2DConst::device_epsilon0;
__constant__ float PIC2DConst::device_mu0;
__constant__ float PIC2DConst::device_dOfLangdonMarderTypeCorrection;
__constant__ float PIC2DConst::device_EPS;

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

__device__ float PIC2DConst::device_dt;

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

__constant__ double IdealMHD2DConst::device_rho0;
__constant__ double IdealMHD2DConst::device_B0;
__constant__ double IdealMHD2DConst::device_p0;

__constant__ double IdealMHD2DConst::device_dx;
__constant__ double IdealMHD2DConst::device_xmin;
__constant__ double IdealMHD2DConst::device_xmax;
__constant__ int IdealMHD2DConst::device_nx;

__constant__ double IdealMHD2DConst::device_dy;
__constant__ double IdealMHD2DConst::device_ymin;
__constant__ double IdealMHD2DConst::device_ymax;
__constant__ int IdealMHD2DConst::device_ny;

__constant__ double IdealMHD2DConst::device_CFL;
__constant__ double IdealMHD2DConst::device_gamma;

__device__ double IdealMHD2DConst::device_dt;

__device__ double IdealMHD2DConst::device_ch; 
__device__ double IdealMHD2DConst::device_cp; 
__device__ double IdealMHD2DConst::device_cr; 



// Interface

__constant__ double Interface2DConst::device_EPS;
__constant__ double Interface2DConst::device_PI;

__constant__ int Interface2DConst::device_interfaceLength;

__constant__ int Interface2DConst::device_nx; 
__constant__ int Interface2DConst::device_ny;  

__constant__ unsigned long long Interface2DConst::device_reloadParticlesTotalNum;


