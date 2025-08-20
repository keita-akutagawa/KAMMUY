#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
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


std::string directoryName = "/cfca-work/akutagawakt/paper4/results_mrx_grid_size_ratio=20_forcefree";
std::string filenameWithoutStep = "mrx";
std::ofstream logfile(    directoryName + "/log_mrx.txt"       );
std::ofstream mpifile_MHD(directoryName + "/mpilog_mhd_mrx.txt");
std::ofstream mpifile_PIC(directoryName + "/mpilog_pic_mrx.txt");
std::ofstream mpifile_Interface(directoryName + "/mpilog_interface_mrx.txt");


const int bufferMHD = 5; 
const int bufferPIC = 5; 

const std::string IdealMHD2DConst::MTXfilename = "/home/akutagawakt/paper4/examples2D/mrx_grid_size_ratio=20_forcefree/poisson_symmetric.mtx";
const std::string IdealMHD2DConst::jsonFilenameForSolver = "/home/akutagawakt/paper4/examples2D/mrx_grid_size_ratio=20_forcefree/PCG_W.json";

const int IdealMHD2DConst::totalStep = 10000;
const int PIC2DConst::totalStep = -1;
const int recordStep = 5;
const bool isParticleRecord = false;
const int particleRecordStep = PIC2DConst::totalStep;

double PIC2DConst::totalTime = 0.0f;
double IdealMHD2DConst::totalTime = 0.0;

const int Interface2DConst::gridSizeRatio = 20; 

const double Interface2DConst::EPS = 0.0001;
const double Interface2DConst::PI = 3.14159265358979;
const double PIC2DConst::EPS = 0.001;
const double IdealMHD2DConst::EPS = 1e-20;
const double IdealMHD2DConst::PI = 3.14159265358979;

double IdealMHD2DConst::eta = 0.0;
double IdealMHD2DConst::viscosity = 0.0;

const int PIC2DConst::nx = 20000;
const double PIC2DConst::dx = 1.0f;
const double PIC2DConst::xmin = 0.0f * PIC2DConst::dx; 
const double PIC2DConst::xmax = PIC2DConst::nx * PIC2DConst::dx + PIC2DConst::xmin;

const int PIC2DConst::ny = 800;
const double PIC2DConst::dy = 1.0f;
const double PIC2DConst::ymin = 0.0f * PIC2DConst::dy; 
const double PIC2DConst::ymax = PIC2DConst::ny * PIC2DConst::dy + PIC2DConst::ymin;


const int IdealMHD2DConst::nx = PIC2DConst::nx / Interface2DConst::gridSizeRatio;
const double IdealMHD2DConst::dx = PIC2DConst::dx * Interface2DConst::gridSizeRatio;
const double IdealMHD2DConst::xmin = 0.0 * IdealMHD2DConst::dx;
const double IdealMHD2DConst::xmax = IdealMHD2DConst::nx * IdealMHD2DConst::dx + IdealMHD2DConst::xmin;

const int IdealMHD2DConst::ny = 10000 / Interface2DConst::gridSizeRatio;
const double IdealMHD2DConst::dy = PIC2DConst::dy * Interface2DConst::gridSizeRatio;
const double IdealMHD2DConst::ymin = 0.0 * IdealMHD2DConst::dy;
const double IdealMHD2DConst::ymax = IdealMHD2DConst::ny * IdealMHD2DConst::dy + IdealMHD2DConst::ymin;


// Interface

const int Interface2DConst::convolutionCount = 1;

const int Interface2DConst::interfaceLength = -1; //使わないこと
const double Interface2DConst::deltaForInterlockingFunction = 2; 
const int Interface2DConst::indexOfInterfaceStartInMHD = IdealMHD2DConst::ny / 2 - PIC2DConst::ny / 2 / Interface2DConst::gridSizeRatio;

const int Interface2DConst::nx = PIC2DConst::nx;
const int Interface2DConst::ny = Interface2DConst::deltaForInterlockingFunction; 

const unsigned long long Interface2DConst::reloadParticlesTotalNum = 10000000;

// PIC

const double PIC2DConst::c = 1.0f;
const double PIC2DConst::epsilon0 = 1.0f;
const double PIC2DConst::mu0 = 1.0f;
const double PIC2DConst::dOfLangdonMarderTypeCorrection = 0.001f;

const int PIC2DConst::numberDensityIon = 20;
const int PIC2DConst::numberDensityElectron = 20;

const double PIC2DConst::B0 = sqrt(static_cast<double>(PIC2DConst::numberDensityElectron)) / 1.0f;

const double PIC2DConst::mRatio = 25.0f;
const double PIC2DConst::mElectron = 1.0f;
const double PIC2DConst::mIon = PIC2DConst::mRatio * PIC2DConst::mElectron;

const double beta = 0.25;
const double PIC2DConst::tRatio = 1.0f;
const double PIC2DConst::tElectron = beta * (PIC2DConst::B0 * PIC2DConst::B0 / 2.0f / PIC2DConst::mu0) / (PIC2DConst::numberDensityIon + PIC2DConst::numberDensityElectron * PIC2DConst::tRatio);
const double PIC2DConst::tIon = tRatio * tElectron;

const double PIC2DConst::qRatio = -1.0f;
const double PIC2DConst::qElectron = -1.0f * sqrt(PIC2DConst::epsilon0 * PIC2DConst::tElectron / static_cast<double>(PIC2DConst::numberDensityElectron));
const double PIC2DConst::qIon = PIC2DConst::qRatio * PIC2DConst::qElectron;

const double PIC2DConst::omegaPe = sqrt(static_cast<double>(PIC2DConst::numberDensityElectron) * pow(PIC2DConst::qElectron, 2) / PIC2DConst::mElectron / PIC2DConst::epsilon0);
const double PIC2DConst::omegaPi = sqrt(static_cast<double>(PIC2DConst::numberDensityIon) * pow(PIC2DConst::qIon, 2) / PIC2DConst::mIon / PIC2DConst::epsilon0);
const double PIC2DConst::omegaCe = abs(PIC2DConst::qElectron * PIC2DConst::B0 / PIC2DConst::mElectron);
const double PIC2DConst::omegaCi = PIC2DConst::qIon * PIC2DConst::B0 / PIC2DConst::mIon;

const double PIC2DConst::debyeLength = sqrt(PIC2DConst::epsilon0 * PIC2DConst::tElectron / static_cast<double>(PIC2DConst::numberDensityElectron) / pow(PIC2DConst::qElectron, 2));
const double PIC2DConst::ionInertialLength = PIC2DConst::c / PIC2DConst::omegaPi;

const double triggerRatio = 0.1f;
const double sheatThickness = 2.0f * PIC2DConst::ionInertialLength; 

double PIC2DConst::dt = 0.0f;

const unsigned long long PIC2DConst::totalNumIon = round(PIC2DConst::nx * PIC2DConst::ny * PIC2DConst::numberDensityIon);
const unsigned long long PIC2DConst::totalNumElectron = round(PIC2DConst::nx * PIC2DConst::ny * PIC2DConst::numberDensityElectron);
const unsigned long long PIC2DConst::totalNumParticles = PIC2DConst::totalNumIon + PIC2DConst::totalNumElectron;

const double PIC2DConst::vThIon = sqrt(PIC2DConst::tIon / PIC2DConst::mIon);
const double PIC2DConst::vThElectron = sqrt(PIC2DConst::tElectron / PIC2DConst::mElectron);
const double PIC2DConst::bulkVxIon = 0.0f;
const double PIC2DConst::bulkVyIon = 0.0f;
const double PIC2DConst::bulkVzIon = 0.0f;
const double PIC2DConst::bulkVxElectron = 0.0f;
const double PIC2DConst::bulkVyElectron = 0.0f;
const double PIC2DConst::bulkVzElectron = 0.0f;


// MHD

const double IdealMHD2DConst::B0 = PIC2DConst::B0;
const double IdealMHD2DConst::rho0 = PIC2DConst::mIon * PIC2DConst::numberDensityIon + PIC2DConst::mElectron * PIC2DConst::numberDensityElectron;
const double IdealMHD2DConst::p0 = PIC2DConst::numberDensityIon * PIC2DConst::tIon + PIC2DConst::numberDensityElectron * PIC2DConst::tElectron;

const double IdealMHD2DConst::CFL = 0.7;
const double IdealMHD2DConst::gamma = 5.0 / 3.0;

double IdealMHD2DConst::dt = 0.0;


////////// device //////////

// PIC

__constant__ int PIC2DConst::device_totalStep;
__device__ double PIC2DConst::device_totalTime;

__constant__ double PIC2DConst::device_c;
__constant__ double PIC2DConst::device_epsilon0;
__constant__ double PIC2DConst::device_mu0;
__constant__ double PIC2DConst::device_dOfLangdonMarderTypeCorrection;
__constant__ double PIC2DConst::device_EPS;

__constant__ int PIC2DConst::device_numberDensityIon;
__constant__ int PIC2DConst::device_numberDensityElectron;

__constant__ double PIC2DConst::device_B0;

__constant__ double PIC2DConst::device_mRatio;
__constant__ double PIC2DConst::device_mElectron;
__constant__ double PIC2DConst::device_mIon;

__constant__ double PIC2DConst::device_tRatio;
__constant__ double PIC2DConst::device_tElectron;
__constant__ double PIC2DConst::device_tIon;

__constant__ double PIC2DConst::device_qRatio;
__constant__ double PIC2DConst::device_qElectron;
__constant__ double PIC2DConst::device_qIon;

__constant__ double PIC2DConst::device_omegaPe;
__constant__ double PIC2DConst::device_omegaPi;
__constant__ double PIC2DConst::device_omegaCe;
__constant__ double PIC2DConst::device_omegaCi;

__constant__ double PIC2DConst::device_debyeLength;
__constant__ double PIC2DConst::device_ionInertialLength;

__constant__ int PIC2DConst::device_nx;
__constant__ double PIC2DConst::device_dx;
__constant__ double PIC2DConst::device_xmin;
__constant__ double PIC2DConst::device_xmax;

__constant__ int PIC2DConst::device_ny;
__constant__ double PIC2DConst::device_dy;
__constant__ double PIC2DConst::device_ymin;
__constant__ double PIC2DConst::device_ymax;

__device__ double PIC2DConst::device_dt;

__constant__ double PIC2DConst::device_vThIon;
__constant__ double PIC2DConst::device_vThElectron;
__constant__ double PIC2DConst::device_bulkVxElectron;
__constant__ double PIC2DConst::device_bulkVyElectron;
__constant__ double PIC2DConst::device_bulkVzElectron;
__constant__ double PIC2DConst::device_bulkVxIon;
__constant__ double PIC2DConst::device_bulkVyIon;
__constant__ double PIC2DConst::device_bulkVzIon;



// MHD

__constant__ double IdealMHD2DConst::device_EPS;
__constant__ double IdealMHD2DConst::device_PI;

__device__ double IdealMHD2DConst::device_eta; 
__device__ double IdealMHD2DConst::device_viscosity; 

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

// Interface

__constant__ double Interface2DConst::device_EPS;
__constant__ double Interface2DConst::device_PI;

__constant__ int Interface2DConst::device_gridSizeRatio;

__constant__ int Interface2DConst::device_interfaceLength;
__constant__ double Interface2DConst::device_deltaForInterlockingFunction; 
__constant__ int Interface2DConst::device_indexOfInterfaceStartInMHD;

__constant__ int Interface2DConst::device_convolutionCount;

__constant__ int Interface2DConst::device_nx; 
__constant__ int Interface2DConst::device_ny;  

__constant__ unsigned long long Interface2DConst::device_reloadParticlesTotalNum;
