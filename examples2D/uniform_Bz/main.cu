#include "../../10momentMHD2D/10momentMHD2D.hpp"
#include "../../10momentMHD2D/noise_remover.hpp"
#include "../../PIC2D/PIC2D.hpp"
#include "../../Interface2D/interface2D.hpp"


__global__ void initializeMHDValue_kernel(
    const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
    const MHDFloat DX, const MHDFloat DY, 
    const MHDFloat XMIN, const MHDFloat YMIN, 
    const MHDFloat RHO0, const MHDFloat B0, const MHDFloat P0, 
    const MHDFloat sheatThickness, const MHDFloat triggerRatio, 
    MHDValue* U
)
{
    MHDUnsignedInt i = blockIdx.x * blockDim.x + threadIdx.x;
    MHDUnsignedInt j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < NX && j < NY) {
        MHDUnsignedLongLong index = getIndex<MHDUnsignedLongLong>(i, j, NX, NY);
        
        MHDFloat rho = RHO0;
        MHDFloat u   = 0.0;
        MHDFloat v   = 0.0;
        MHDFloat w   = 0.0;
        MHDFloat bX  = 0.0;
        MHDFloat bY  = 0.0; 
        MHDFloat bZ  = B0;
        MHDFloat p   = P0;

        U[index].rho = rho;
        U[index].u   = u;
        U[index].v   = v;
        U[index].w   = w;
        U[index].bX  = bX;
        U[index].bY  = bY;
        U[index].bZ  = bZ;
        U[index].pXX = p;
        U[index].pYY = p;
        U[index].pZZ = p;
        U[index].pXY = 0.0;
        U[index].pXZ = 0.0;
        U[index].pYZ = 0.0;
        U[index].psi = 0.0;
    }
}

void OROCHI2D::initializeMHDValue()
{
    MHDFloat sheatThickness = 20.0; 
    MHDFloat triggerRatio = 0.1; 

    for (MHDUnsignedInt level = 0; level < mHDGridParameter.NUMBER_OF_LEVELS; level++) {
        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid((mHDGridParameter.NX[level] + threadsPerBlock.x - 1) / threadsPerBlock.x,
                           (mHDGridParameter.NY[level] + threadsPerBlock.y - 1) / threadsPerBlock.y);

        initializeMHDValue_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            mHDGridParameter.NX[level], mHDGridParameter.NY[level], 
            mHDGridParameter.DX[level], mHDGridParameter.DY[level], 
            mHDGridParameter.XMIN[level], mHDGridParameter.YMIN[level], 
            mHDConstParameter.RHO0, mHDConstParameter.B0, mHDConstParameter.P0, 
            sheatThickness, triggerRatio, 
            thrust::raw_pointer_cast(timeIntegrators[level]->getURef().data())
        );
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch failed at initializeMHDValue_kernel: %s\n", cudaGetErrorString(err));
            printf("Program aborted.\n");
            exit(EXIT_FAILURE);
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("Kernel execution failed at initializeMHDValue_kernel: %s\n", cudaGetErrorString(err));
            printf("Program aborted.\n");
            exit(EXIT_FAILURE);
        }
    }
}


__global__ void initializePICField_kernel(
    const PICUnsignedInt NX, const PICUnsignedInt NY, 
    const PICFloat DX, const PICFloat DY, 
    const PICFloat XMIN, const PICFloat YMIN, 
    const PICFloat rho0, const PICFloat B0, const PICFloat P0, 
    ElectricField* E, MagneticField* B
)
{
    PICUnsignedInt i = blockIdx.x * blockDim.x + threadIdx.x;
    PICUnsignedInt j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < NX && j < NY) {
        PICUnsignedLongLong index = getIndex<PICUnsignedLongLong>(i, j, NX, NY);
        
        PICFloat bX = 0.0;
        PICFloat bY = 0.0; 
        PICFloat bZ = B0;
        PICFloat eX = 0.0;
        PICFloat eY = 0.0;
        PICFloat eZ = 0.0;

        E[index].eX = eX;
        E[index].eY = eY;
        E[index].eZ = eZ;
        B[index].bX = bX;
        B[index].bY = bY; 
        B[index].bZ = bZ;
    }
}

void PIC2D::initialize()
{ 
    PICUnsignedLongLong countIon = 0, countElectron = 0;
    for (MHDUnsignedInt i = 0; i < NX; i++) {
        for (MHDUnsignedInt j = 0; j < NY; j++) {
            PICFloat xminLocal = i * DX + pICGridParameter.XMIN;
            PICFloat xmaxLocal = (i + 1) * DX + pICGridParameter.XMIN;
            PICFloat yminLocal = j * DY + pICGridParameter.YMIN;
            PICFloat ymaxLocal = (j + 1) * DY + pICGridParameter.YMIN;

            PICUnsignedInt ni = pICConstParameter.NUMBER_DENSITY_ION;
            PICUnsignedInt ne = pICConstParameter.NUMBER_DENSITY_ELECTRON;

            PICFloat bulkVxIonLocal = 0.0, bulkVyIonLocal = 0.0, bulkVzIonLocal = 0.0; 
            PICFloat bulkVxElectronLocal = 0.0; 
            PICFloat bulkVyElectronLocal = 0.0; 
            PICFloat bulkVzElectronLocal = 0.0; 

            PICFloat vThIon = sqrt(pICConstParameter.P0 / 2 / ni / pICConstParameter.M_ION); 
            PICFloat vThElectron = sqrt(pICConstParameter.P0 / 2 / ne / pICConstParameter.M_ELECTRON); 
            initializeParticle.uniformPosition_maxwellDistributionVelocity_eachCell(
                xminLocal, xmaxLocal, yminLocal, ymaxLocal, 
                bulkVxIonLocal, bulkVyIonLocal, bulkVzIonLocal, 
                vThIon, vThIon, vThIon, 
                countIon, countIon + ni, 
                j + i * NY, 
                particlesIon
            ); 
            initializeParticle.uniformPosition_maxwellDistributionVelocity_eachCell(
                xminLocal, xmaxLocal, yminLocal, ymaxLocal, 
                bulkVxElectronLocal, bulkVyElectronLocal, bulkVzElectronLocal, 
                vThElectron, vThElectron, vThElectron, 
                countElectron, countElectron + ne, 
                j + i * NY + NX * NY, 
                particlesElectron
            ); 

            countIon += ni; 
            countElectron += ne; 
        }
    }
    pICConstParameter.EXIST_NUM_ION = countIon; 
    pICConstParameter.EXIST_NUM_ELECTRON = countElectron;


    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (NY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    PICFloat RHO0 = pICConstParameter.M_ION * pICConstParameter.NUMBER_DENSITY_ION + pICConstParameter.M_ELECTRON * pICConstParameter.NUMBER_DENSITY_ELECTRON;
    initializePICField_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        NX, NY, 
        DX, DY, 
        pICGridParameter.XMIN, pICGridParameter.YMIN, 
        RHO0, pICConstParameter.B0, pICConstParameter.P0, 
        thrust::raw_pointer_cast(E.data()), thrust::raw_pointer_cast(B.data())
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed at initializePICField_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed at initializePICField_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
}


__device__ static void calculateCurrentDensity(
    const MHDValue* U,
    MHDUnsignedLongLong index,
    MHDUnsignedLongLong NY, 
    MHDFloat DX, MHDFloat DY, 
    MHDFloat& jX, MHDFloat& jY, MHDFloat& jZ
)
{
    jX = (U[index + 1].bZ - U[index - 1].bZ) / (2.0 * DY);
    jY = -(U[index + NY].bZ - U[index - NY].bZ) / (2.0 * DX);
    jZ = (U[index + NY].bY - U[index - NY].bY) / (2.0 * DX)
       - (U[index + 1].bX - U[index - 1].bX) / (2.0 * DY);
}

__device__ static MHDFloat calculateEta(
    MHDFloat x, MHDFloat y, 
    MHDFloat TOTAL_TIME
)
{
    return 0.0; 
}

__device__ static void calculateOhmElectricField(
    const MHDValue* U,
    MHDUnsignedLongLong index,
    MHDUnsignedLongLong NY, 
    MHDFloat DX, MHDFloat DY, 
    MHDFloat eta,
    MHDFloat& eOhmX, MHDFloat& eOhmY, MHDFloat& eOhmZ
)
{
    MHDFloat jX, jY, jZ;
    calculateCurrentDensity(U, index, NY, DX, DY, jX, jY, jZ);
    eOhmX = eta * jX;
    eOhmY = eta * jY;
    eOhmZ = eta * jZ;
}

__device__ static void calculateHallElectricField(
    const MHDValue* U,
    MHDUnsignedLongLong index,
    MHDUnsignedLongLong NY, 
    MHDFloat DX, MHDFloat DY, 
    MHDFloat coefHall,
    MHDFloat& eHallX, MHDFloat& eHallY, MHDFloat& eHallZ
)
{
    MHDFloat jX, jY, jZ;
    calculateCurrentDensity(U, index, NY, DX, DY, jX, jY, jZ);

    MHDFloat bX  = U[index].bX;
    MHDFloat bY  = U[index].bY;
    MHDFloat bZ  = U[index].bZ;

    eHallX = coefHall * (jY * bZ - jZ * bY);
    eHallY = coefHall * (jZ * bX - jX * bZ);
    eHallZ = coefHall * (jX * bY - jY * bX);
}


__global__ static void calculateSourceTerm_kernel(
    const MHDValue* U, 
    const MHDUnsignedInt NX, const MHDUnsignedInt NY, 
    const MHDUnsignedInt BUFFER, 
    const MHDFloat DX, const MHDFloat DY, 
    const MHDFloat XMIN, const MHDFloat YMIN, 
    const MHDFloat TOTAL_TIME, 
    const MHDFloat M_ION, const MHDFloat M_ELECTRON, 
    const MHDFloat Q_ELECTRON, 
    const bool ACTIVATE_HALL_EFFECT,  
    SourceValue* source
)
{
    MHDUnsignedInt i = blockIdx.x * blockDim.x + threadIdx.x;
    MHDUnsignedInt j = blockIdx.y * blockDim.y + threadIdx.y;

    if (BUFFER <= i && i < NX - BUFFER && BUFFER <= j && j < NY - BUFFER) {
        MHDUnsignedLongLong index = getIndex<MHDUnsignedLongLong>(i, j, NX, NY);
        MHDFloat x = XMIN + i * DX; 
        MHDFloat y = YMIN + j * DY; 
        MHDFloat r = sqrt(x * x + y * y);

        SourceValue sourceValue;

        //Ohm項
        MHDFloat jX, jY, jZ;
        calculateCurrentDensity(U, index, NY, DX, DY, jX, jY, jZ);

        MHDFloat eta = calculateEta(x, y, TOTAL_TIME);

        MHDFloat eta_xp = calculateEta(x + DX, y     , TOTAL_TIME);
        MHDFloat eta_xm = calculateEta(x - DX, y     , TOTAL_TIME);
        MHDFloat eta_yp = calculateEta(x,      y + DY, TOTAL_TIME);
        MHDFloat eta_ym = calculateEta(x,      y - DY, TOTAL_TIME);

        MHDFloat eOhmX_xp, eOhmY_xp, eOhmZ_xp;
        MHDFloat eOhmX_xm, eOhmY_xm, eOhmZ_xm;
        MHDFloat eOhmX_yp, eOhmY_yp, eOhmZ_yp;
        MHDFloat eOhmX_ym, eOhmY_ym, eOhmZ_ym;
        
        calculateOhmElectricField(U, index + NY, NY, DX, DY, eta_xp, eOhmX_xp, eOhmY_xp, eOhmZ_xp);
        calculateOhmElectricField(U, index - NY, NY, DX, DY, eta_xm, eOhmX_xm, eOhmY_xm, eOhmZ_xm);
        calculateOhmElectricField(U, index + 1,  NY, DX, DY, eta_yp, eOhmX_yp, eOhmY_yp, eOhmZ_yp);
        calculateOhmElectricField(U, index - 1,  NY, DX, DY, eta_ym, eOhmX_ym, eOhmY_ym, eOhmZ_ym);
        
        MHDFloat ohmInductionX = -((eOhmZ_yp - eOhmZ_ym) / (2.0 * DY));
        MHDFloat ohmInductionY = -(-(eOhmZ_xp - eOhmZ_xm) / (2.0 * DX));
        MHDFloat ohmInductionZ = -((eOhmY_xp - eOhmY_xm) / (2.0 * DX) - (eOhmX_yp - eOhmX_ym) / (2.0 * DY));

        MHDFloat coefHall; 
        if (ACTIVATE_HALL_EFFECT) {
            MHDFloat ne = U[index].rho / (M_ION + M_ELECTRON); 
            coefHall = 1.0 / ne / abs(Q_ELECTRON); 
        } else {
            coefHall = 0.0; 
        }

        MHDFloat eHallX_xp, eHallY_xp, eHallZ_xp;
        MHDFloat eHallX_xm, eHallY_xm, eHallZ_xm;
        MHDFloat eHallX_yp, eHallY_yp, eHallZ_yp;
        MHDFloat eHallX_ym, eHallY_ym, eHallZ_ym;
        
        calculateHallElectricField(U, index + NY, NY, DX, DY, coefHall, eHallX_xp, eHallY_xp, eHallZ_xp);
        calculateHallElectricField(U, index - NY, NY, DX, DY, coefHall, eHallX_xm, eHallY_xm, eHallZ_xm);
        calculateHallElectricField(U, index + 1,  NY, DX, DY, coefHall, eHallX_yp, eHallY_yp, eHallZ_yp);
        calculateHallElectricField(U, index - 1,  NY, DX, DY, coefHall, eHallX_ym, eHallY_ym, eHallZ_ym);
        
        MHDFloat hallInductionX = -((eHallZ_yp - eHallZ_ym) / (2.0 * DY));
        MHDFloat hallInductionY = -(-(eHallZ_xp - eHallZ_xm) / (2.0 * DX));
        MHDFloat hallInductionZ = -((eHallY_xp - eHallY_xm) / (2.0 * DX) - (eHallX_yp - eHallX_ym) / (2.0 * DY));
        

        sourceValue.s4 = ohmInductionX + hallInductionX;
        sourceValue.s5 = ohmInductionY + hallInductionY;
        sourceValue.s6 = ohmInductionZ + hallInductionZ;
        sourceValue.s7 = 2.0 * eta * (jX * jX + jY * jY + jZ * jZ) / 3;
        sourceValue.s8 = 2.0 * eta * (jX * jX + jY * jY + jZ * jZ) / 3;
        sourceValue.s9 = 2.0 * eta * (jX * jX + jY * jY + jZ * jZ) / 3;

        source[index] = sourceValue; 
    }
}

void SourceTermCalculator::calculateSourceTerm(
    const thrust::device_vector<MHDValue>& U 
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mHDGridParameter.NX[level] + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (mHDGridParameter.NY[level] + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    calculateSourceTerm_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        mHDGridParameter.NX[level], mHDGridParameter.NY[level],
        mHDGridParameter.BUFFER, 
        mHDGridParameter.DX[level], mHDGridParameter.DY[level], 
        mHDGridParameter.XMIN[level], mHDGridParameter.YMIN[level], 
        mHDConstParameter.TOTAL_TIME, 
        mHDConstParameter.M_ION, mHDConstParameter.M_ELECTRON, 
        mHDConstParameter.Q_ELECTRON, 
        mHDConstParameter.ACTIVATE_HALL_EFFECT, 
        thrust::raw_pointer_cast(source.data())
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed at calculateSourceTerm_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed at calculateSourceTerm_kernel: %s\n", cudaGetErrorString(err));
        printf("Program aborted.\n");
        exit(EXIT_FAILURE);
    }
}

//--------------------------------------------------

void CustomBoundaryXLeft::apply(
    thrust::device_vector<MHDValue>& U
) 
{
}

void CustomBoundaryXRight::apply(
    thrust::device_vector<MHDValue>& U
) 
{
}

void CustomBoundaryYDown::apply(
    thrust::device_vector<MHDValue>& U
) 
{
}

void CustomBoundaryYUp::apply(
    thrust::device_vector<MHDValue>& U
) 
{
}

//--------------------------------------------------

void SMRCustomBoundaryXLeft::apply(
    const thrust::device_vector<MHDValue>& coarseUPast, 
    const thrust::device_vector<MHDValue>& coarseUNext, 
    const MHDFloat timeRatio, 
    thrust::device_vector<MHDValue>& smrU
) 
{
}


void SMRCustomBoundaryXRight::apply(
    const thrust::device_vector<MHDValue>& coarseUPast, 
    const thrust::device_vector<MHDValue>& coarseUNext, 
    const MHDFloat timeRatio, 
    thrust::device_vector<MHDValue>& smrU
) 
{
}


void SMRCustomBoundaryYDown::apply(
    const thrust::device_vector<MHDValue>& coarseUPast, 
    const thrust::device_vector<MHDValue>& coarseUNext, 
    const MHDFloat timeRatio, 
    thrust::device_vector<MHDValue>& smrU
) 
{
}


void SMRCustomBoundaryYUp::apply(
    const thrust::device_vector<MHDValue>& coarseUPast, 
    const thrust::device_vector<MHDValue>& coarseUNext, 
    const MHDFloat timeRatio, 
    thrust::device_vector<MHDValue>& smrU
) 
{
}

//--------------------------------------------------

int main()
{
    std::string mHDConstJSONFilename = "const_mhd.json";
    std::ifstream ifsMHDConst(mHDConstJSONFilename.c_str());
    nlohmann::json mHDConstJSON; 
    ifsMHDConst >> mHDConstJSON;

    std::string mHDConfigJSONFilename = "config_mhd.json";
    std::ifstream ifsMHDConfig(mHDConfigJSONFilename.c_str());
    nlohmann::json mHDConfigJSON; 
    ifsMHDConfig >> mHDConfigJSON;

    std::string mHDGridJSONFilename = "grid_mhd.json";
    std::ifstream ifsMHDGrid(mHDGridJSONFilename.c_str());
    nlohmann::json mHDGridJSON; 
    ifsMHDGrid >> mHDGridJSON;

    std::string pICConstJSONFilename = "const_pic.json";
    std::ifstream ifsPICConst(pICConstJSONFilename.c_str());
    nlohmann::json pICConstJSON; 
    ifsPICConst >> pICConstJSON;

    std::string pICGridJSONFilename = "grid_pic.json";
    std::ifstream ifsPICGrid(pICGridJSONFilename.c_str());
    nlohmann::json pICGridJSON; 
    ifsPICGrid >> pICGridJSON;

    std::string interfaceConstJSONFilename = "const_interface.json";
    std::ifstream ifsInterfaceConst(interfaceConstJSONFilename.c_str());
    nlohmann::json interfaceConstJSON; 
    ifsInterfaceConst >> interfaceConstJSON;

    std::string interfaceGridJSONFilename = "grid_interface.json";
    std::ifstream ifsInterfaceGrid(interfaceGridJSONFilename.c_str());
    nlohmann::json interfaceGridJSON; 
    ifsInterfaceGrid >> interfaceGridJSON;

    OROCHI2D oROCHI2D(mHDConfigJSON, mHDConstJSON, mHDGridJSON);
    PIC2D pIC2D(pICConstJSON, pICGridJSON);
    MHDConstParameter& mHDConstParameter = oROCHI2D.getMHDConstParameterRef();
    MHDGridParameter& mHDGridParameter = oROCHI2D.getMHDGridParameterRef();
    PICConstParameter& pICConstParameter = pIC2D.getPICConstParameterRef();
    PICGridParameter& pICGridParameter = pIC2D.getPICGridParameterRef();
    const MHDUnsignedInt pICEmbeddedLevel = 0; 
    NoiseRemover2D noiseRemover2D(
        pICEmbeddedLevel, 
        mHDConstParameter, mHDGridParameter
    );
    Interface2D interface2D(
        interfaceConstJSON, interfaceGridJSON, 
        pICEmbeddedLevel, 
        mHDConstParameter, mHDGridParameter, 
        pICConstParameter, pICGridParameter
    );
    InterfaceConstParameter& interfaceConstParameter = interface2D.getInterfaceConstParameterRef();
    InterfaceGridParameter& interfaceGridParameter = interface2D.getInterfaceGridParameterRef();


    for (MHDUnsignedInt level = 0; level < mHDGridParameter.NUMBER_OF_LEVELS; level++) {
        std::cout << "MHD grid layer " << level << ": " << mHDGridParameter.NX[level] << " X " << mHDGridParameter.NY[level] << std::endl; 
    }
    std::cout << "PIC grid: " << pICGridParameter.NX << " X " << pICGridParameter.NY << std::endl; 
    std::cout << "PIC total number of particles: " << pICConstParameter.TOTAL_NUM_ION + pICConstParameter.TOTAL_NUM_ELECTRON << std::endl;
    
    size_t free_mem = 0;
    size_t total_mem = 0;
    cudaError_t status = cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "Free memory: " << free_mem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Total memory: " << total_mem / (1024 * 1024) << " MB" << std::endl;


    oROCHI2D.initializeMHDValue(); 
    pIC2D.initialize();

    pIC2D.calculateFullMoments(); 
    thrust::device_vector<MagneticField>& B = pIC2D.getBRef();
    thrust::device_vector<ZerothMoment>& zerothMomentIon = pIC2D.getZerothMomentIonRef(); 
    thrust::device_vector<ZerothMoment>& zerothMomentElectron = pIC2D.getZerothMomentElectronRef(); 
    thrust::device_vector<FirstMoment>& firstMomentIon = pIC2D.getFirstMomentIonRef(); 
    thrust::device_vector<FirstMoment>& firstMomentElectron = pIC2D.getFirstMomentElectronRef(); 
    thrust::device_vector<SecondMoment>& secondMomentIon = pIC2D.getSecondMomentIonRef(); 
    thrust::device_vector<SecondMoment>& secondMomentElectron = pIC2D.getSecondMomentElectronRef(); 
    interface2D.calculateSpaceAveragedPICtoMHDParameters(
        B, 
        zerothMomentIon, zerothMomentElectron, 
        firstMomentIon, firstMomentElectron, 
        secondMomentIon, secondMomentElectron
    );
    thrust::device_vector<MHDValue>& U = oROCHI2D.getTimeIntegratorsRef()[pICEmbeddedLevel]->getURef(); 
    interface2D.sendPICtoMHD(U);
    interface2D.sendMHDtoPIC_particle(
        U, 
        zerothMomentIon, zerothMomentElectron, 
        firstMomentIon, firstMomentElectron, 
        secondMomentIon, secondMomentElectron, 
        0, 
        pIC2D.getParticlesIonRef(), pIC2D.getParticlesElectronRef()
    );
    
    std::ofstream logfile(mHDConstParameter.SAVE_DIRNAME + "/log_" + mHDConstParameter.SAVE_FILENAME_WITHOUT_STEP + ".txt");
    
    const PICUnsignedInt totalSubstep = round(sqrt(pICConstParameter.M_ION / pICConstParameter.M_ELECTRON))
                                      * interfaceGridParameter.GRID_SIZE_RATIO; 
    mHDConstParameter.DT = pICConstParameter.DT * totalSubstep; //MHDのDTは固定することにする
    for (MHDUnsignedInt step = 0; step < mHDConstParameter.TOTAL_STEP + 1; step++) {
        mHDConstParameter.CURRENT_STEP = step; 
        pICConstParameter.CURRENT_STEP = step; 

        // output
        if (step % mHDConstParameter.RECORD_STEP == 0) {
            std::cout << std::to_string(step) << " step done : total time is "
                      << std::setprecision(4) << step * totalSubstep * pICConstParameter.DT * pICConstParameter.OMEGA_PE
                      << " [omega_pe * t]"
                      << std::endl;
        }
        if (step % mHDConstParameter.RECORD_STEP == 0) {
            logfile << std::setprecision(6) << mHDConstParameter.TOTAL_TIME << std::endl;
            pIC2D.saveParticle();
            pIC2D.saveFields();
            pIC2D.saveZerothMoments();
            pIC2D.saveFirstMoments();
            pIC2D.saveSecondMoments();
            oROCHI2D.save();
        }


        // STEP1 : MHD step
        oROCHI2D.oneStep(step);
        thrust::device_vector<MHDValue>& UPast = oROCHI2D.getTimeIntegratorsRef()[pICEmbeddedLevel]->getUPastRef(); 
        thrust::device_vector<MHDValue>& UNext = oROCHI2D.getTimeIntegratorsRef()[pICEmbeddedLevel]->getURef(); 

        
        // STEP2 : PIC step & send MHD to PIC
        for (PICUnsignedInt substep = 0; substep < totalSubstep; substep++) {
            interface2D.calculateTimeInterpolatedU(
                UPast, UNext, substep, totalSubstep
            ); 
            thrust::device_vector<MHDValue>& timeInterpolatedU = interface2D.getTimeInterpolatedURef(); 

            PICUnsignedLongLong seedForReload = substep + step * totalSubstep;
            pIC2D.oneStep(
                interface2D, 
                timeInterpolatedU, 
                seedForReload
            );

            pICConstParameter.TOTAL_TIME += pICConstParameter.DT;
        }


        // STEP3 : send PIC to MHD
        //pIC2D.calculateFullMoments(); 
        thrust::device_vector<MagneticField>& B = pIC2D.getBRef();
        thrust::device_vector<ZerothMoment>& zerothMomentIon = pIC2D.getZerothMomentIonRef(); 
        thrust::device_vector<ZerothMoment>& zerothMomentElectron = pIC2D.getZerothMomentElectronRef(); 
        thrust::device_vector<FirstMoment>& firstMomentIon = pIC2D.getFirstMomentIonRef(); 
        thrust::device_vector<FirstMoment>& firstMomentElectron = pIC2D.getFirstMomentElectronRef(); 
        thrust::device_vector<SecondMoment>& secondMomentIon = pIC2D.getSecondMomentIonRef(); 
        thrust::device_vector<SecondMoment>& secondMomentElectron = pIC2D.getSecondMomentElectronRef(); 
        interface2D.calculateSpaceAveragedPICtoMHDParameters(
            B, 
            zerothMomentIon, zerothMomentElectron, 
            firstMomentIon, firstMomentElectron, 
            secondMomentIon, secondMomentElectron
        );
        interface2D.sendPICtoMHD(UNext);


        // remove noise 
        if (step % interfaceConstParameter.CONVOLUTION_INTERVAL == 0) {
            noiseRemover2D.convolutionU(UNext);
            oROCHI2D.getTimeIntegratorsRef()[pICEmbeddedLevel]->getBoundaryRef().applyUForAllDirection(UNext);
        }

        mHDConstParameter.TOTAL_TIME += mHDConstParameter.DT;


        // check simulation crash
        if (oROCHI2D.isCrashed()) {
            logfile << std::setprecision(6) << mHDConstParameter.TOTAL_TIME << std::endl;
            pIC2D.saveParticle();
            pIC2D.saveFields();
            pIC2D.saveZerothMoments();
            pIC2D.saveFirstMoments();
            pIC2D.saveSecondMoments();
            oROCHI2D.save();
            std::cout << "Calculation stopped! : " << step << " steps" << std::endl;
            break;
        }
    }

    return 0;
}



