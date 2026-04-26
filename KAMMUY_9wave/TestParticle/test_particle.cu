#include "test_particle.hpp"


TestParticle::TestParticle()
    : particles(TestParticleConst::numParticle), 
      E(IdealMHD2DConst::nx * IdealMHD2DConst::ny),
      B(IdealMHD2DConst::nx * IdealMHD2DConst::ny), 

      host_particles(TestParticleConst::numParticle)
{
}


__device__
ParticleField getParticleFields_testParticle(
    const MagneticField* B,
    const ElectricField* E, 
    const int gridSizeRatio, 
    Particle& particle
)
{
    ParticleField particleField;

    double cx1, cx2; 
    int xIndex1, xIndex2;
    double xOverDx;
    double cy1, cy2; 
    int yIndex1, yIndex2;
    double yOverDy;

    xOverDx = (particle.x - IdealMHD2DConst::device_xmin) / gridSizeRatio;
    yOverDy = (particle.y - IdealMHD2DConst::device_ymin) / gridSizeRatio;

    xIndex1 = floor(xOverDx);
    xIndex2 = xIndex1 + 1;
    xIndex2 = (xIndex2 == IdealMHD2DConst::device_nx) ? 0 : xIndex2;
    yIndex1 = floor(yOverDy);
    yIndex2 = yIndex1 + 1;
    yIndex2 = (yIndex2 == IdealMHD2DConst::device_ny) ? 0 : yIndex2;

    cx1 = xOverDx - xIndex1;
    cx2 = 1.0 - cx1;
    cy1 = yOverDy - yIndex1;
    cy2 = 1.0 - cy1;

    unsigned long long index11 = yIndex1 + IdealMHD2DConst::device_ny * xIndex1; 
    unsigned long long index12 = yIndex2 + IdealMHD2DConst::device_ny * xIndex1; 
    unsigned long long index21 = yIndex1 + IdealMHD2DConst::device_ny * xIndex2; 
    unsigned long long index22 = yIndex2 + IdealMHD2DConst::device_ny * xIndex2; 

    particleField.bX += B[index11].bX * cx2 * cy2;
    particleField.bX += B[index12].bX * cx2 * cy1;
    particleField.bX += B[index21].bX * cx1 * cy2;
    particleField.bX += B[index22].bX * cx1 * cy1;

    particleField.bY += B[index11].bY * cx2 * cy2;
    particleField.bY += B[index12].bY * cx2 * cy1;
    particleField.bY += B[index21].bY * cx1 * cy2;
    particleField.bY += B[index22].bY * cx1 * cy1;

    particleField.bZ += B[index11].bZ * cx2 * cy2;
    particleField.bZ += B[index12].bZ * cx2 * cy1;
    particleField.bZ += B[index21].bZ * cx1 * cy2;
    particleField.bZ += B[index22].bZ * cx1 * cy1;

    particleField.eX += E[index11].eX * cx2 * cy2;
    particleField.eX += E[index12].eX * cx2 * cy1;
    particleField.eX += E[index21].eX * cx1 * cy2;
    particleField.eX += E[index22].eX * cx1 * cy1;

    particleField.eY += E[index11].eY * cx2 * cy2;
    particleField.eY += E[index12].eY * cx2 * cy1;
    particleField.eY += E[index21].eY * cx1 * cy2;
    particleField.eY += E[index22].eY * cx1 * cy1;

    particleField.eZ += E[index11].eZ * cx2 * cy2;
    particleField.eZ += E[index12].eZ * cx2 * cy1;
    particleField.eZ += E[index21].eZ * cx1 * cy2;
    particleField.eZ += E[index22].eZ * cx1 * cy1;

    return particleField;
}


__global__ void pushVelocity_testParticle_kernel(
    Particle* particles, const MagneticField* B, const ElectricField* E, 
    double q, double m, unsigned long long existNumSpecies, double dt, 
    const int gridSizeRatio
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        double qOverMTimesDtOver2;
        double tmpForT, tmpForS, tmp1OverC2;
        double vx, vy, vz, gamma;
        double tx, ty, tz;
        double sx, sy, sz;
        double vxMinus, vyMinus, vzMinus;
        double vx0, vy0, vz0;
        double vxPlus, vyPlus, vzPlus; 
        double bx, by, bz;
        double ex, ey, ez;
        ParticleField particleField;

        qOverMTimesDtOver2 = q / m * dt / 2.0;
        tmp1OverC2 = 1.0 / (PIC2DConst::device_c * PIC2DConst::device_c);

        vx = particles[i].vx;
        vy = particles[i].vy;
        vz = particles[i].vz;
        gamma = particles[i].gamma;

        particleField = getParticleFields_testParticle(B, E, gridSizeRatio, particles[i]);

        bx = particleField.bX;
        by = particleField.bY;
        bz = particleField.bZ; 
        ex = particleField.eX;
        ey = particleField.eY; 
        ez = particleField.eZ;

        tmpForT = qOverMTimesDtOver2 / gamma;
        tx = tmpForT * bx;
        ty = tmpForT * by;
        tz = tmpForT * bz;

        tmpForS = 2.0 / (1.0 + tx * tx + ty * ty + tz * tz);
        sx = tmpForS * tx;
        sy = tmpForS * ty;
        sz = tmpForS * tz;

        vxMinus = vx + qOverMTimesDtOver2 * ex;
        vyMinus = vy + qOverMTimesDtOver2 * ey;
        vzMinus = vz + qOverMTimesDtOver2 * ez;

        vx0 = vxMinus + (vyMinus * tz - vzMinus * ty);
        vy0 = vyMinus + (vzMinus * tx - vxMinus * tz);
        vz0 = vzMinus + (vxMinus * ty - vyMinus * tx);

        vxPlus = vxMinus + (vy0 * sz - vz0 * sy);
        vyPlus = vyMinus + (vz0 * sx - vx0 * sz);
        vzPlus = vzMinus + (vx0 * sy - vy0 * sx);

        vx = vxPlus + qOverMTimesDtOver2 * ex;
        vy = vyPlus + qOverMTimesDtOver2 * ey;
        vz = vzPlus + qOverMTimesDtOver2 * ez;
        gamma = sqrt(1.0 + (vx * vx + vy * vy + vz * vz) * tmp1OverC2);

        particles[i].vx = vx;
        particles[i].vy = vy;
        particles[i].vz = vz;
        particles[i].gamma = gamma;
    }
}


void TestParticle::pushVelocity(const double dt)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((TestParticleConst::numParticle + threadsPerBlock.x - 1) / threadsPerBlock.x);

    pushVelocity_testParticle_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particles.data()), 
        thrust::raw_pointer_cast(B.data()), 
        thrust::raw_pointer_cast(E.data()), 
        PIC2DConst::qIon, PIC2DConst::mIon, 
        TestParticleConst::numParticle, dt, 
        Interface2DConst::gridSizeRatio
    );
    cudaDeviceSynchronize();
}


__global__
void pushPosition_testParticle_kernel(
    Particle* particles, 
    const unsigned long long existNumSpecies, 
    const double dt, 
    const double xmin, const double xmax, 
    const double ymin, const double ymax
)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < existNumSpecies) {
        
        double vx, vy, vz, gamma;
        double xPast, yPast, zPast;
        double x, y, z;
        double dtOverGamma;

        vx = particles[i].vx;
        vy = particles[i].vy;
        vz = particles[i].vz;
        gamma = particles[i].gamma;
        xPast = particles[i].x;
        yPast = particles[i].y;
        zPast = particles[i].z;

        dtOverGamma = dt / gamma;
        x = xPast + dtOverGamma * vx;
        y = yPast + dtOverGamma * vy;
        z = zPast + dtOverGamma * vz;

        if (x <= xmin) x = xmin; 
        if (x >= xmax) x = xmax; 
        if (y <= ymin) y = ymin; 
        if (y >= ymax) y = ymax;

        particles[i].x = x;
        particles[i].y = y;
        particles[i].z = z;

    }
}


void TestParticle::pushPosition(const double dt)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((TestParticleConst::numParticle + threadsPerBlock.x - 1) / threadsPerBlock.x);

    pushPosition_testParticle_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(particles.data()), 
        TestParticleConst::numParticle, 
        dt, 
        IdealMHD2DConst::xmin, IdealMHD2DConst::xmax, 
        IdealMHD2DConst::ymin, IdealMHD2DConst::ymax
    );
    cudaDeviceSynchronize();
}


void TestParticle::oneStep()
{
    pushPosition(PIC2DConst::dt / 2.0);

    pushVelocity(PIC2DConst::dt);

    pushPosition(PIC2DConst::dt / 2.0);
}   


void TestParticle::saveField(
    std::string directoryName, 
    std::string filenameWithoutStep, 
    int step
)
{
    thrust::host_vector<MagneticField> host_B(IdealMHD2DConst::nx * IdealMHD2DConst::ny);
    thrust::host_vector<ElectricField> host_E(IdealMHD2DConst::nx * IdealMHD2DConst::ny);
        
    host_B = B; 
    host_E = E; 

    std::string filenameB, filenameE;
    filenameB = directoryName + "/"
             + filenameWithoutStep + "_B_" + std::to_string(step)
             + ".bin";
    filenameE = directoryName + "/"
             + filenameWithoutStep + "_E_" + std::to_string(step)
             + ".bin";

    std::ofstream ofsB(filenameB, std::ios::binary);
    ofsB << std::fixed << std::setprecision(6);
    for (int i = 0; i < IdealMHD2DConst::nx; i++) {
        for (int j = 0; j < IdealMHD2DConst::ny; j++) {
            unsigned long long index = j + IdealMHD2DConst::ny * i;
            ofsB.write(reinterpret_cast<const char*>(&host_B[index].bX), sizeof(double));
            ofsB.write(reinterpret_cast<const char*>(&host_B[index].bY), sizeof(double));
            ofsB.write(reinterpret_cast<const char*>(&host_B[index].bZ), sizeof(double));
        }
    }
    std::ofstream ofsE(filenameE, std::ios::binary);
    ofsE << std::fixed << std::setprecision(6);
    for (int i = 0; i < IdealMHD2DConst::nx; i++) {
        for (int j = 0; j < IdealMHD2DConst::ny; j++) {
            unsigned long long index = j + IdealMHD2DConst::ny * i;
            ofsE.write(reinterpret_cast<const char*>(&host_E[index].eX), sizeof(double));
            ofsE.write(reinterpret_cast<const char*>(&host_E[index].eY), sizeof(double));
            ofsE.write(reinterpret_cast<const char*>(&host_E[index].eZ), sizeof(double));
        }
    }
}


void TestParticle::saveParticle(
    std::string directoryname, 
    std::string filenameWithoutStep, 
    int step
)
{
    host_particles = particles;

    std::string filenameX;
    std::string filenameV;

    filenameX = directoryname + "/"
             + filenameWithoutStep + "_x_" + std::to_string(step)
             + "_test_particle"
             + ".bin";
    filenameV = directoryname + "/"
             + filenameWithoutStep + "_v_" + std::to_string(step)
             + "_test_particle"
             + ".bin";

    double x, y, z;
    double vx, vy, vz;

    std::ofstream ofsX(filenameX, std::ios::binary);
    ofsX << std::fixed << std::setprecision(6);
    std::ofstream ofsV(filenameV, std::ios::binary);
    ofsV << std::fixed << std::setprecision(6);
    for (unsigned long long i = 0; i < particles.size(); i++) {
        x = host_particles[i].x;
        y = host_particles[i].y;
        z = host_particles[i].z;
        vx = host_particles[i].vx / host_particles[i].gamma;
        vy = host_particles[i].vy / host_particles[i].gamma;
        vz = host_particles[i].vz / host_particles[i].gamma;

        ofsX.write(reinterpret_cast<const char*>(&x), sizeof(double));
        ofsX.write(reinterpret_cast<const char*>(&y), sizeof(double));
        ofsX.write(reinterpret_cast<const char*>(&z), sizeof(double));

        ofsV.write(reinterpret_cast<const char*>(&vx), sizeof(double));
        ofsV.write(reinterpret_cast<const char*>(&vy), sizeof(double));
        ofsV.write(reinterpret_cast<const char*>(&vz), sizeof(double));
    }
}



