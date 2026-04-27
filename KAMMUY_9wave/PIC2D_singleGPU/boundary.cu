#include "boundary.hpp"


void BoundaryPIC::boundaryParticle(
    thrust::device_vector<Particle>& particlesIon, 
    unsigned long long& existNumIon, 
    thrust::device_vector<Particle>& particlesElectron, 
    unsigned long long& existNumElectron
)
{
    boundaryParticleXLeft(particlesIon, existNumIon);
    boundaryParticleXRight(particlesIon, existNumIon);
    boundaryParticleYDown(particlesIon, existNumIon);
    boundaryParticleYUp(particlesIon, existNumIon);

    boundaryParticleXLeft(particlesElectron, existNumElectron);
    boundaryParticleXRight(particlesElectron, existNumElectron);
    boundaryParticleYDown(particlesElectron, existNumElectron);
    boundaryParticleYUp(particlesElectron, existNumElectron);
}


void BoundaryPIC::boundaryB(
    thrust::device_vector<MagneticField>& B
)
{
    boundaryFieldXLeft(B);
    boundaryFieldXRight(B);
    boundaryFieldYDown(B);
    boundaryFieldYUp(B);
}


void BoundaryPIC::boundaryE(
    thrust::device_vector<ElectricField>& E
)
{
    boundaryFieldXLeft(E);
    boundaryFieldXRight(E);
    boundaryFieldYDown(E);
    boundaryFieldYUp(E);
}


void BoundaryPIC::boundaryCurrent(
    thrust::device_vector<CurrentField>& current
)
{
    boundaryFieldXLeft(current);
    boundaryFieldXRight(current); 
    boundaryFieldYDown(current);
    boundaryFieldYUp(current);
}


void BoundaryPIC::boundaryZerothMoment(
    thrust::device_vector<ZerothMoment>& zerothMoment
)
{
    boundaryFieldXLeft(zerothMoment);
    boundaryFieldXRight(zerothMoment);
    boundaryFieldYDown(zerothMoment);
    boundaryFieldYUp(zerothMoment);
}


void BoundaryPIC::boundaryFirstMoment(
    thrust::device_vector<FirstMoment>& firstMoment
)
{
    boundaryFieldXLeft(firstMoment);
    boundaryFieldXRight(firstMoment);
    boundaryFieldYDown(firstMoment);
    boundaryFieldYUp(firstMoment);
}


void BoundaryPIC::boundarySecondMoment(
    thrust::device_vector<SecondMoment>& secondMoment
)
{
    boundaryFieldXLeft(secondMoment);
    boundaryFieldXRight(secondMoment);
    boundaryFieldYDown(secondMoment);
    boundaryFieldYUp(secondMoment);
}

