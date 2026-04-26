#include "boundary.hpp"


void BoundaryPIC::boundaryParticle(
    thrust::device_vector<Particle>& particlesIon, 
    thrust::device_vector<Particle>& particlesElectron
)
{
    boundaryParticleXLeft(particlesIon, particlesElectron);
    boundaryParticleXRight(particlesIon, particlesElectron);
    boundaryParticleYDown(particlesIon, particlesElectron);
    boundaryParticleYUp(particlesIon, particlesElectron);
}


void BoundaryPIC::boundaryB(
    thrust::device_vector<MagneticField>& B
)
{
    boundaryBXLeft(B);
    boundaryBXRight(B);
    boundaryBYDown(B);
    boundaryBYUp(B);
}


void BoundaryPIC::boundaryE(
    thrust::device_vector<ElectricField>& E
)
{
    boundaryEXLeft(E);
    boundaryEXRight(E);
    boundaryEYDown(E);
    boundaryEYUp(E);
}


void BoundaryPIC::boundaryCurrent(
    thrust::device_vector<CurrentField>& current
)
{
    boundaryCurrentXLeft(current);
    boundaryCurrentXRight(current); 
    boundaryCurrentYDown(current);
    boundaryCurrentYUp(current);
}


void BoundaryPIC::boundaryZerothMoment(
    thrust::device_vector<ZerothMoment>& zerothMoment
)
{
    boundaryZerothMomentXLeft(zerothMoment);
    boundaryZerothMomentXRight(zerothMoment);
    boundaryZerothMomentYDown(zerothMoment);
    boundaryZerothMomentYUp(zerothMoment);
}


void BoundaryPIC::boundaryFirstMoment(
    thrust::device_vector<FirstMoment>& firstMoment
)
{
    boundaryFirstMomentXLeft(firstMoment);
    boundaryFirstMomentXRight(firstMoment);
    boundaryFirstMomentYDown(firstMoment);
    boundaryFirstMomentYUp(firstMoment);
}


void BoundaryPIC::boundarySecondMoment(
    thrust::device_vector<SecondMoment>& secondMoment
)
{
    boundarySecondMomentXLeft(secondMoment);
    boundarySecondMomentXRight(secondMoment);
    boundarySecondMomentYDown(secondMoment);
    boundarySecondMomentYUp(secondMoment);
}

