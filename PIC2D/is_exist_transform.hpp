#ifndef PIC_IS_EXIST_TRANSFORM_HPP
#define PIC_IS_EXIST_TRANSFORM_HPP

struct IsExistTransform
{
    __host__ __device__
    unsigned int operator()(const Particle& p) const {
        return p.isExist ? 1 : 0;
    }
};

#endif

