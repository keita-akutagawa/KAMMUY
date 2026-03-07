#ifndef IS_EXIST_TRANSFORM_H
#define IS_EXIST_TRANSFORM_H

struct IsExistTransform
{
    __host__ __device__
    unsigned int operator()(const Particle& p) const {
        return p.isExist ? 1 : 0;
    }
};

#endif

