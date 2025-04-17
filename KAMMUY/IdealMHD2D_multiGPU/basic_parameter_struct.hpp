#ifndef BASIC_PARAMETER_STRUCT_H
#define BASIC_PARAMETER_STRUCT_H


struct BasicParameter
{
    double rho;
    double u;
    double v;
    double w;
    double bX; 
    double bY;
    double bZ;
    double p;

    __host__ __device__
    BasicParameter() : 
        rho(0.0), 
        u(0.0), 
        v(0.0), 
        w(0.0), 
        bX(0.0), 
        bY(0.0), 
        bZ(0.0), 
        p(0.0)
        {}
    
    __host__ __device__
    BasicParameter(double rho, double u, double v, double w, 
                   double bX, double bY, double bZ, double p) :
        rho(rho), 
        u(u),
        v(v), 
        w(w), 
        bX(bX), 
        bY(bY), 
        bZ(bZ), 
        p(p)
    {}
    
    __host__ __device__
    BasicParameter& operator+=(const BasicParameter& other)
    {
        rho += other.rho;
        u   += other.u;
        v   += other.v;
        w   += other.w;
        bX  += other.bX;
        bY  += other.bY;
        bZ  += other.bZ;
        p   += other.p;
        
        return *this;
    }

    __host__ __device__
    BasicParameter operator/(double scalar) const
    {
        return BasicParameter(rho / scalar, u / scalar, v / scalar, w / scalar,
                              bX / scalar, bY / scalar, bZ / scalar, p / scalar);
    }
};

#endif
