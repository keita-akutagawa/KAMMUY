
struct ReloadParticlesData
{
    int number; 
    float u;
    float v;
    float w;
    float vth;
    bool hasData;

    __host__ __device__
    ReloadParticlesData() : 
        number(0), 
        u(0.0f), 
        v(0.0f), 
        w(0.0f),
        vth(0.0f), 
        hasData(false)
        {}
    
};


