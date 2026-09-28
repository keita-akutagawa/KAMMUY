#ifndef GET_INDEX_HPP
#define GET_INDEX_HPP 


template <typename T1, typename T2>
__host__ __device__
inline T1 getIndex(
    const T2& i, const T2& j,
    const T2& NX, const T2& NY
)
{
    return j + static_cast<T1>(i) * NY; 
}

#endif
