#include <stdlib.h>
#include <cstring>
#define deviceFunction
class OMPdevice{
public:
    int Malloc(void** ptr, size_t N){
        *ptr = malloc(N);
        return *ptr == nullptr;
    }
    void MemcpyDeviceToHost(void* dst, void* src, size_t count){
        memcpy(dst, src, count);
    }
    void Free(void* ptr){
        free(ptr);
    } 
    template <typename T>
    void LaunchKernel(const unsigned int Nblock, const unsigned int Nthread, const T& functor) const {
    #pragma omp parallel for
    for(int i = 0; i<static_cast<int>(Nthread); i++){
        const int offset = i * static_cast<int>(Nthread);
        for(int j = offset; j< static_cast<int>(offset + Nthread); functor(j++)){}
    }
    }
};