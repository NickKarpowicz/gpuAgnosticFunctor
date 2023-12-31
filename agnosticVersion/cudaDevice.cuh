#include <cuda_runtime.h>
#define deviceFunction __device__
template <typename T>

__global__ static void deviceLaunchFunctorWrapper(const T functor){
    functor(threadIdx.x + blockIdx.x * blockDim.x);
}

class CUDAdevice{
public:
    int Malloc(void** ptr, size_t N){
        return cudaMalloc(ptr, N);
    }
    void MemcpyDeviceToHost(void* dst, void* src, size_t count){
        cudaDeviceSynchronize();
        cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
    }
    void Free(void* ptr){
        cudaFree(ptr);
    } 
    template <typename T>
    void LaunchKernel(const unsigned int Nblock, const unsigned int Nthread, const T& functor) const {
        deviceLaunchFunctorWrapper<<<Nblock, Nthread>>>(functor);
    }
};