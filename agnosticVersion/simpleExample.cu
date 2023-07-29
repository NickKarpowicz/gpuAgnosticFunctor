#include <iostream>
#include <chrono>
#include <vector>
#ifdef __CUDACC__
    #include "cudaDeviceHeader.cuh"
    typedef CUDAdevice device;
//#elif defined USESYCL
#else
    #include "syclDeviceHeader.hpp"
    typedef SYCLdevice device;    

    std::cout << "Scanning for devices...\n";
    for (const auto& p : sycl::platform::get_platforms()) {
        for (const auto& d : p.get_devices()) {
            std::cout << "Found: " << d.get_info<sycl::info::device::name>() << '\n';
        }
    }
// #else
//     #include "ompDeviceHeader.hpp"
//     typedef OMPdevice device;
#endif

class setValues{
public:
    float* A; 
    float* B;
    deviceFunction void operator()(const unsigned int i) const {
        A[i] = 0.1f * static_cast<float>(i);
        B[i] = 2.1f * A[i] + 1.0f;
    }
};

class multiplyAbyB{
public:
    float* A;
    float* B;
    deviceFunction void operator()(const unsigned int i) const{
        A[i] *= B[i];
    }
};

class divideAbyB{
public:
    float* A; 
    float* B;
    deviceFunction void operator()(const unsigned int i) const{
        A[i] /= B[i];
    }
};

int main(){
    unsigned int vectorSize = 8192*65536;
    unsigned int repetitions = 32;
    unsigned int Nthreads = 64;
    unsigned int Nblocks = vectorSize/Nthreads;
    float* deviceA;
    float* deviceB;
    std::vector<float> cpuA(vectorSize, 0.0f);

    auto timerBegin = std::chrono::high_resolution_clock::now();
    device d;
    d.Malloc((void**)&deviceA, vectorSize * sizeof(float));
    d.Malloc((void**)&deviceB, vectorSize * sizeof(float));
    d.LaunchKernel(Nblocks, Nthreads, setValues{deviceA, deviceB});
    for(unsigned int i = 0; i < repetitions; ++i){
        d.LaunchKernel(Nblocks, Nthreads, multiplyAbyB{deviceA, deviceB});
        d.LaunchKernel(Nblocks, Nthreads, divideAbyB{deviceA, deviceB});
    }
    d.MemcpyDeviceToHost(cpuA.data(),deviceA,vectorSize*sizeof(float));
    d.Free(deviceA);
    d.Free(deviceB);

    auto timerEnd = std::chrono::high_resolution_clock::now();
    std::cout << 
    "Took " << 
    1e-3 * static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>
        (timerEnd - timerBegin).count()) << 
        " ms\n";

    for(int i = 0; i<10; ++i){
        std::cout << i << ": " << cpuA[i] <<"\n"; 
    }
}