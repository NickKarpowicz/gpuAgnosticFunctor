#include <iostream>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>

__global__ void setValues(
    float* A, 
    float* B)
    {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    A[i] = 0.1f * static_cast<float>(i);
    B[i] = 2.1f * A[i] + 1.0f;
}

__global__ void multiplyAbyB(
    float* A, 
    float* B)
    {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    A[i] *= B[i];
}

__global__ void divideAbyB(
    float* A, 
    float* B)
    {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    A[i] /= B[i];
}

int main()
{
    unsigned int vectorSize = 8192*65536;
    unsigned int repetitions = 32;
    unsigned int Nthreads = 64;
    unsigned int Nblocks = vectorSize/Nthreads;
    float* deviceA;
    float* deviceB;
    std::vector<float> cpuA(vectorSize, 0.0f);

    auto timerBegin = std::chrono::high_resolution_clock::now();
    
    cudaMalloc(&deviceA, vectorSize * sizeof(float));
    cudaMalloc(&deviceB, vectorSize * sizeof(float));
    setValues<<<Nblocks, Nthreads>>>(deviceA, deviceB);
    for(int i = 0; i < repetitions; ++i){
        multiplyAbyB<<<Nblocks, Nthreads>>>(deviceA, deviceB);
        divideAbyB<<<Nblocks, Nthreads>>>(deviceA, deviceB);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(cpuA.data(),deviceA,vectorSize*sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(deviceA);
    cudaFree(deviceB);
    
    auto timerEnd = std::chrono::high_resolution_clock::now();
    
    std::cout << 
    "Took " << 
    1e-3 * static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>
        (timerEnd - timerBegin).count()) << 
    " ms\n";
    
    for(int i = 0; i<3; ++i){
        std::cout << i << ": " << cpuA[i] <<"\n"; 
    }
}