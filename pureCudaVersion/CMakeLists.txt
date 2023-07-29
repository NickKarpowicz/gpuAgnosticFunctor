#usage:
#cmake --fresh -DCMAKE_CUDA_HOST_COMPILER=clang++-15 -DCMAKE_CUDA_ARCHITECTURES=75  ..
cmake_minimum_required(VERSION 3.8)
project(AgnosticFunctor)
enable_language("CUDA")
add_executable(demoCUDA simpleExampleOnlyCUDA.cu)
