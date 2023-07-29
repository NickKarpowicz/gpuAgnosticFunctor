cmake_minimum_required(VERSION 3.9)
project(AgnosticFunctor)
option(BUILD_CUDA "Build the CUDA version" OFF)
option(BUILD_OMP "Build the OMP version" OFF)
option(BUILD_SYCL "Build the SYCL version" OFF)

#example usage:
#cmake -DCMAKE_CUDA_HOST_COMPILER=clang++-15 -DBUILD_CUDA=ON ..
if(BUILD_CUDA)
    enable_language("CUDA")
    add_executable(demoCUDA simpleExample.cu)
endif(BUILD_CUDA)

#example usage:
# cmake -DBUILD_OMP=ON ..
if(BUILD_OMP)
    find_package(OpenMP REQUIRED)
    add_executable(demoOMP simpleExample.cpp)
    target_link_libraries(demoOMP PUBLIC OpenMP::OpenMP_CXX)
endif(BUILD_OMP)

#example usage:
# . /opt/intel/oneapi/setvars.sh
# cmake -DONEAPI_ROOT=${ONEAPI_ROOT}  -DCMAKE_CXX_COMPILER=icpx -DBUILD_SYCL=ON ..
if(BUILD_SYCL)
    set(SYCL_INCLUDE_DIR "${ONEAPI_ROOT}/compiler/latest/linux/include/sycl")
    set(SYCL_LIBRARY_DIR "${ONEAPI_ROOT}/compiler/latest/linux/lib")
    link_directories(${SYCL_LIBRARY_DIR})
    add_executable(demoSYCL simpleExample.cpp)
    add_compile_options(-fsycl -D USESYCL)
endif(BUILD_SYCL)