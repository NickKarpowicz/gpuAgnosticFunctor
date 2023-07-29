mkdir build
cd build
cmake -DCMAKE_CUDA_HOST_COMPILER=clang++-15 -DBUILD_CUDA=ON ..
make
cmake --fresh -DBUILD_OMP=ON ..
make
. /opt/intel/oneapi/setvars.sh
cmake --fresh -DONEAPI_ROOT=${ONEAPI_ROOT}  -DCMAKE_CXX_COMPILER=icpx -DBUILD_SYCL=ON ..
make