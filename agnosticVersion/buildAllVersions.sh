mkdir build
cd build
cmake -DCMAKE_CUDA_HOST_COMPILER=clang++-15 -DBUILD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 ..
make
cmake --fresh -DBUILD_OMP=ON ..
make
. /opt/intel/oneapi/setvars.sh
cmake --fresh -DONEAPI_ROOT=${ONEAPI_ROOT}  -DCMAKE_CXX_COMPILER=icpx -DCMAKE_LINKER=icx -DBUILD_SYCL=ON ..
make