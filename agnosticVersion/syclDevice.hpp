#include <sycl/sycl.hpp>
#include <iostream>
#define deviceFunction



class SYCLdevice{
    sycl::queue stream;
public:
    SYCLdevice(){
        stream = sycl::queue{ sycl::default_selector_v, sycl::property::queue::in_order() };
        std::cout << "Scanning for devices...\n";
        for (const auto& p : sycl::platform::get_platforms()) {
            for (const auto& d : p.get_devices()) {
                std::cout << "Found: " << d.get_info<sycl::info::device::name>() << '\n';
            }
        }
    }
    int Malloc(void** ptr, const int64_t N) {
		(*ptr) = sycl::malloc_device(N, stream.get_device(), stream.get_context());
		stream.wait();
		return 0;
	}

    void MemcpyDeviceToHost(void* dst, const void* src, const int64_t count) {
		stream.wait();
		stream.memcpy(dst, src, count);
		stream.wait();
	}

    void Free(void* block) {
		stream.wait();
		sycl::free(block, stream);
	}

    template <typename T>
    void LaunchKernel(const unsigned int Nblock, const unsigned int Nthread, const T& functor) {
        auto i = Nblock * Nthread;
        stream.submit([&](sycl::handler& h) {
            h.parallel_for(i, functor);
            });
    }
};