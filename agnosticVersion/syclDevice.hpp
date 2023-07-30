#include <sycl/sycl.hpp>
#include <iostream>
#define deviceFunction



class SYCLdevice{
    sycl::queue queue;
public:
    SYCLdevice(){
        queue = sycl::queue{ sycl::default_selector_v, sycl::property::queue::in_order() };
        std::cout << "Scanning for devices...\n";
        for (const auto& p : sycl::platform::get_platforms()) {
            for (const auto& d : p.get_devices()) {
                std::cout << "Found: " << d.get_info<sycl::info::device::name>() << '\n';
            }
        }
    }
    int Malloc(void** ptr, const int64_t N) {
		(*ptr) = sycl::malloc_device(N, queue.get_device(), queue.get_context());
		queue.wait();
		return 0;
	}

    void MemcpyDeviceToHost(void* dst, const void* src, const int64_t count) {
		queue.wait();
		queue.memcpy(dst, src, count);
		queue.wait();
	}

    void Free(void* block) {
		queue.wait();
		sycl::free(block, queue);
	}

    template <typename T>
    void LaunchKernel(const unsigned int Nblock, const unsigned int Nthread, const T& functor) {
        auto i = Nblock * Nthread;
        queue.submit([&](sycl::handler& h) {
            h.parallel_for(i, functor);
            });
    }
};