#include <sycl/sycl.hpp>
#define deviceFunction

class SYCLdevice{
    sycl::queue stream{ sycl::default_selector_v, sycl::property::queue::in_order() };
public:
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
        void deviceLaunch(const unsigned int Nblock, const unsigned int Nthread, const T& functor) {
            auto i = Nblock * Nthread;
            stream.submit([&](sycl::handler& h) {
                h.parallel_for(i, functor);
                });
    }
};