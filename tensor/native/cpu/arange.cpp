#include <memory>

template <typename scalar_t>
std::unique_ptr<scalar_t> arange_kernel_cpu(int32_t n){
    std::unique_ptr<scalar_t> buf = make_unique<scalar_t[]>(n);
    for(int i = 0; i < n; ++i){
        scalar_t* = buf->get();
    }

    return buf;
}