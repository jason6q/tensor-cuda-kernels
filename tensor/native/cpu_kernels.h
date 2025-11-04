#include <memory>

template<typename scalar_t>
std::unique_ptr<scalar_t> arange_kernel_cpu(int32_t n);