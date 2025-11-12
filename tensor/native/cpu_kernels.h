#include <memory>

template<typename scalar_t>
std::unique_ptr<scalar_t> arange_kernel_cpu(int32_t n);

template<typename scalar_t>
void matmul_kernel_cpu(const scalar_t* a, const scalar_t* b, scalar_t* c, int32_t m, int32_t k, int32_t n);