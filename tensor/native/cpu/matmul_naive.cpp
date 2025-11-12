#include <memory>
#include <omp.h>

template <typename scalar_t>
void matmul_kernel_cpu(const scalar_t* a, const scalar_t* b, scalar_t* c, int32_t m, int32_t k, int32_t n){
    // Use OMP
}

template void matmul_kernel_cpu<float>(const float*, const float*, float*, int32_t m, int32_t k, int32_t n);