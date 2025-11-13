#include <memory>
#include <omp.h>

template <typename scalar_t>
void matmul_serial_kernel_cpu(const scalar_t* a, const scalar_t* b, scalar_t* c, int32_t m, int32_t k, int32_t n){
    for(int32_t mi = 0; mi< m; ++mi){
        for(int32_t ji = 0; ji < n; ++ji){
            float val = 0;
            for(int32_t ki = 0; ki < k; ++ki){
                val += a[mi*m + k] * b[ji + mi*n];
            }
            c[mi*m + ji] = val;
        }
    }
}

template <typename scalar_t>
void matmul_omp_kernel_cpu(const scalar_t* a, const scalar_t* b, scalar_t* c, int32_t m, int32_t k, int32_t n){
    // Use OMP, launch threads
    //#pragma omp parallel for
    //for(int32_t i = 0; i < k; ++i){

    //}
}

template void matmul_serial_kernel_cpu<float>(const float*, const float*, float*, int32_t m, int32_t k, int32_t n);
template void matmul_omp_kernel_cpu<float>(const float*, const float*, float*, int32_t m, int32_t k, int32_t n);