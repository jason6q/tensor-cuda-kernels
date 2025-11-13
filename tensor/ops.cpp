/**
 * Main jqTen API to call lower level kernel code
 */
#include "tensor/tensor.h"
#include "tensor/tensor_factory.h"
#include "tensor/native/cpu_kernels.h"

namespace jqTen{
    Tensor matmul_naive_cuda(const Tensor& a, const Tensor& b){
        int m = a.shape(0);
        int k = a.shape(1);
        int n = b.shape(2);

        JQ_ASSERT(k != b.shape(0), "Columns of a do not match rows of b.");

    }

    Tensor matmul_naive_cpu(const Tensor& a, const Tensor& b){
        int m = a.shape(0);
        int k = a.shape(1);
        int n = b.shape(1);
        JQ_ASSERT(k == b.shape(0), "Columns of a do not match rows of b.");
        JQ_ASSERT(a.device() == core::Device::CPU, "Tensor a is not on CPU.");
        JQ_ASSERT(b.device() == core::Device::CPU, "Tensor b is not on CPU.");

        Tensor c = jqTen::empty({m,n});
        const float* a_ptr = static_cast<const float*>(a.data());
        const float* b_ptr = static_cast<const float*>(b.data());
        float* c_ptr = static_cast<float*>(c.data());

        // TODO: Macro template select here for scalar_t
        matmul_serial_kernel_cpu<float>(a_ptr, b_ptr, c_ptr, m, k, n);

        return c;
    }

    Tensor matmul_tile_cuda(const Tensor& a, const Tensor& b){

    }
}