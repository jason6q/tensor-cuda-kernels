#include "core/device.h"
#include "tensor/tensor.h"
#include "tensor/tensor_factory.h"
#include "tensor/native/cu_kernels.cuh"
#include "tensor/native/cpu_kernels.h"

namespace jqTen{
    Tensor arange(int32_t n, core::Device device){
        if(device == core::Device::CUDA){
            Tensor a = empty({n});
            a.to(core::Device::CUDA);
            float* a_buf = static_cast<float*>(a.data());

            // TODO: Change this soon.
            dim3 gridDim = dim3((n - 1) / 32 + 1);
            dim3 blockDim = dim3(32);
            arange_kernel<float><<<gridDim, blockDim, 0>>>(a_buf, n);

            return a;
        }
        else if(device == core::Device::CPU){
            Tensor a = empty({n});
        }
        else{
            JQ_ASSERT(true, "Device not supported!");
        }
    }
}