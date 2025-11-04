/**
 * Tensor C++ API Frontend
 * 
 */
#include <random>
#include <vector>
#include <cstdint>
#include <algorithm>

#include <cuda_runtime_api.h>

#include "core/data_ptr.h"
#include "core/device.h"
#include "tensor/tensor.h"
namespace jqTen{
    size_t compute_nbytes(const std::vector<int32_t>& shape, core::DType dtype){
        // Calculate bytes
        size_t size = 1;
        for(int32_t i = 0; i < shape.size(); ++i){
            size *= shape[i];
        }
        if(dtype == core::DType::FP32){
            size *= sizeof(float);
        }
        else{
            size *= sizeof(float);
        }

        return size;
    }

    Tensor::Tensor(
        const std::vector<int32_t>& shape, 
        core::DType dtype,
        core::Device device): 
        shape_(shape), dtype_(dtype), device_(device), nbytes_(compute_nbytes(shape_, dtype_)),
        data_ptr_(device == core::Device::CUDA ? core::DataPtr::cuda(nbytes_, true) : core::DataPtr::cpu(nbytes_, true, 64)) // 64 byte alignment
        {

    }

    void Tensor::to(core::Device device){
        this->data_ptr_.to(device);
        this->device_ = device;
    }

}