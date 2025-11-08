#pragma once
/**
 * Any function that lets you build a tensor with various
 * initializations.
 */
#include <random>
#include <cuda_runtime_api.h>

#include "core/data_ptr.h"
#include "core/device.h"
#include "tensor/tensor.h"
#include "core/dispatch.h"

namespace jqTen{
    Tensor empty(
        const std::vector<int32_t>& shape, 
        std::optional<core::DType> dtype = std::nullopt,
        std::optional<core::Device> device = std::nullopt
    );

    Tensor random_uniform(
        const std::vector<int32_t>& shape, 
        std::optional<core::DType> dtype = std::nullopt,
        std::optional<int64_t> seed = std::nullopt,
        std::optional<core::Device> device = std::nullopt
    );

    Tensor arange(
        int32_t n,
        std::optional<core::DType> dtype = std::nullopt,
        std::optional<core::Device> device = std::nullopt
    );
}

/**
 * Operator Registration.
 * For every operator define their type signature then create an OpRegistry
 * struct for them.
 */
//static bool _ = []{
//    using EmptyFn = void(*)();
//    OpRegistry<>
//}();