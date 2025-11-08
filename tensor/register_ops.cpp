#include <vector>
#include <iostream>

#include "core/dispatch.h"
#include "tensor/tensor.h"
#include "tensor/ops.h"

/**
 * We'll register all operators here.
 * Should just do a code-gen for this.
 */
static bool _ = [](){
    std::cout << "Matmul Naive Function Registered." << std::endl; // Replace with macro print later
    using TensorFn = jqTen::Tensor(*)(const jqTen::Tensor& a, const jqTen::Tensor& b); // Generic Tensor Function signature. This will probably cover most binary ops.
    core::OpRegistry<TensorFn> matmul_naive_registry;
    matmul_naive_registry.add_kernel(core::DispatchKey::CUDA, &jqTen::matmul_naive_cuda);

    return true;
}();