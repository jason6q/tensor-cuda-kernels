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
    // Setup
    using TensorFn = jqTen::Tensor(*)(const jqTen::Tensor& a, const jqTen::Tensor& b); // Generic Tensor Function signature. This will probably cover most binary ops.

    // Matmul Naive Registration
    core::OpRegistry<TensorFn> matmul_naive_registry;
    matmul_naive_registry.add_kernel(core::DispatchKey::CUDA, &jqTen::matmul_naive_cuda);
    std::cout << "Matmul Naive Function Registered." << std::endl; // Replace with macro print later

    // Matmul Tile Registration
    core::OpRegistry<TensorFn> matmul_tile_registry;
    matmul_tile_registry.add_kernel(core::DispatchKey::CUDA, &jqTen::matmul_tile_cuda);
    std::cout << "Matmul Tile Function Registered." << std::endl; // Replace with macro print later

    return true;
}();