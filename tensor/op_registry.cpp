#include <vector>
#include <iostream>

#include "core/dispatch.h"
#include "tensor/op_registry.h"
#include "tensor/tensor.h"
#include "tensor/ops.h"

/**
 * We'll register all operators here.
 * Should just do a code-gen for this.
 */
core::OpRegistry<TensorFn> matmul_naive_registry;
core::OpRegistry<TensorFn> matmul_tile_registry;

static bool _ = [](){
    // Setup

    // Matmul Naive Registration
    matmul_naive_registry.add_kernel(core::DispatchKey::CUDA, &jqTen::matmul_naive_cuda);
    std::cout << "Matmul Naive CUDA Function Registered." << std::endl; // Replace with macro print later
    matmul_naive_registry.add_kernel(core::DispatchKey::CPU, &jqTen::matmul_naive_cpu);
    std::cout << "Matmul Naive CPU Function Registered." << std::endl; // Replace with macro print later

    // Matmul Tile Registration
    matmul_tile_registry.add_kernel(core::DispatchKey::CUDA, &jqTen::matmul_tile_cuda);
    std::cout << "Matmul Tile Function Registered." << std::endl; // Replace with macro print later

    return true;
}();