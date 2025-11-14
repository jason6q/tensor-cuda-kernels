#include "core/dispatch.h"
#include "tensor/tensor.h"
#include "tensor/ops.h"

using TensorFn = jqTen::Tensor(*)(const jqTen::Tensor& a, const jqTen::Tensor& b); // Generic Tensor Function signature. This will probably cover most binary ops.
extern core::OpRegistry<TensorFn> matmul_naive_registry;
extern core::OpRegistry<TensorFn> matmul_tile_registry;