#include <vector>

#include "core/dispatch.h"
#include "tensor/tensor.h"

/**
 * We'll register all operators here.
 * Should just do a code-gen for this.
 */
static bool _ = []{
    using MatmulNaiveFn = void(*)(const Tensor& a, const Tensor& b);
    OpRegistry<EmptyFn> matmul_naive_registry;
    matmul_naive_registry.add_kernel(core::DispatchKey::CUDA, )
}();