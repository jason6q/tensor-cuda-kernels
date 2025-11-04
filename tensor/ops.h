/**
 * All supported operators. Strictly C++/CUDA no Torch API.
 * 
 */

 // Should I return a tensor?
 // Or a raw buffer and have that wrapped?
#include <iostream>
#include "tensor.h"
#include "core/dispatch.h"

namespace jqTen{
    
    Tensor matmul_naive(const Tensor& a, const Tensor& b);
    Tensor matmul_tile(const Tensor& a, const Tensor& b);
}