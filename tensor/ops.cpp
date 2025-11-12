/**
 * Main jqTen API to call lower level kernel code
 */
#include "tensor/tensor.h"

namespace jqTen{
    Tensor matmul_naive_cuda(const Tensor& a, const Tensor& b){
        int m = a.shape(0);
        int k = a.shape(1);
        int n = b.shape(2);

        JQ_ASSERT(k != b.shape(0), "Columns of a do not match rows of b.");

    }

    Tensor matmul_tile_cuda(const Tensor& a, const Tensor& b){

    }
}