#include <iostream>
#include <cuda_runtime_api.h>

#include "tensor/ops.h"
#include "tensor/tensor.h"
#include "tensor/tensor_factory.h"
#include "tensor/op_registry.h"
#include "core/dispatch.h"


void test_matmul_naive(){
    jqTen::Tensor a = jqTen::random_uniform({1,10});
    jqTen::Tensor b = jqTen::random_uniform({10,1});

    // CPU Kernel Test
    jqTen::Tensor c = matmul_naive_registry.lookup(core::DispatchKey::CPU)(a,b);
    a.print();
    b.print();
    c.print();

    // CUDA Kernel Test
    a.to(core::Device::CUDA);
    b.to(core::Device::CUDA);

    //jqTen::Tensor c = jqTen::matmul_naive_cuda(a, b);
    //jqTen::Tensor d = jqTen::matmul_tile_cuda(a, b);

}

int main(int argc, char* argv[]){
    test_matmul_naive();
    return 0;
}