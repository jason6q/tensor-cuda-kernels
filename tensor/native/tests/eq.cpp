#include "tensor/tensor.h"
#include "tensor/tensor_factory.h"

int main(int argc, char* argv[]){
    jqTen::Tensor a = jqTen::random_uniform({1024,1024});
    jqTen::Tensor b = jqTen::random_uniform({1024,1024});

    eq_kernel()

    return 0;
}