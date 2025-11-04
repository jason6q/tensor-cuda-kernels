#include "tensor/tensor.h"
#include "tensor/tensor_factory.h"
#include "core/device.h"

int main(int argc, char* argv[]){
    // CPU Comparison
    jqTen::Tensor a = jqTen::empty({32,32});
    jqTen::Tensor b = jqTen::empty({32,32});

    // CUDA Comparison
    jqTen::Tensor c = jqTen::empty({32,32});
    c.to(core::Device::CUDA);
    jqTen::Tensor d = jqTen::empty({32,32});
    d.to(core::Device::CUDA);

    return 0;
}