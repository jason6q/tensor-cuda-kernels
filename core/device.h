#pragma once
#include <cstdint>
namespace core{
    // Forward Declarations

    enum class DType: int32_t { 
        FP32
    };

    enum class Device: int32_t {
        CPU,
        CUDA,
    };
}