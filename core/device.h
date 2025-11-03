#pragma once
#include <cstdint>
namespace core{
    // Forward Declarations

    enum class DType { 
        FP32
    };

    enum class Device {
        CPU,
        CUDA,
    };
}