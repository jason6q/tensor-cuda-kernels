#pragma once
#include "unordered_map"
#include "string"

class Tensor;

namespace core{
    /**
     * We will use a DispatchKey to select the correct kernels to use.
     * These keys should be composite in some sense on a tensor in the DispatchKeySet.
     * At the moment these only represent backend kernels but can evolve to more later on.
     * Pytorch has Autograd -> Backend -> Wrapper/Modes
     */
    enum class DispatchKey {
        UNDEFINED = 0, 
        CPU, 
        CUDA,
        CUTE,
        CUTLASS
    };

    /**
     * Object that contains a set of dispatch keys.
     */
    class DispatchKeySet {

    };

    class Dispatcher {
    };

    /**
     * Maintain an operator registry for every operator type with a specific signature
     */
    template<class Fn>
    struct OpRegistry{
        using MapFn = std::unordered_map<DispatchKey, Fn>;
        MapFn table;

        void add_kernel(DispatchKey dispatch_key, Fn fn){
            table[dispatch_key] = fn;
        }

        // This might be a bit too slow since we have a linear lookup.
        // Use the bit-wise method like PyTorch?
        Fn lookup(DispatchKey dispatch_key){
            auto it = table.find(dispatch_key);
            if(table.end() == it){
                throw std::runtime_error("Unable to find kernel.");
            }
            return it->second;
        }
    };
}