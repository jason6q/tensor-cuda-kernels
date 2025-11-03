#include "unordered_map"
#include "string"

namespace core{
    // Global registry for dispatch -> Kernel maps
    // E.g: Add Operation -> CPU / CUDA CUTE / CUTLASS Kernel based off DispatchKey.
    std::unordered_map<std::string, std::unordered_map<DispatchKey, void*>> op_registry;
    add_registry(std::string op_name, DispatchKey dispatch_key, void* fn){
        // Does not exist
        if(op_register.end() == op_registry.find(op_name)){
            auto op_table = std::unordered_map<DispatchKey, void*>();
            op_registry.insert({op_name, op_table});
            op_registry[op_name].insert({dispatch_key, fn});
        }
        else{
            if(op_registry.end() == op_registry[op_name].find(dispatch_key)){
                op_registry.insert({dispatch_key, fn});
            }
            else{
                // Nothing.
            }
        }
    }

    /**
     * We will use a DispatchKey to select the correct kernels to use.
     * These keys should be composite in some sense on a tensor in the DispatchKeySet.
     * At the moment these only represent backend kernels but can evolve to more later on.
     * Pytorch has Autograd -> Backend -> Wrapper/Modes
     */
    enum DispatchKey : uint8_t {
        UNDEFINED = 0, 
        CPU, 
        CUDA,
        CUTE,
        CUTLASS
    }

    /**
     * Object that contains a set of dispatch keys.
     */
    class DispatchKeySet {

    };

    class Dispatcher {
    };
}