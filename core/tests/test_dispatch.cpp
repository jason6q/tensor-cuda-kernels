#include <iostream>
#include "core/dispatch.h"

void test_fn(){
    std::cout << "Test function!" << std::endl;
}

int main(int argc, char* argv[]){
    using TestFn = void(*)();
    core::OpRegistry<TestFn> op_registry;
    op_registry.add_kernel(core::DispatchKey::CPU, &test_fn);
    op_registry.lookup(core::DispatchKey::CPU)();
    return 0;
}