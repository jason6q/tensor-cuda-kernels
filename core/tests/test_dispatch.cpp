#include <iostream>
#include "core/dispatch.h"

void test_fn(){
    std::cout << "Test function!" << std::endl;
}

int main(int argc, char* argv[]){
    core::add_registry("test", core::DispatchKey::CPU, &test_fn);
    core::op_registry["test"][core::DispatchKey::CPU]();
    return 0;
}