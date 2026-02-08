#include <cassert>
#include <iostream>
#include "safetensors.h"
using namespace ttsinfer::safetensors;

int main() {
    SafeOpen f("model.safetensors");

    assert(f.contains("linear.weight"));

    auto t = f.getTensor("linear.weight");

    assert(t.dtype() == DType::F32);
    assert(t.shape()[0] == 2);
    assert(t.shape()[1] == 3);

    const float* p = static_cast<const float*>(t.data());
    std::cout << p[0] << " " << p[1] << " " << p[2] << std::endl;

    std::cout << "[OK] test_safetensors passed\n";
    return 0;
}