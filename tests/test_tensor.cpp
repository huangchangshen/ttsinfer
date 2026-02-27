#include <cassert>
#include <iostream>
#include "ttsinfer.h"

using namespace ttsinfer;

int main() {
    Tensor a = Tensor::empty({2, 3}, DType::F32);

    assert(a.unsafe_get_impl() != nullptr);
    assert(a.shape().size() == 2);
    assert(a.shape()[0] == 2);
    assert(a.shape()[1] == 3);


    float* data = a.data();
    for (int i = 0; i < 6; ++i) {
        data[i] = static_cast<float>(i);
    }

    for (int i = 0; i < 6; ++i) {
        assert(a.data()[i] == static_cast<float>(i));
    }

    Tensor b = a;

    assert(b.unsafe_get_impl() == a.unsafe_get_impl());
    assert(b.data()[3] == 3.0f);

    b.data()[0] = 100.f;
    assert(a.data()[0] == 100.f);

    Tensor c = std::move(a);
    // a is now in a moved-from state
    // but our simple Tensor impl with shared_ptr (or IntrusivePtr) 
    // might not nullify 'a' if move ctor isn't explicitly defined to do so.
    // However, std::move just casts to rvalue. 
    // If Tensor has default move ctor/assign, it will move IntrusivePtr.
    
    // Let's check if 'a' became null. 
    // IntrusivePtr move ctor should nullify source.
    
    assert(c.unsafe_get_impl() != nullptr);
    assert(a.unsafe_get_impl() == nullptr);  // moved-from

    Tensor d = c;
    assert(d.unsafe_get_impl() == c.unsafe_get_impl());

    c.dump();

    std::cout << "[OK] test_tensor passed\n";
    return 0;
}