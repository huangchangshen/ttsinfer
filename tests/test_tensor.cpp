#include <cassert>
#include <iostream>
#include "ttsinfer.h"

using namespace ttsinfer;

void test_tensor_basic() {
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

    std::cout << "[OK] test_tensor_basic passed\n";
}

void test_tensor_view_ops() {
    Tensor x = Tensor::empty({2, 3, 4}, DType::F32);
    float* px = x.data();
    for (int i = 0; i < 24; ++i) {
        px[i] = static_cast<float>(i);
    }

    Tensor r = reshape(x, {6, 4});
    assert(r.unsafe_get_impl() != nullptr);
    assert(r.shape().size() == 2);
    assert(r.shape()[0] == 6);
    assert(r.shape()[1] == 4);
    assert(r.stride().size() == 2);
    assert(r.stride()[0] == 4);
    assert(r.stride()[1] == 1);
    assert(r.data() == x.data());
    assert(r.data()[7] == 7.0f);

    Tensor p = permute(x, {1, 0, 2});
    assert(p.shape().size() == 3);
    assert(p.shape()[0] == 3);
    assert(p.shape()[1] == 2);
    assert(p.shape()[2] == 4);
    assert(p.stride().size() == 3);
    assert(p.stride()[0] == 4);
    assert(p.stride()[1] == 12);
    assert(p.stride()[2] == 1);
    assert(p.data() == x.data());
    assert(p.data()[13] == 13.0f);

    Tensor t = transpose(x);
    assert(t.shape().size() == 3);
    assert(t.shape()[0] == 2);
    assert(t.shape()[1] == 4);
    assert(t.shape()[2] == 3);
    assert(t.stride()[0] == 12);
    assert(t.stride()[1] == 1);
    assert(t.stride()[2] == 4);
    assert(t.data() == x.data());

    Tensor t01 = transpose(x, 0, 1);
    assert(t01.shape()[0] == 3);
    assert(t01.shape()[1] == 2);
    assert(t01.shape()[2] == 4);
    assert(t01.stride()[0] == 4);
    assert(t01.stride()[1] == 12);
    assert(t01.stride()[2] == 1);

    std::cout << "[OK] test_tensor_view_ops passed\n";
}

int main() {
    test_tensor_basic();
    test_tensor_view_ops();

    std::cout << "[OK] test_tensor passed\n";
    return 0;
}
