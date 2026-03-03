#include "ttsinfer.h"
#include "graph.h"
#include "module.h"
#include <cassert>
#include <iostream>

using namespace ttsinfer;
using namespace ttsinfer::nn;

void test_linear() {
    std::cout << "=== Linear Unit Test ===\n";

    TensorImpl* last_impl = nullptr;
    {
        Graph g;
        Linear linear(16, 32);

        StateDict sd;
        linear.state_dict(sd);

        assert(sd.count("weight") == 1);
        assert(sd.count("bias") == 1);

        std::cout << "Parameters registered: ";
        for (auto& [k, _] : sd) {
            std::cout << k << " ";
        }
        std::cout << "\n";

        Tensor x = Tensor::empty({4, 16}, DType::F32);

        Tensor y = linear(x);

        assert(y.shape().size() == 2);
        assert(y.shape()[0] == 4);
        assert(y.shape()[1] == 32);

        std::cout << "Output shape OK: ["
                  << y.shape()[0] << ", "
                  << y.shape()[1] << "]\n";
 
        g.build_forward(y);
        g.dump();

        last_impl = y.unsafe_get_impl();

        assert(last_impl->refcount_.load() > 0);
        std::cout << "Refcount inside scope: "
                  << last_impl->refcount_.load() << "\n";
    }

    std::cout << "Refcount after scope: "
              << last_impl->refcount_.load() << "\n";

    assert(last_impl->refcount_.load() == 0);

    std::cout << "[OK] Linear test passed\n";
}

int main() {
    test_linear();
    return 0;
}
