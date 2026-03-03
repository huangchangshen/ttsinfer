#include "graph.h"
#include "ttsinfer.h"
#include <iostream>
#include <cassert>

using namespace ttsinfer;

void test_graph_build() {
    std::cout << "=== Tensor Graph Unit Test ===\n";

    {
        Graph graph;
        Tensor A = Tensor::empty({2, 3}, DType::F32);
        Tensor B = Tensor::empty({3, 4}, DType::F32);

        Tensor C = matmul(A, B);
        Tensor D = Tensor::empty({2, 4}, DType::F32);

        Tensor E = add(D, C);
        graph.build_forward(E);
        graph.dump();
    }
    
    std::cout << "[OK] test_tensor passed\n";
}

void test_graph_compute() {
    std::cout << "=== Tensor Graph Compute Test ===\n";

    {
        Graph graph;

        Tensor A = Tensor::ones({2, 3}, DType::F32);

        Tensor B = Tensor::ones({3, 4}, DType::F32);

        Tensor C = matmul(A, B);

        graph.build_forward(C);
        graph.compute_forward();
        graph.dump();

        float* out = static_cast<float*>(C.data());

        for (int i = 0; i < 2 * 4; ++i) {
            if (out[i] != 3.0f) {
                std::cerr << "Mismatch at index "
                          << i << " value=" << out[i]
                          << " expected=3\n";
                assert(false);
            }
        }

        std::cout << "Output:\n";
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 4; ++j) {
                std::cout << out[i * 4 + j] << " ";
            }
            std::cout << "\n";
        }
    }

    std::cout << "[OK] test_graph_compute passed\n";
}

int main() {
    
    test_graph_build();
    test_graph_compute();

    return 0;
}
