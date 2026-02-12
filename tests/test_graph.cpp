#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>
#include "ttsinfer.h"
#include "graph.h"
#include "backend-cpu.h"

using namespace ttsinfer;

void test_simple_graph() {
    std::cout << "Running test_simple_graph...\n";
    
    Tensor a = Tensor::zeros({1}, DType::F32);
    a.name("A");
    Tensor b = Tensor::zeros({1}, DType::F32);
    b.name("B");
    
    Tensor c = add(&a, &b);
    c.name("C");
    
    Graph g = Graph::build_forward(&c);
    
    const auto& leafs = g.leafs();
    bool found_a = false, found_b = false;
    for(auto* t : leafs) {
        if(t->name() == "A") found_a = true;
        if(t->name() == "B") found_b = true;
    }
    assert(found_a && found_b);
    assert(leafs.size() == 2);
    
    const auto& nodes = g.nodes();
    assert(nodes.size() == 1);
    assert(nodes[0]->name() == "C");
    
    g.dump();
    std::cout << "[OK] test_simple_graph passed\n";
}

void test_diamond_graph() {
    std::cout << "Running test_diamond_graph...\n";

    // A
    // | \
    // B  C
    // | /
    // D
    
    Tensor a = Tensor::zeros({1}, DType::F32); a.name("A");
    
    Tensor b = relu(&a); b.name("B");
    Tensor c = relu(&a); c.name("C");
    
    Tensor d = add(&b, &c); d.name("D");
    
    Graph g = Graph::build_forward(&d);
    
    const auto& leafs = g.leafs();
    assert(leafs.size() == 1);
    assert(leafs[0]->name() == "A");
    
    const auto& nodes = g.nodes();
    assert(nodes.size() == 3); 
    
    auto get_index = [&](const std::string& name) -> int {
        for(size_t i=0; i<nodes.size(); ++i) {
            if (nodes[i]->name() == name) return (int)i;
        }
        return -1;
    };
    
    int idx_b = get_index("B");
    int idx_c = get_index("C");
    int idx_d = get_index("D");
    
    assert(idx_b != -1);
    assert(idx_c != -1);
    assert(idx_d != -1);
    
    assert(idx_b < idx_d);
    assert(idx_c < idx_d);
    
    g.dump();
    std::cout << "[OK] test_diamond_graph passed\n";
}

void test_graph_compute() {
    std::cout << "Running test_graph_compute...\n";
    ttsinfer::backend::Backend backend = ttsinfer::backend::make_cpu_backend();
    
    Tensor a = Tensor::ones({1}, DType::F32); a.name("A");
    Tensor b = Tensor::ones({1}, DType::F32); b.name("B");
    
   Tensor c = add(&a, &b); c.name("C");
    
    Graph g = Graph::build_forward(&c, backend);

    g.dump();
    g.compute_forward();
    
    float* c_data = c.data<float>();
    assert(c_data[0] == 2.0f);
    
    std::cout << "[OK] test_graph_compute passed\n";
}

void test_transformer_block() {
    std::cout << "Running test_transformer_block...\n";

    using namespace ttsinfer;
    backend::Backend backend = backend::make_cpu_backend();

    int seq = 1;
    int dim = 2;
    int ff_dim = 2;

    Tensor X = Tensor::ones({seq, dim}, DType::F32); 
    X.name("X");

    Tensor Wq = Tensor::eye({dim, dim}, DType::F32); Wq.name("Wq");
    Tensor Wk = Tensor::eye({dim, dim}, DType::F32); Wk.name("Wk");
    Tensor Wv = Tensor::eye({dim, dim}, DType::F32); Wv.name("Wv");
    Tensor W1 = Tensor::eye({dim, ff_dim}, DType::F32); W1.name("W1");
    Tensor W2 = Tensor::eye({ff_dim, dim}, DType::F32); W2.name("W2");


    Tensor Q = matmul(&X, &Wq); Q.name("Q");
    Tensor K = matmul(&X, &Wk); K.name("K");
    Tensor V = matmul(&X, &Wv); V.name("V");

    Tensor Kt = transpose(&K); Kt.name("Kt");

    Tensor scores = matmul(&Q, &Kt); scores.name("scores");
    Tensor attn   = softmax(&scores); attn.name("attn");

    Tensor context = matmul(&attn, &V); context.name("context");

    Tensor res1 = add(&context, &X); res1.name("res1");

    Tensor ff1 = matmul(&res1, &W1); ff1.name("ff1");
    Tensor ff2 = relu(&ff1); ff2.name("ff2");
    Tensor ff3 = matmul(&ff2, &W2); ff3.name("ff3");

    Tensor out = add(&ff3, &res1);
    out.name("OUT");

    Graph g = Graph::build_forward(&out, backend);
    g.dump();
    g.compute_forward();

    float* data = out.data<float>();

    assert(std::abs(data[0] - 4.0f) < 1e-5);
    assert(std::abs(data[1] - 4.0f) < 1e-5);

    std::cout << "Output: [" 
              << data[0] << ", " 
              << data[1] << "]\n";

    std::cout << "[OK] test_transformer_block passed\n";
}


int main() {
    try {
        test_simple_graph();
        test_diamond_graph();
        test_graph_compute();
        test_transformer_block();
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
