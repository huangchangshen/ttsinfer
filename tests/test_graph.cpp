#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>
#include "ttsinfer.h"
#include "graph.h"

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

int main() {
    try {
        test_simple_graph();
        test_diamond_graph();
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
