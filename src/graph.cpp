#include "graph.h"
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <memory>

namespace ttsinfer {

static const char* op_name(OpType op) {
    switch (op) {
    case OpType::NONE:   return "NONE";
    case OpType::MATMUL: return "MATMUL";
    case OpType::ADD:    return "ADD";
    case OpType::RELU:   return "RELU";
    default:             return "UNKNOWN";
    }
}

static void print_shape(const Tensor* t) {
    std::cout << "[";
    for (std::size_t i = 0; i < t->ndim(); ++i) {
        std::cout << t->shape()[i];
        if (i + 1 < t->ndim()) std::cout << ", ";
    }
    std::cout << "]";
}

Graph::Graph() = default;

Graph::Graph(const backend::Backend& backend)
    : backend_(std::make_unique<backend::Backend>(backend)) {}

void Graph::compute_forward() {
    if (!backend_) {
        throw std::runtime_error("backend is not set");
    }

    for (Tensor* t : nodes_) {
        if (is_leaf(*t)) continue;
        if (!t->data()) {
            t->allocate();
        }
        backend_->dispatch(t);
    }
}

void Graph::dump() const {
    std::cout << "Graph dump\n==========\n\n";

    std::cout << "[Leafs]\n";
    for (const Tensor* t : leafs_) {
        std::cout << "  " << t->name() << "  shape=";
        print_shape(t);
        std::cout << "\n";
    }

    std::cout << "\n[Nodes] (topological order)\n";
    for (std::size_t i = 0; i < nodes_.size(); ++i) {
        const Tensor* t = nodes_[i];
        std::cout << "  " << i << ": "
                  << t->name() << " = "
                  << op_name(t->op()) << "(";

        for (std::size_t j = 0; j < t->num_inputs(); ++j) {
            std::cout << t->input(j)->name();
            if (j + 1 < t->num_inputs()) std::cout << ", ";
        }
        std::cout << ") shape=";
        print_shape(t);
        std::cout << "\n";
    }

    std::cout << std::endl;
}

void Graph::visit_dfs(
    Tensor* t,
    std::unordered_set<Tensor*>& visiting,
    std::unordered_set<Tensor*>& visited,
    std::vector<Tensor*>& nodes,
    std::vector<Tensor*>& leafs,
    std::vector<Tensor*>& stack
) {
    if (visited.count(t)) return;

    if (visiting.count(t)) {
        std::string msg = "cycle detected: ";
        auto it = std::find(stack.begin(), stack.end(), t);
        for (; it != stack.end(); ++it) {
            msg += (*it)->name() + " -> ";
        }
        msg += t->name();
        throw std::runtime_error(msg);
    }

    visiting.insert(t);
    stack.push_back(t);

    if (is_leaf(*t)) {
        leafs.push_back(t);
    } else {
        for (std::size_t i = 0; i < t->num_inputs(); ++i) {
            visit_dfs(t->input(i), visiting, visited, nodes, leafs, stack);
        }
        nodes.push_back(t);
    }

    stack.pop_back();
    visiting.erase(t);
    visited.insert(t);
}

Graph Graph::build_forward(Tensor* out) {
    Graph g;

    std::unordered_set<Tensor*> visiting;
    std::unordered_set<Tensor*> visited;
    std::vector<Tensor*> stack;

    g.nodes_.reserve(64);
    g.leafs_.reserve(32);
    stack.reserve(16);

    visit_dfs(out, visiting, visited, g.nodes_, g.leafs_, stack);

    return g;
}

Graph Graph::build_forward(Tensor* out, const backend::Backend& backend) {
    Graph g(backend);

    std::unordered_set<Tensor*> visiting;
    std::unordered_set<Tensor*> visited;
    std::vector<Tensor*> stack;

    g.nodes_.reserve(64);
    g.leafs_.reserve(32);
    stack.reserve(16);

    visit_dfs(out, visiting, visited, g.nodes_, g.leafs_, stack);

    return g;
}

} // namespace ttsinfer
