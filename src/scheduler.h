#pragma once
#include "ttsinfer.h"
#include <unordered_set>
#include <vector>
#include <iostream>
#include <functional>

namespace ttsinfer {

constexpr int kMaxBackends = 8;
constexpr int kMaxCopies = 4;

struct Device {
    int id;

}

struct BackendImpl {
    bool (*support_op)(const Tensor& t);
    bool (*support_storage)(const Tensor& t);
    void (*graph_compute)(const Graph& graph);
}


struct Backend {
    BackendImpl impl;
};

class Scheduler {

public:
    std::array<Backend*, kMaxBackends> backends;

    Graph graph;
    std::vector<int> node_backend_ids;
    std::vector<int> leaf_backend_ids;

    int backend_from_storage(const Tensor& t, const Tensor& op) {
        for (int i = 0; i < backends.size(); ++i) {
            if (backends[i]->impl.support_storage(t) 
                && backends[i]->impl.support_op(op)) {
                return i;
            }
        }
        return -1;
    }

    void graph_compute() {
        graph.build_forward();
        backends[i]->impl.graph_compute(graph);
    }

};

}