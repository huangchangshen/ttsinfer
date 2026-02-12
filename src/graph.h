#pragma once

#include <vector>
#include <unordered_set>
#include "ttsinfer.h"
#include "backend.h"

namespace ttsinfer {

class Graph {
public:
    Graph();
    explicit Graph(const backend::Backend& backend);

    static Graph build_forward(Tensor* out);
    static Graph build_forward(Tensor* out, const backend::Backend& backend);

    void compute_forward();
    void dump() const;

    const std::vector<Tensor*>& nodes() const { return nodes_; }
    const std::vector<Tensor*>& leafs() const { return leafs_; }

private:
    std::vector<Tensor*> nodes_;
    std::vector<Tensor*> leafs_;
    std::unique_ptr<backend::Backend> backend_;

    static void visit_dfs(
        Tensor* t,
        std::unordered_set<Tensor*>& visiting,
        std::unordered_set<Tensor*>& visited,
        std::vector<Tensor*>& nodes,
        std::vector<Tensor*>& leafs,
        std::vector<Tensor*>& stack
    );
};

} // namespace ttsinfer
