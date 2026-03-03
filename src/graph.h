#pragma once
#include "ttsinfer.h"
#include <unordered_set>
#include <vector>
#include <iostream>
#include <functional>

namespace ttsinfer {

class Graph {
public:
    std::vector<Tensor> leafs_;
    std::vector<Tensor> nodes_;

    void build_forward(const Tensor& out) {
        leafs_.clear();
        nodes_.clear();

        std::unordered_set<TensorImpl*> visited;

        std::function<void(const Tensor&)> dfs =
            [&](const Tensor& t) {

            TensorImpl* impl = t.unsafe_get_impl();
            if (!impl || visited.count(impl))
                return;

            visited.insert(impl);

            for (const Tensor& inp : impl->inputs_) {
                dfs(inp);
            }

            if (impl->inputs_.empty()) {
                leafs_.push_back(t);
            } else {
                nodes_.push_back(t);
            }
        };

        dfs(out);
    }

    void compute_forward();

    void dump() const;

private:
    static void compute_node(Tensor& t);
};

} // namespace ttsinfer
