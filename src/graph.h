#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace ttsinfer {

class Graph {
public:
    const std::vector<Tensor*>& nodes() const { return nodes_; }
    const std::vector<Tensor*>& leafs() const { return leafs_; }

    static Graph build_forward(Tensor* out);

    void dump() const;

private:
    std::vector<Tensor*> nodes_;
    std::vector<Tensor*> leafs_;
};

} // namespace ttsinfer
