#include "graph.h"
#include <algorithm>
#include <cmath>

namespace ttsinfer {

void Graph::compute_node(Tensor& t) {
    TensorImpl* impl = t.unsafe_get_impl();
    if (!impl) return;

    OpType op = impl->op_;
    auto& inputs = impl->inputs_;
    
    float* out_ptr = t.data();
    size_t n = 1; 
    for(int s : t.shape()) n *= s;

    switch (op) {
        case OpType::ADD: {
            if (inputs.size() < 2) return;
            Tensor a = inputs[0];
            Tensor b = inputs[1];
            float* pa = a.data();
            float* pb = b.data();
            for (size_t i = 0; i < n; ++i) {
                out_ptr[i] = pa[i] + pb[i];
            }
            break;
        }
        case OpType::MATMUL: {
            if (inputs.size() < 2) return;
            Tensor a = inputs[0];
            Tensor b = inputs[1];
            float* pa = a.data();
            float* pb = b.data();
            
            if (a.shape().size() < 2 || b.shape().size() < 2) return;

            int M = a.shape()[0];
            int K = a.shape()[1];
            int N = b.shape()[1]; 
            
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    float sum = 0.f;
                    for (int k = 0; k < K; ++k) {
                        sum += pa[i * K + k] * pb[k * N + j];
                    }
                    out_ptr[i * N + j] = sum;
                }
            }
            break;
        }
        case OpType::RELU: {
            if (inputs.empty()) return;
            Tensor a = inputs[0];
            float* pa = a.data();
            for (size_t i = 0; i < n; ++i) {
                out_ptr[i] = std::max(0.f, pa[i]);
            }
            break;
        }
        case OpType::TRANSPOSE: {
            if (inputs.empty()) return;
            Tensor a = inputs[0];
            float* pa = a.data();
            int M = a.shape()[0];
            int N = a.shape()[1];
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    out_ptr[j * M + i] = pa[i * N + j];
                }
            }
            break;
        }
        case OpType::SOFTMAX: {
            if (inputs.empty()) return;
            Tensor a = inputs[0];
            float* pa = a.data();
            int rows = a.shape()[0];
            int cols = a.shape()[1];
            
            for (int i = 0; i < rows; ++i) {
                float max_val = -1e9;
                for (int j = 0; j < cols; ++j) {
                    max_val = std::max(max_val, pa[i * cols + j]);
                }
                
                float sum = 0.f;
                for (int j = 0; j < cols; ++j) {
                    float val = std::exp(pa[i * cols + j] - max_val);
                    out_ptr[i * cols + j] = val;
                    sum += val;
                }
                
                for (int j = 0; j < cols; ++j) {
                    out_ptr[i * cols + j] /= sum;
                }
            }
            break;
        }
        default:
            break;
    }
}

void Graph::compute_forward() {
    for (Tensor& t : nodes_) {
        compute_node(t);
    }
}

void Graph::dump() const {
    std::cout << "Graph dump:\n";
    std::cout << "  Leafs  : " << leafs_.size() << "\n";
    std::cout << "  Nodes  : " << nodes_.size() << "\n\n";

    std::cout << "=== Leafs ===\n";
    for (size_t i = 0; i < leafs_.size(); ++i) {
        std::cout << "Leaf " << i << ": ";
        const_cast<Tensor&>(leafs_[i]).dump();
    }

    std::cout << "\n=== Nodes ===\n";
    for (size_t i = 0; i < nodes_.size(); ++i) {
        std::cout << "Node " << i << ": ";
        const_cast<Tensor&>(nodes_[i]).dump();
    }
}

} // namespace ttsinfer