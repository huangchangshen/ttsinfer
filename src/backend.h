#pragma once
#include "ttsinfer.h"
#include <stdexcept>

namespace ttsinfer::backend {

struct Backend {
    void (*matmul)(Tensor*, Tensor*, Tensor*);
    void (*add)(Tensor*, Tensor*, Tensor*);
    void (*relu)(Tensor*, Tensor*);
    void (*transpose)(Tensor*, Tensor*);
    void (*softmax)(Tensor*, Tensor*);

    void dispatch(Tensor* t) const {
        switch (t->op()) {
        case OpType::MATMUL:
            matmul(t->input(0), t->input(1), t);
            break;
        case OpType::ADD:
            add(t->input(0), t->input(1), t);
            break;
        case OpType::RELU:
            relu(t->input(0), t);
            break;
        case OpType::TRANSPOSE:
            transpose(t->input(0), t);
            break;
        case OpType::SOFTMAX:
            softmax(t->input(0), t);
            break;
        default:
            throw std::runtime_error("unsupported op");
        }
    }
};

} // namespace ttsinfer::backend
