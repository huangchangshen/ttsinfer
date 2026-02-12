#pragma once
#include "backend.h"
#include <algorithm>
#include <cmath>

namespace ttsinfer::backend {

inline void cpu_matmul(Tensor* a, Tensor* b, Tensor* out) {
    auto* A = a->data<float>();
    auto* B = b->data<float>();
    auto* C = out->data<float>();
    int M = a->shape(0);
    int K = a->shape(1);
    int N = b->shape(1);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.f;
            for (int k = 0; k < K; ++k)
                sum += A[i*K + k] * B[k*N + j];
            C[i*N + j] = sum;
        }
    }
}

inline void cpu_add(Tensor* a, Tensor* b, Tensor* out) {
    int size = out->numel();
    auto* A = a->data<float>();
    auto* B = b->data<float>();
    auto* C = out->data<float>();
    for (int i = 0; i < size; ++i) C[i] = A[i] + B[i];
}

inline void cpu_relu(Tensor* a, Tensor* out) {
    int size = out->numel();
    auto* A = a->data<float>();
    auto* C = out->data<float>();
    for (int i = 0; i < size; ++i) C[i] = std::max(0.f, A[i]);
}

inline void cpu_transpose(Tensor* in, Tensor* out) {
    int rows = in->shape(0);
    int cols = in->shape(1);

    float* src = in->data<float>();
    float* dst = out->data<float>();

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            dst[j*rows + i] = src[i*cols + j];
}

void cpu_softmax(Tensor* in, Tensor* out) {
    int rows = in->shape(0);
    int cols = in->shape(1);

    float* src = in->data<float>();
    float* dst = out->data<float>();

    for (int i = 0; i < rows; ++i) {
        float maxv = -1e9f;
        for (int j = 0; j < cols; ++j)
            maxv = std::max(maxv, src[i*cols + j]);

        float sum = 0.f;
        for (int j = 0; j < cols; ++j) {
            dst[i*cols + j] = std::exp(src[i*cols + j] - maxv);
            sum += dst[i*cols + j];
        }

        for (int j = 0; j < cols; ++j)
            dst[i*cols + j] /= sum;
    }
}

inline Backend make_cpu_backend() {
    Backend b;
    b.matmul = cpu_matmul;
    b.add    = cpu_add;
    b.relu   = cpu_relu;
    b.transpose = cpu_transpose;
    b.softmax = cpu_softmax;
    return b;
}

} // namespace ttsinfer::backend
