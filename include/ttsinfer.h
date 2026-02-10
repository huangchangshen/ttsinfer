#pragma once

#include <cstddef>
#include <array>
#include <cassert>
#include <cstring>
#include <initializer_list>
#include <string>
#include <atomic>
#include <cstdarg>
#include <cstdio>

namespace ttsinfer {

inline constexpr std::size_t kMaxDims   = 4;
inline constexpr std::size_t kMaxInputs = 4;

enum class OpType { NONE=0, MATMUL, ADD, RELU, Count };
enum class DType   { F16, F32, BF16, I32, I64, U8, BOOL };
enum class MemType { OWNED, VIEW, EXTERNAL };

class Tensor {
public:
    friend Tensor matmul(Tensor* a, Tensor* b);

    Tensor();
    Tensor(OpType type, std::initializer_list<Tensor*> ins);
    ~Tensor();

    Tensor& name(const char* fmt, ...);

    const std::string& name() const { return name_; }
    OpType op() const { return op_; }
    std::size_t ndim() const { return ndim_; }
    const int* shape() const { return shape_; }
    const std::size_t* stride() const { return stride_; }
    void* data() const { return data_; }
    DType dtype() const { return dtype_; }
    MemType mem_type() const { return mem_type_; }
    std::size_t num_inputs() const { return num_inputs_; }
    Tensor* input(std::size_t idx) const { 
        assert(idx < num_inputs_); 
        return inputs_[idx]; 
    }
    Tensor* view_src() const { return view_src_; }



    static Tensor empty(const std::array<int, kMaxDims>& shp, DType dt = DType::F32);
    static Tensor zeros(const std::array<int, kMaxDims>& shp, DType dt = DType::F32);
    static Tensor ones(const std::array<int, kMaxDims>& shp, DType dt = DType::F32);
    static Tensor from_data(void* raw_data, const std::array<int, kMaxDims>& shp,
                            DType dt = DType::F32, MemType mt = MemType::EXTERNAL);

private:
    std::string name_;
    OpType op_ = OpType::NONE;
    std::size_t ndim_ = 0;
    int shape_[kMaxDims]{};
    std::size_t stride_[kMaxDims]{};
    void* data_ = nullptr;
    DType dtype_ = DType::F32;
    MemType mem_type_ = MemType::OWNED;
    Tensor* inputs_[kMaxInputs]{nullptr};
    std::size_t num_inputs_ = 0;
    Tensor* view_src_ = nullptr;

    static inline std::atomic<size_t> s_tensor_counter_{0};

    void assign_name_if_empty() {
        if (name_.empty()) {
            name_ = "tensor_" + std::to_string(s_tensor_counter_.fetch_add(1));
        }
    }
};
inline bool is_leaf(const Tensor& t) {
    return t.op() == OpType::NONE;
}

inline Tensor add(Tensor* a, Tensor* b) {
    return Tensor(OpType::ADD, {a, b});
}

inline Tensor matmul(Tensor* a, Tensor* b) {
    Tensor t(OpType::MATMUL, {a, b});
    assert(a->ndim() >= 2 && b->ndim() >= 2);
    t.ndim_ = 2;
    t.shape_[0] = a->shape()[0];
    t.shape_[1] = b->shape()[1];
    t.stride_[0] = t.shape_[1];
    t.stride_[1] = 1;
    return t;
}

inline Tensor relu(Tensor* a) {
    return Tensor(OpType::RELU, {a});
}

inline Tensor::Tensor() {
    assign_name_if_empty();
}

inline Tensor::Tensor(OpType type, std::initializer_list<Tensor*> ins)
    : op_(type), ndim_(0), num_inputs_(0) 
{
    assign_name_if_empty();
    assert(ins.size() <= kMaxInputs);
    for (Tensor* t : ins)
        inputs_[num_inputs_++] = t;

    if (num_inputs_ > 0) {
        ndim_ = inputs_[0]->ndim();
        for (std::size_t i = 0; i < ndim_; ++i) {
            shape_[i] = inputs_[0]->shape()[i];
            stride_[i] = inputs_[0]->stride()[i];
        }
    }
}

inline Tensor::~Tensor() {
    if (!data_) return;
    if (mem_type_ == MemType::OWNED) {
        switch (dtype_) {
            case DType::F32: delete[] static_cast<float*>(data_); break;
            case DType::I32: delete[] static_cast<int*>(data_); break;
            default: break;
        }
    }
}

inline Tensor& Tensor::name(const char* fmt, ...) {
    char buffer[128];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    name_ = buffer;
    return *this;
}

} // namespace ttsinfer
