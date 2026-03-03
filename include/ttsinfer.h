#pragma once
#include <atomic>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <iostream>

namespace ttsinfer {

inline constexpr std::size_t kMaxDims   = 4;
inline constexpr std::size_t kMaxInputs = 4;

enum class OpType {
    NONE = 0,
    MATMUL,
    ADD,
    RELU,
    TRANSPOSE,
    SOFTMAX,
    Count
};

enum class DType   { F16, F32, BF16, I32, I64, U8, BOOL, UNKNOWN };

class RefCounted {
public:
    mutable std::atomic<int> refcount_{0};

    void retain() const {
        refcount_.fetch_add(1, std::memory_order_relaxed);
    }

    void release() const {
        if(refcount_.fetch_sub(1, std::memory_order_acq_rel) == 1)
            delete this;
    }

protected:
    virtual ~RefCounted() = default;
};

template<typename T>
class IntrusivePtr {
    T* ptr_;

public:
    IntrusivePtr() : ptr_(nullptr) {}

    explicit IntrusivePtr(T* p) : ptr_(p) {
        if(ptr_) ptr_->retain();
    }

    IntrusivePtr(const IntrusivePtr& other) : ptr_(other.ptr_) {
        if(ptr_) ptr_->retain();
    }

    IntrusivePtr(IntrusivePtr&& other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }

    IntrusivePtr& operator=(const IntrusivePtr& other) {
        if (this != &other) {
            if (ptr_) ptr_->release();
            ptr_ = other.ptr_;
            if (ptr_) ptr_->retain();
        }
        return *this;
    }

    IntrusivePtr& operator=(IntrusivePtr&& other) noexcept {
        if (this != &other) {
            if (ptr_) ptr_->release();
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    ~IntrusivePtr() {
        if(ptr_) ptr_->release();
    }

    T* get() const { return ptr_; }
    T* operator->() const { return ptr_; }

    explicit operator bool() const noexcept {
        return ptr_ != nullptr;
    }
};


class Storage : public RefCounted {
public:
    void* data_;
    size_t size_;

    Storage(size_t size) : size_(size) {
        data_ = malloc(size);
    }

    ~Storage() {
        free(data_);
    }
};

class Tensor;

class TensorImpl : public RefCounted {
public:
    std::vector<int> shape_;
    std::vector<int> stride_;
    size_t numel_;
    DType dtype_;
    IntrusivePtr<Storage> storage_;

    OpType op_;
    std::vector<Tensor> inputs_;

    TensorImpl(std::vector<int> shape, DType dt)
        : shape_(std::move(shape)),    
          stride_(),                  
          dtype_(dt),                 
          storage_(nullptr),           
          op_(OpType::NONE),
          numel_(1),
          inputs_()                
    {
        for (int s : shape_) numel_ *= s;

        storage_ = IntrusivePtr<Storage>(new Storage(numel_ * sizeof(float)));
    }

    float* data() {
        return static_cast<float*>(storage_->data_);
    }

    size_t numel() const { return numel_; }

    void dump() const;

    ~TensorImpl() {
        std::cout << "[TensorImpl destroyed] @" << this << "\n";
    }
    
};

class Tensor {
    IntrusivePtr<TensorImpl> impl_;

public:
    Tensor() {}

    explicit Tensor(TensorImpl* impl)
        : impl_(IntrusivePtr<TensorImpl>(impl)) {}

    static Tensor make(OpType op, const std::vector<int>& shape, DType dt,
                    const std::vector<Tensor>& inputs)
    {
        Tensor t(new TensorImpl(shape, dt));
        t.impl_->op_ = op;
        t.impl_->inputs_ = inputs;
        return t;
    }

    static Tensor empty(std::vector<int> shape, DType dt) {
        return make(OpType::NONE, shape, dt, {});
    }

    static Tensor zeros(const std::vector<int>& shape, DType dt) {
        Tensor t = empty(shape, dt);
        size_t n = t.impl_->numel();
        std::memset(t.impl_->data(), 0, n * sizeof(float));
        return t;
    }

    static Tensor ones(const std::vector<int>& shape, DType dt) {
        Tensor t = empty(shape, dt);
        size_t n = t.impl_->numel();
        float* ptr = static_cast<float*>(t.impl_->data());
        for (size_t i = 0; i < n; ++i) ptr[i] = 1.f;
        return t;
    }

    static Tensor full(const std::vector<int>& shape, DType dt, float value) {
        Tensor t = empty(shape, dt);
        size_t n = t.impl_->numel();
        float* ptr = static_cast<float*>(t.impl_->data());
        for (size_t i = 0; i < n; ++i) ptr[i] = value;
        return t;
    }

    static Tensor rand_uniform(const std::vector<int>& shape, DType dt,
                               float min = 0.f, float max = 1.f)
    {
        Tensor t = empty(shape, dt);
        size_t n = t.impl_->numel();
        float* ptr = static_cast<float*>(t.impl_->data());
        for (size_t i = 0; i < n; ++i) {
            ptr[i] = min + static_cast<float>(rand()) / RAND_MAX * (max - min);
        }
        return t;
    }

    float* data() {
        return static_cast<float*>(impl_->data());
    }

    const std::vector<int>& shape() const {
        return impl_->shape_;
    }

    TensorImpl* unsafe_get_impl() { return impl_.get(); }
    TensorImpl* unsafe_get_impl() const { return impl_.get(); }

    void dump() const {
        if (!impl_) {
            std::cout << "Tensor: nullptr\n";
            return;
        }

        impl_->dump();
    }

};

inline void TensorImpl::dump() const {
    std::cout << "TensorImpl @" << this
                << " refcount=" << refcount_.load()
                << " dtype=" << static_cast<int>(dtype_)
                << " shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        std::cout << shape_[i];
        if (i + 1 < shape_.size()) std::cout << ", ";
    }
    std::cout << "]";
    if (op_ != OpType::NONE) {
        std::cout << " op=" << static_cast<int>(op_) << " inputs=[";
        for (size_t i = 0; i < inputs_.size(); ++i) {
            std::cout << inputs_[i].unsafe_get_impl();
            if (i + 1 < inputs_.size()) std::cout << ", ";
        }
        std::cout << "]";
    }
    std::cout << "\n";
}

inline Tensor matmul(const Tensor& A, const Tensor& B) {
    if (A.shape().size() != 2 || B.shape().size() != 2)
        throw std::runtime_error("matmul only supports 2D");

    int m = A.shape()[0];
    int k = A.shape()[1];
    int k2 = B.shape()[0];
    int n = B.shape()[1];

    if (k != k2)
        throw std::runtime_error("matmul dimension mismatch");

    return Tensor::make(OpType::MATMUL, {m, n}, DType::F32, {A, B});
}

inline Tensor add(const Tensor& A, const Tensor& B) {
    const auto& a_shape = A.shape();
    const auto& b_shape = B.shape();

    if (a_shape == b_shape) {
        return Tensor::make(OpType::ADD, a_shape, DType::F32, {A, B});
    }

    // 2D broadcast
    if (a_shape.size() == 2 &&
        b_shape.size() == 2 &&
        a_shape[1] == b_shape[1] &&
        b_shape[0] == 1)
    {
        return Tensor::make(OpType::ADD, a_shape, DType::F32, {A, B});
    }

    throw std::runtime_error("add requires same shape or broadcastable");
}

inline Tensor relu(const Tensor& X) {
    return Tensor::make(OpType::RELU, X.shape(), DType::F32, {X});
}

inline Tensor transpose(const Tensor& X) {
    if (X.shape().size() != 2)
        throw std::runtime_error("transpose only supports 2D");

    return Tensor::make(OpType::TRANSPOSE, {X.shape()[1], X.shape()[0]}, DType::F32, {X});
}

inline Tensor softmax(const Tensor& X) {
    return Tensor::make(OpType::SOFTMAX, X.shape(), DType::F32, {X});
}

}