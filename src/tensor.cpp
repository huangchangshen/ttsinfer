#include "ttsinfer.h"
#include <cstring>
#include <stdexcept>
#include <utility>

namespace ttsinfer {

static std::size_t element_size(DType dt) {
    switch (dt) {
        case DType::F16:  return 2;
        case DType::F32:  return 4;
        case DType::BF16: return 2;
        case DType::I32:  return 4;
        case DType::I64:  return 8;
        case DType::U8:   return 1;
        case DType::BOOL: return 1;
        default: return 0;
    }
}

Tensor::Tensor(Tensor&& other) noexcept 
    : name_(std::move(other.name_)), op_(other.op_), ndim_(other.ndim_),
      data_(other.data_), dtype_(other.dtype_), mem_type_(other.mem_type_),
      num_inputs_(other.num_inputs_), view_src_(other.view_src_)
{
    std::memcpy(shape_, other.shape_, sizeof(shape_));
    std::memcpy(stride_, other.stride_, sizeof(stride_));
    std::memcpy(inputs_, other.inputs_, sizeof(inputs_));
    

    other.data_ = nullptr;
    other.ndim_ = 0;
    other.num_inputs_ = 0;
    other.mem_type_ = MemType::EXTERNAL; 
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {

        if (data_ && mem_type_ == MemType::OWNED) {
            switch (dtype_) {
                case DType::F32: delete[] static_cast<float*>(data_); break;
                case DType::I32: delete[] static_cast<int*>(data_); break;
                default: break; 
            }
        }
        
        name_ = std::move(other.name_);
        op_ = other.op_;
        ndim_ = other.ndim_;
        std::memcpy(shape_, other.shape_, sizeof(shape_));
        std::memcpy(stride_, other.stride_, sizeof(stride_));
        data_ = other.data_;
        dtype_ = other.dtype_;
        mem_type_ = other.mem_type_;
        num_inputs_ = other.num_inputs_;
        std::memcpy(inputs_, other.inputs_, sizeof(inputs_));
        view_src_ = other.view_src_;
        
        other.data_ = nullptr;
        other.ndim_ = 0;
        other.num_inputs_ = 0;
        other.mem_type_ = MemType::EXTERNAL;
    }
    return *this;
}

void Tensor::allocate() {
    if (data_) return;

    if (mem_type_ != MemType::OWNED) return;
    
    size_t num_elements = 1;
    for(size_t i=0; i<ndim_; ++i) num_elements *= shape_[i];

    size_t sz = num_elements * element_size(dtype_);
    if (sz > 0) {
        switch (dtype_) {
            case DType::F32: data_ = new float[num_elements]; break;
            case DType::I32: data_ = new int[num_elements]; break;
            default: break;
        }
    }
}

Tensor Tensor::empty(const std::array<int, kMaxDims>& shp, DType dt) {
    Tensor t;
    t.dtype_ = dt;
    t.mem_type_ = MemType::OWNED;
    
    t.ndim_ = 0;
    std::size_t num_elements = 1;
    for (std::size_t i = 0; i < kMaxDims; ++i) {
        if (shp[i] <= 0) break;
        t.shape_[i] = shp[i];
        t.ndim_++;
        num_elements *= shp[i];
    }
    
    if (t.ndim_ > 0) {
        t.stride_[t.ndim_ - 1] = 1;
        for (int i = static_cast<int>(t.ndim_) - 2; i >= 0; --i) {
            t.stride_[i] = t.stride_[i + 1] * t.shape_[i + 1];
        }
    }

    size_t sz = num_elements * element_size(dt);
    if (sz > 0) {
        switch (dt) {
            case DType::F32: t.data_ = new float[num_elements]; break;
            case DType::I32: t.data_ = new int[num_elements]; break;
            default: break;
        }
    }
    
    return t;
}

Tensor Tensor::zeros(const std::array<int, kMaxDims>& shp, DType dt) {
    Tensor t = empty(shp, dt);
    std::size_t num_elements = 1;
    for(size_t i=0; i<t.ndim_; ++i) num_elements *= t.shape_[i];
    
    size_t sz = num_elements * element_size(dt);
    if (t.data_) {
        std::memset(t.data_, 0, sz);
    }
    return t;
}


Tensor Tensor::ones(const std::array<int, kMaxDims>& shp, DType dt) {
    Tensor t = empty(shp, dt);
    std::size_t num_elements = 1;
    for(size_t i=0; i<t.ndim_; ++i) num_elements *= t.shape_[i];

    if (dt == DType::F32) {
        float* p = static_cast<float*>(t.data_);
        for(size_t i=0; i<num_elements; ++i) p[i] = 1.0f;
    } else if (dt == DType::I32) {
        int* p = static_cast<int*>(t.data_);
        for(size_t i=0; i<num_elements; ++i) p[i] = 1;
    }
    return t;
}

Tensor Tensor::eye(const std::array<int, kMaxDims>& shp, DType dt) {
    if (shp[0] <= 0 || shp[1] <= 0) {
        throw std::runtime_error("eye requires at least 2D shape");
    }

    Tensor t = empty(shp, dt);
    std::size_t rows = shp[0];
    std::size_t cols = shp[1];

    if (dt == DType::F32) {
        float* p = static_cast<float*>(t.data_);
        std::fill(p, p + rows*cols, 0.f);
        std::size_t diag = std::min(rows, cols);
        for (std::size_t i = 0; i < diag; ++i) {
            p[i*cols + i] = 1.f;
        }
    } else if (dt == DType::I32) {
        int* p = static_cast<int*>(t.data_);
        std::fill(p, p + rows*cols, 0);
        std::size_t diag = std::min(rows, cols);
        for (std::size_t i = 0; i < diag; ++i) {
            p[i*cols + i] = 1;
        }
    } else {
        throw std::runtime_error("unsupported dtype for eye()");
    }

    return t;
}

Tensor Tensor::from_data(void* raw_data, const std::array<int, kMaxDims>& shp,
                        DType dt, MemType mt) {
    Tensor t;
    t.dtype_ = dt;
    t.mem_type_ = mt;
    t.data_ = raw_data;
    
    t.ndim_ = 0;
    for (std::size_t i = 0; i < kMaxDims; ++i) {
        if (shp[i] <= 0) break;
        t.shape_[i] = shp[i];
        t.ndim_++;
    }
     if (t.ndim_ > 0) {
        t.stride_[t.ndim_ - 1] = 1;
        for (int i = static_cast<int>(t.ndim_) - 2; i >= 0; --i) {
            t.stride_[i] = t.stride_[i + 1] * t.shape_[i + 1];
        }
    }
    
    return t;
}

} // namespace ttsinfer
