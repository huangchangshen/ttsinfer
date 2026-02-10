#include "ttsinfer.h"
#include <cstring>
#include <stdexcept>

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

static void compute_strides(Tensor& t) {
    if (t.ndim() == 0) return;
   
    std::size_t* strides = const_cast<std::size_t*>(t.stride());
    const int* shape = t.shape();
    
    strides[t.ndim() - 1] = 1;
    if (t.ndim() > 1) {
        for (int i = static_cast<int>(t.ndim()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
}

static void init_shape(Tensor& t, const std::array<int, kMaxDims>& shp) {
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

            default: 
                if (dt == DType::I64) t.data_ = new long long[num_elements]; 
                t.data_ = new float[num_elements];
                break;
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
