#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace ttsinfer {
namespace safetensors {

enum class DType {
    F16,
    F32,
    BF16,
    I32,
    I64,
    U8,
    BOOL,
    UNKNOWN
};

size_t dtypeSize(DType t);

class TensorView {
public:
    const std::string& name() const { return name_; }
    DType dtype() const { return dtype_; }
    const std::vector<int64_t>& shape() const { return shape_; }

    const void* data() const { return data_; }
    size_t nbytes() const { return nbytes_; }

private:
    friend class SafeOpen;

    std::string name_;
    DType dtype_;
    std::vector<int64_t> shape_;
    const void* data_ = nullptr;
    size_t nbytes_ = 0;
};

class SafeOpen {
public:
    explicit SafeOpen(const std::string& path);
    ~SafeOpen();

    bool contains(const std::string& tensorName) const;
    TensorView getTensor(const std::string& tensorName) const;

    std::vector<std::string> keys() const;

private:
    SafeOpen(const SafeOpen&) = delete;
    SafeOpen& operator=(const SafeOpen&) = delete;

private:
    struct TensorMeta {
        DType dtype;
        std::vector<int64_t> shape;
        uint64_t offsetBegin;
        uint64_t offsetEnd;
    };

    int fd_ = -1;
    size_t fileSize_ = 0;
    const uint8_t* base_ = nullptr;
    size_t dataBase_ = 0;

    std::unordered_map<std::string, TensorMeta> metas_;
};

} // namespace safetensors
} // namespace ttsinfer
