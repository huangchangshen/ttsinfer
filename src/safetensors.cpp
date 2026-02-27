#include "safetensors.h"
#include "json.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <stdexcept>
#include <cstring>

namespace ttsinfer {
namespace safetensors {

namespace {

uint64_t readU64LE(const uint8_t* p) {
    uint64_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

DType parseDType(const std::string& s) {
    if (s == "F16")  return DType::F16;
    if (s == "F32")  return DType::F32;
    if (s == "BF16") return DType::BF16;
    if (s == "I32")  return DType::I32;
    if (s == "I64")  return DType::I64;
    if (s == "U8")   return DType::U8;
    if (s == "BOOL") return DType::BOOL;
    return DType::UNKNOWN;
}

} // anonymous namespace

size_t dtypeSize(DType t) {
    switch (t) {
        case DType::F16:  return 2;
        case DType::BF16: return 2;
        case DType::F32:  return 4;
        case DType::I32:  return 4;
        case DType::I64:  return 8;
        case DType::U8:   return 1;
        case DType::BOOL: return 1;
        default: return 0;
    }
}

SafeOpen::SafeOpen(const std::string& path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw std::runtime_error("safe_open: failed to open " + path);
    }

    struct stat st {};
    if (::fstat(fd_, &st) != 0) {
        ::close(fd_);
        throw std::runtime_error("safe_open: fstat failed");
    }
    fileSize_ = static_cast<size_t>(st.st_size);

    base_ = static_cast<const uint8_t*>(
        ::mmap(nullptr, fileSize_, PROT_READ, MAP_PRIVATE, fd_, 0)
    );

    if (base_ == MAP_FAILED) {
        ::close(fd_);
        throw std::runtime_error("safe_open: mmap failed");
    }

    uint64_t headerLen = readU64LE(base_);
    size_t headerStart = 8;
    size_t headerEnd   = headerStart + headerLen;

    if (headerEnd > fileSize_) {
        throw std::runtime_error("safe_open: invalid header length");
    }

    std::string headerJson(
        reinterpret_cast<const char*>(base_ + headerStart),
        headerLen
    );

    JsonValue root = JsonParser::parse(headerJson);
    const JsonObject& obj = root.as<JsonObject>();

    dataBase_ = headerEnd;

    for (const auto& kv : obj) {
        const std::string& name = kv.first;
        const JsonObject& meta  = kv.second.as<JsonObject>();

        TensorMeta tm;
        tm.dtype = parseDType(meta.at("dtype").as<std::string>());

        const JsonArray& shape = meta.at("shape").as<JsonArray>();
        for (const auto& v : shape) {
            tm.shape.push_back(v.as<int64_t>());
        }

        const JsonArray& offsets = meta.at("data_offsets").as<JsonArray>();
        tm.offsetBegin = offsets[0].as<int64_t>();
        tm.offsetEnd   = offsets[1].as<int64_t>();

        metas_.emplace(name, std::move(tm));
    }
}

SafeOpen::~SafeOpen() {
    if (base_) {
        ::munmap(const_cast<uint8_t*>(base_), fileSize_);
    }
    if (fd_ >= 0) {
        ::close(fd_);
    }
}

bool SafeOpen::contains(const std::string& tensorName) const {
    return metas_.find(tensorName) != metas_.end();
}

TensorView SafeOpen::get_tensor(const std::string& tensorName) const {
    auto it = metas_.find(tensorName);
    if (it == metas_.end()) {
        throw std::runtime_error("safe_open: tensor not found: " + tensorName);
    }

    const TensorMeta& m = it->second;

    TensorView tv;
    tv.name_   = tensorName;
    tv.dtype_  = m.dtype;
    tv.shape_  = m.shape;
    tv.data_   = base_ + dataBase_ + m.offsetBegin;
    tv.nbytes_ = static_cast<size_t>(m.offsetEnd - m.offsetBegin);

    return tv;
}

std::vector<std::string> SafeOpen::keys() const {
    std::vector<std::string> out;
    out.reserve(metas_.size());
    for (const auto& kv : metas_) {
        out.push_back(kv.first);
    }
    return out;
}

} // namespace safetensors
} // namespace ttsinfer
