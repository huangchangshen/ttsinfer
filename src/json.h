#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <variant>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace ttsinfer {

class JsonValue;

using JsonArray = std::vector<JsonValue>;
using JsonObject = std::unordered_map<std::string, JsonValue>;

class JsonValue {
public:
    using ValueType = std::variant<
        std::nullptr_t,
        bool,
        int64_t,
        double,
        std::string,
        JsonArray,
        JsonObject
    >;

private:
    ValueType value_;

public:
    JsonValue() : value_(nullptr) {}
    JsonValue(std::nullptr_t) : value_(nullptr) {}
    JsonValue(bool b) : value_(b) {}
    JsonValue(int i) : value_(static_cast<int64_t>(i)) {}
    JsonValue(int64_t i) : value_(i) {}
    JsonValue(double d) : value_(d) {}
    JsonValue(const std::string& s) : value_(s) {}
    JsonValue(std::string&& s) : value_(std::move(s)) {}
    JsonValue(const char* s) : value_(std::string(s)) {}
    JsonValue(const JsonArray& a) : value_(a) {}
    JsonValue(JsonArray&& a) : value_(std::move(a)) {}
    JsonValue(const JsonObject& o) : value_(o) {}
    JsonValue(JsonObject&& o) : value_(std::move(o)) {}

    bool isNull() const { return std::holds_alternative<std::nullptr_t>(value_); }
    bool isBool() const { return std::holds_alternative<bool>(value_); }
    bool isInt() const { return std::holds_alternative<int64_t>(value_); }
    bool isDouble() const { return std::holds_alternative<double>(value_); }
    bool isString() const { return std::holds_alternative<std::string>(value_); }
    bool isArray() const { return std::holds_alternative<JsonArray>(value_); }
    bool isObject() const { return std::holds_alternative<JsonObject>(value_); }

    template<typename T>
    T as() const {
        if constexpr (std::is_same_v<T, int>) {
            if (isInt()) return static_cast<int>(std::get<int64_t>(value_));
            if (isDouble()) return static_cast<int>(std::get<double>(value_));
        } else if constexpr (std::is_same_v<T, int64_t>) {
            if (isInt()) return std::get<int64_t>(value_);
            if (isDouble()) return static_cast<int64_t>(std::get<double>(value_));
        } else if constexpr (std::is_same_v<T, double>) {
            if (isDouble()) return std::get<double>(value_);
            if (isInt()) return static_cast<double>(std::get<int64_t>(value_));
        } else {
            return std::get<T>(value_);
        }
        throw std::runtime_error("JSON type mismatch");
    }

    JsonArray& asArray() {
        if (!isArray()) throw std::runtime_error("Not a JSON array");
        return std::get<JsonArray>(value_);
    }

    const JsonArray& asArray() const {
        if (!isArray()) throw std::runtime_error("Not a JSON array");
        return std::get<JsonArray>(value_);
    }

    JsonObject& asObject() {
        if (!isObject()) throw std::runtime_error("Not a JSON object");
        return std::get<JsonObject>(value_);
    }

    const JsonObject& asObject() const {
        if (!isObject()) throw std::runtime_error("Not a JSON object");
        return std::get<JsonObject>(value_);
    }

    JsonValue& operator[](const std::string& key) {
        if (!isObject()) throw std::runtime_error("Not a JSON object");
        return std::get<JsonObject>(value_)[key];
    }

    JsonValue& operator[](size_t idx) {
        if (!isArray()) throw std::runtime_error("Not a JSON array");
        return std::get<JsonArray>(value_).at(idx);
    }

    size_t size() const {
        if (isArray()) return std::get<JsonArray>(value_).size();
        if (isObject()) return std::get<JsonObject>(value_).size();
        return 0;
    }
};

class JsonParser {
public:
    static JsonValue parse(const std::string& jsonStr);
    static JsonValue parseFile(const std::string& filePath);
};


} // namespace ttsinfer
