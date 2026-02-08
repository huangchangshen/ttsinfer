#include "json.h"
#include <fstream>
#include <sstream>
#include <cctype>
#include <stdexcept>

namespace ttsinfer {

class JsonParserImpl {
public:
    explicit JsonParserImpl(const std::string& src)
        : src_(src), pos_(0) {}

    JsonValue parse() {
        skipWhitespace();
        JsonValue v = parseValue();
        skipWhitespace();
        if (!eof()) {
            error("unexpected trailing characters");
        }
        return v;
    }

private:
    bool eof() const { return pos_ >= src_.size(); }
    char peek() const { return eof() ? '\0' : src_[pos_]; }
    char get() { return eof() ? '\0' : src_[pos_++]; }

    [[noreturn]] void error(const std::string& msg) const {
        throw std::runtime_error(
            "Json parse error at pos " + std::to_string(pos_) + ": " + msg);
    }

    void skipWhitespace() {
        while (!eof() && std::isspace(static_cast<unsigned char>(peek()))) {
            ++pos_;
        }
    }

    JsonValue parseValue() {
        skipWhitespace();
        if (eof()) error("unexpected end of input");

        char c = peek();
        if (c == '{') return parseObject();
        if (c == '[') return parseArray();
        if (c == '"') return parseString();
        if (c == 't' || c == 'f') return parseBool();
        if (c == 'n') return parseNull();
        if (c == '-' || std::isdigit(c)) return parseNumber();

        error(std::string("unexpected character '") + c + "'");
        return {}; 
    }

    JsonValue parseObject() {
        expect('{');
        skipWhitespace();

        JsonObject obj;

        if (peek() == '}') {
            get();
            return JsonValue(std::move(obj));
        }

        while (true) {
            skipWhitespace();
            if (peek() != '"') error("expected string key");

            std::string key = parseString().as<std::string>();

            skipWhitespace();
            expect(':');
            skipWhitespace();

            obj.emplace(std::move(key), parseValue());

            skipWhitespace();
            char c = get();
            if (c == '}') break;
            if (c != ',') error("expected ',' or '}'");
        }

        return JsonValue(std::move(obj));
    }

    JsonValue parseArray() {
        expect('[');
        skipWhitespace();

        JsonArray arr;

        if (peek() == ']') {
            get();
            return JsonValue(std::move(arr));
        }

        while (true) {
            arr.emplace_back(parseValue());
            skipWhitespace();
            char c = get();
            if (c == ']') break;
            if (c != ',') error("expected ',' or ']'");
        }

        return JsonValue(std::move(arr));
    }

    JsonValue parseString() {
        expect('"');
        std::string s;

        while (!eof()) {
            char c = get();
            if (c == '"') break;

            if (c == '\\') {
                if (eof()) error("invalid escape");
                char esc = get();
                switch (esc) {
                    case '"': s.push_back('"'); break;
                    case '\\': s.push_back('\\'); break;
                    case '/': s.push_back('/'); break;
                    case 'b': s.push_back('\b'); break;
                    case 'f': s.push_back('\f'); break;
                    case 'n': s.push_back('\n'); break;
                    case 'r': s.push_back('\r'); break;
                    case 't': s.push_back('\t'); break;
                    default:
                        error("unsupported escape sequence");
                }
            } else {
                s.push_back(c);
            }
        }

        return JsonValue(std::move(s));
    }

    JsonValue parseNumber() {
        size_t start = pos_;
        if (peek() == '-') get();

        if (!std::isdigit(peek())) error("invalid number");

        while (std::isdigit(peek())) get();

        bool isFloat = false;

        if (peek() == '.') {
            isFloat = true;
            get();
            if (!std::isdigit(peek())) error("invalid number");
            while (std::isdigit(peek())) get();
        }

        if (peek() == 'e' || peek() == 'E') {
            isFloat = true;
            get();
            if (peek() == '+' || peek() == '-') get();
            if (!std::isdigit(peek())) error("invalid exponent");
            while (std::isdigit(peek())) get();
        }

        const std::string num = src_.substr(start, pos_ - start);
        if (isFloat) {
            return JsonValue(std::stod(num));
        }
        return JsonValue(static_cast<int64_t>(std::stoll(num)));
    }

    JsonValue parseBool() {
        if (src_.compare(pos_, 4, "true") == 0) {
            pos_ += 4;
            return JsonValue(true);
        }
        if (src_.compare(pos_, 5, "false") == 0) {
            pos_ += 5;
            return JsonValue(false);
        }
        error("invalid boolean");
        return {};
    }

    JsonValue parseNull() {
        if (src_.compare(pos_, 4, "null") == 0) {
            pos_ += 4;
            return JsonValue(nullptr);
        }
        error("invalid null");
        return {};
    }

    void expect(char c) {
        if (get() != c) {
            error(std::string("expected '") + c + "'");
        }
    }

private:
    const std::string& src_;
    size_t pos_;
};


JsonValue JsonParser::parse(const std::string& jsonStr) {
    JsonParserImpl p(jsonStr);
    return p.parse();
}

JsonValue JsonParser::parseFile(const std::string& filePath) {
    std::ifstream ifs(filePath);
    if (!ifs) {
        throw std::runtime_error("Failed to open file: " + filePath);
    }
    std::stringstream ss;
    ss << ifs.rdbuf();
    JsonParserImpl p(ss.str());
    return p.parse();
}
} // namespace ttsinfer
