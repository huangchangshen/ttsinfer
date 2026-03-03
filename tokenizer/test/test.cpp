#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "tokenizer/BPE.hpp"

// 非严格 JSON 解析器: 从 vocab.json 里抽取 "token": id 形式
static bool LoadVocabFromJson(
        const std::string& path,
        std::unordered_map<std::string, int>& vocab) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        std::cerr << "Failed to open vocab.json: " << path << std::endl;
        return false;
    }

    std::string s((std::istreambuf_iterator<char>(ifs)),
                  std::istreambuf_iterator<char>());

    size_t i = 0;
    const size_t n = s.size();

    while (i < n) {
        // 找 key 开始的引号
        while (i < n && s[i] != '"') ++i;
        if (i >= n) break;
        ++i; // 跳过 "

        // 解析 key（token）
        std::string key;
        while (i < n) {
            char c = s[i++];
            if (c == '\\') {
                if (i >= n) break;
                char esc = s[i++];
                if (esc == '"' || esc == '\\' || esc == '/') {
                    key.push_back(esc);
                } else if (esc == 'b') {
                    key.push_back('\b');
                } else if (esc == 'f') {
                    key.push_back('\f');
                } else if (esc == 'n') {
                    key.push_back('\n');
                } else if (esc == 'r') {
                    key.push_back('\r');
                } else if (esc == 't') {
                    key.push_back('\t');
                } else if (esc == 'u' && i + 4 <= n) {
                    // 简单处理 \uXXXX，只支持 BMP，常见中文足够
                    unsigned int code = 0;
                    for (int k = 0; k < 4; ++k) {
                        char h = s[i++];
                        code <<= 4;
                        if (h >= '0' && h <= '9') code |= (h - '0');
                        else if (h >= 'a' && h <= 'f') code |= (h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') code |= (h - 'A' + 10);
                    }
                    if (code <= 0x7F) {
                        key.push_back(static_cast<char>(code));
                    } else if (code <= 0x7FF) {
                        key.push_back(static_cast<char>(0xC0 | (code >> 6)));
                        key.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                    } else {
                        key.push_back(static_cast<char>(0xE0 | (code >> 12)));
                        key.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
                        key.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                    }
                }
            } else if (c == '"') {
                break;
            } else {
                key.push_back(c);
            }
        }

        // 跳到冒号
        while (i < n && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) ++i;
        if (i >= n || s[i] != ':') continue;
        ++i;

        // 跳过空白
        while (i < n && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) ++i;

        // 解析整数 id
        bool neg = false;
        if (i < n && s[i] == '-') {
            neg = true;
            ++i;
        }
        int value = 0;
        while (i < n && s[i] >= '0' && s[i] <= '9') {
            value = value * 10 + (s[i] - '0');
            ++i;
        }
        if (neg) value = -value;

        if (!key.empty()) {
            vocab[key] = value;
        }
    }

    return !vocab.empty();
}

int main() {
    using namespace ttsinfer::tokenizer;

    BPE bpe;

    system("chcp 65001");

    // 1. 从 IndexTTS2 的词表文件加载 vocab.json
    std::unordered_map<std::string, int> vocab;
    const std::string vocab_path  = "./vocab.json";
    const std::string merges_path = "./merges.txt";

    if (!LoadVocabFromJson(vocab_path, vocab)) {
        std::cerr << "Load vocab.json failed, path = " << vocab_path << std::endl;
        return 1;
    }

    bpe.SetVocab(vocab);

    int ret = bpe.LoadMerges(merges_path);
    if (ret != SUCCESS) {
        std::cerr << "Load merges.txt failed, path = " << merges_path
                  << ", err = " << ret << std::endl;
        return 1;
    }

    ret = bpe.Init();
    if (ret != SUCCESS) {
        std::cerr << "BPE Init failed, err = " << ret << std::endl;
        return 1;
    }

    // 2. 英文测试
    {
        std::string text = "Hello world!";
        std::vector<int> ids;
        ret = bpe.Encode(text, ids);
        std::cout << "---- English Test ----" << std::endl;
        if (ret == SUCCESS) {
            std::cout << "Input text: " << text << std::endl;
            std::cout << "Token IDs: ";
            for (int id : ids) std::cout << id << ' ';
            std::cout << std::endl;

            std::string decoded;
            ret = bpe.Decode(ids, decoded);
            if (ret == SUCCESS) {
                std::cout << "Decoded text: " << decoded << std::endl;
            } else {
                std::cout << "Decode failed, err = " << ret << std::endl;
            }
        } else {
            std::cout << "Encode failed, err = " << ret << std::endl;
        }

        std::vector<std::string> tokens;
        ret = bpe.Tokenize(text, tokens);
        if (ret == SUCCESS) {
            std::cout << "Tokens: ";
            for (const auto& t : tokens) std::cout << "'" << t << "' ";
            std::cout << std::endl;
        } else {
            std::cout << "Tokenize failed, err = " << ret << std::endl;
        }
    }

    // 3. 中文测试
    {
        std::string text = "你是谁！";
        std::vector<int> ids;
        ret = bpe.Encode(text, ids);
        std::cout << "---- Chinese Test ----" << std::endl;
        if (ret == SUCCESS) {
            std::cout << "输入文本: " << text << std::endl;
            std::cout << "Token IDs: ";
            for (int id : ids) std::cout << id << ' ';
            std::cout << std::endl;

            std::string decoded;
            ret = bpe.Decode(ids, decoded);
            if (ret == SUCCESS) {
                std::cout << "反解文本: " << decoded << std::endl;
            } else {
                std::cout << "Decode failed, err = " << ret << std::endl;
            }
        } else {
            std::cout << "Encode failed, err = " << ret << std::endl;
        }

        std::vector<std::string> tokens;
        ret = bpe.Tokenize(text, tokens);
        if (ret == SUCCESS) {
            std::cout << "Tokens: ";
            for (const auto& t : tokens) std::cout << "'" << t << "' ";
            std::cout << std::endl;
        } else {
            std::cout << "Tokenize failed, err = " << ret << std::endl;
        }
    }

    return 0;
}
