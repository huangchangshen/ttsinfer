#pragma once

#include "tokenizer.hpp"
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <set>
#include <climits>

namespace ttsinfer::tokenizer
{

struct PairHash {
    size_t operator()(const std::pair<std::string, std::string>& p) const noexcept {
        return std::hash<std::string>()(p.first) ^ (std::hash<std::string>()(p.second) << 1);
    }
};

class BPE : public BaseTokenizer {
public:
    BPE() = default;
    ~BPE() override = default;

    int Init() override;
    int Tokenize(const std::string& text, std::vector<Token>& tokens_out) const override;
    int Encode(const std::string& text, std::vector<TokenId>& ids_out) const override;
    int Decode(const std::vector<TokenId>& ids, std::string& text_out) const override;

    int LoadVocab(const std::string& vocab_file);
    int LoadMerges(const std::string& merges_file);

    void SetVocab(const std::unordered_map<Token, TokenId>& vocab);

private:
    std::unordered_map<Token, TokenId> _token_to_id;
    std::unordered_map<TokenId, Token> _id_to_token;
    std::vector<std::pair<Token, Token>> _merges;
    std::unordered_map<std::pair<Token, Token>, int, PairHash> _merge_ranks;

    std::unordered_map<int, std::string> _byte_to_char;
    std::unordered_map<std::string, int> _char_to_byte;

    void InitByteMapping();
    std::string TextToByteTokens(const std::string& text) const;
    std::string ByteTokensToText(const std::string& byte_tokens) const;

    std::vector<Token> BpeTokenize(const std::string& byte_tokens) const;
};

inline static std::string utf8_from_codepoint(int cp) {
    std::string out;
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return out;
}

inline void BPE::InitByteMapping() {
    _byte_to_char.clear();
    _char_to_byte.clear();

    // Exact same logic as HF bytes_to_unicode()
    std::vector<int> bs;
    for (int i = static_cast<int>('!'); i <= static_cast<int>('~'); ++i) {
        bs.push_back(i);
    }
    for (int i = 161; i <= 172; ++i) {
        bs.push_back(i);
    }
    for (int i = 174; i <= 255; ++i) {
        bs.push_back(i);
    }

    std::vector<int> cs = bs; // copy
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }

    // Now map first 256 entries
    for (int i = 0; i < 256; ++i) {
        int b = bs[i];
        int cp = cs[i];
        std::string uni = utf8_from_codepoint(cp);
        _byte_to_char[b] = uni;
        _char_to_byte[uni] = b;
    }
}

inline std::string BPE::TextToByteTokens(const std::string& text) const {
    // text is UTF-8; treat as raw bytes and map each byte via _byte_to_char
    std::string result;
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(text.data());
    for (size_t i = 0; i < text.size(); ++i) {
        int b = static_cast<int>(bytes[i]);
        auto it = _byte_to_char.find(b);
        if (it != _byte_to_char.end()) {
            result += it->second; // this is UTF-8 sequence of the mapped codepoint
        } else {
            result.push_back(static_cast<char>(bytes[i]));
        }
    }
    return result;
}

inline std::string BPE::ByteTokensToText(const std::string& byte_tokens) const {
    // byte_tokens is concatenation of mapped codepoints (UTF-8). We need to invert them.
    std::string result;
    size_t i = 0;
    const size_t n = byte_tokens.size();

    while (i < n) {
        bool matched = false;
        // mapped codepoints are at most 3 bytes in this scheme
        for (int len = 3; len >= 1; --len) {
            if (i + static_cast<size_t>(len) > n) continue;
            std::string candidate = byte_tokens.substr(i, static_cast<size_t>(len));
            auto it = _char_to_byte.find(candidate);
            if (it != _char_to_byte.end()) {
                result.push_back(static_cast<char>(it->second));
                i += static_cast<size_t>(len);
                matched = true;
                break;
            }
        }
        if (!matched) {
            // Fallback: pass through one byte
            result.push_back(byte_tokens[i]);
            ++i;
        }
    }
    return result;
}

inline std::vector<BaseTokenizer::Token> BPE::BpeTokenize(const std::string& byte_tokens) const {
    // Split byte_tokens into base tokens: each mapped codepoint (UTF-8 sequence)
    std::vector<Token> word;
    size_t i = 0;
    const size_t n = byte_tokens.size();
    while (i < n) {
        bool matched = false;
        for (int len = 3; len >= 1; --len) {
            if (i + static_cast<size_t>(len) > n) continue;
            std::string candidate = byte_tokens.substr(i, static_cast<size_t>(len));
            if (_char_to_byte.find(candidate) != _char_to_byte.end()) {
                word.push_back(candidate);
                i += static_cast<size_t>(len);
                matched = true;
                break;
            }
        }
        if (!matched) {
            word.push_back(byte_tokens.substr(i, 1));
            ++i;
        }
    }

    if (word.empty() || _merges.empty()) {
        return word;
    }

    std::vector<Token> current = word;

    while (true) {
        // Find all pairs in current word
        std::vector<std::pair<Token, Token>> pairs;
        for (size_t i = 0; i + 1 < current.size(); ++i) {
            pairs.emplace_back(current[i], current[i + 1]);
        }

        if (pairs.empty()) {
            break;
        }

        // Find the pair with lowest rank (earliest in merges list)
        std::pair<Token, Token> bigram;
        bool found = false;
        int min_rank = INT_MAX;

        for (const auto& pair : pairs) {
            auto it = _merge_ranks.find(pair);
            if (it != _merge_ranks.end()) {
                if (it->second < min_rank) {
                    min_rank = it->second;
                    bigram = pair;
                    found = true;
                }
            }
        }

        if (!found) {
            break;
        }

        // Merge the bigram
        std::vector<Token> new_word;
        size_t i = 0;
        while (i < current.size()) {
            bool merged = false;
            if (i + 1 < current.size() &&
                current[i] == bigram.first &&
                current[i + 1] == bigram.second) {
                new_word.push_back(bigram.first + bigram.second);
                merged = true;
                i += 2;
            }
            if (!merged) {
                new_word.push_back(current[i]);
                ++i;
            }
        }

        current = std::move(new_word);

        if (current.size() == 1) {
            break;
        }
    }

    return current;
}

inline int BPE::Init() {
    if (_token_to_id.empty()) {
        return ERR_VOCAB_NOT_FOUND;
    }

    if (_byte_to_char.empty()) {
        InitByteMapping();
    }

    _vocab_size = static_cast<int32_t>(_token_to_id.size());
    return SUCCESS;
}

inline int BPE::LoadVocab(const std::string& vocab_file) {
    std::ifstream file(vocab_file);
    if (!file.is_open()) {
        return ERR_VOCAB_NOT_FOUND;
    }

    std::string line;
    TokenId id = 0;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            _token_to_id[line] = id;
            _id_to_token[id] = line;
            ++id;
        }
    }

    _vocab_size = id;
    return SUCCESS;
}

inline int BPE::LoadMerges(const std::string& merges_file) {
    std::ifstream file(merges_file);
    if (!file.is_open()) {
        return ERR_VOCAB_NOT_FOUND;
    }

    std::string line;
    std::getline(file, line);  // Skip the first line "#version: 0.2"

    int rank = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::istringstream iss(line);
        Token first, second;
        if (iss >> first >> second) {
            _merges.emplace_back(first, second);
            _merge_ranks[{first, second}] = rank;
            ++rank;
        }
    }

    if (_byte_to_char.empty()) {
        InitByteMapping();
    }

    return SUCCESS;
}

inline void BPE::SetVocab(const std::unordered_map<Token, TokenId>& vocab) {
    _token_to_id = vocab;
    _id_to_token.clear();
    for (const auto& [token, id] : vocab) {
        _id_to_token[id] = token;
    }
    _vocab_size = static_cast<int32_t>(_token_to_id.size());

    if (_byte_to_char.empty()) {
        InitByteMapping();
    }
}

inline int BPE::Tokenize(const std::string& text, std::vector<Token>& tokens_out) const {
    if (_token_to_id.empty()) {
        return ERR_VOCAB_NOT_FOUND;
    }

    // Convert text to byte-level tokens using bytes_to_unicode mapping
    std::string byte_tokens = TextToByteTokens(text);
    
    // Apply BPE
    std::vector<Token> bpe_tokens = BpeTokenize(byte_tokens);

    tokens_out.clear();
    for (const auto& token : bpe_tokens) {
        tokens_out.push_back(token);
    }

    return SUCCESS;
}

inline int BPE::Encode(const std::string& text, std::vector<TokenId>& ids_out) const {
    if (_token_to_id.empty()) {
        return ERR_VOCAB_NOT_FOUND;
    }

    // Convert text to byte-level tokens
    std::string byte_tokens = TextToByteTokens(text);
    
    // Apply BPE
    std::vector<Token> bpe_tokens = BpeTokenize(byte_tokens);

    ids_out.clear();
    for (const auto& token : bpe_tokens) {
        auto it = _token_to_id.find(token);
        if (it != _token_to_id.end()) {
            ids_out.push_back(it->second);
        }
    }

    return SUCCESS;
}

inline int BPE::Decode(const std::vector<TokenId>& ids, std::string& text_out) const {
    text_out.clear();

    std::string byte_tokens;
    for (TokenId id : ids) {
        auto it = _id_to_token.find(id);
        if (it == _id_to_token.end()) {
            continue;
        }
        byte_tokens += it->second;
    }

    // Convert back from byte-level tokens to text
    text_out = ByteTokensToText(byte_tokens);
    return SUCCESS;
}

}
