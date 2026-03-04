#include "bpe.h"

#include <algorithm>
#include <climits>
#include <cctype>
#include <fstream>
#include <sstream>

namespace ttsinfer::tokenizer {

namespace {

std::string utf8_from_codepoint(int cp) {
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

bool parse_vocab_json(
    const std::string& path,
    std::unordered_map<std::string, int>& vocab) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        return false;
    }

    std::string s((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    size_t i = 0;
    const size_t n = s.size();
    while (i < n) {
        while (i < n && s[i] != '"') {
            ++i;
        }
        if (i >= n) {
            break;
        }
        ++i;

        std::string key;
        while (i < n) {
            char c = s[i++];
            if (c == '\\') {
                if (i >= n) {
                    break;
                }
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
                    unsigned int code = 0;
                    for (int k = 0; k < 4; ++k) {
                        char h = s[i++];
                        code <<= 4;
                        if (h >= '0' && h <= '9') {
                            code |= (h - '0');
                        } else if (h >= 'a' && h <= 'f') {
                            code |= (h - 'a' + 10);
                        } else if (h >= 'A' && h <= 'F') {
                            code |= (h - 'A' + 10);
                        }
                    }
                    key += utf8_from_codepoint(static_cast<int>(code));
                }
            } else if (c == '"') {
                break;
            } else {
                key.push_back(c);
            }
        }

        while (i < n && std::isspace(static_cast<unsigned char>(s[i]))) {
            ++i;
        }
        if (i >= n || s[i] != ':') {
            continue;
        }
        ++i;

        while (i < n && std::isspace(static_cast<unsigned char>(s[i]))) {
            ++i;
        }

        bool neg = false;
        if (i < n && s[i] == '-') {
            neg = true;
            ++i;
        }
        int value = 0;
        bool has_digit = false;
        while (i < n && s[i] >= '0' && s[i] <= '9') {
            has_digit = true;
            value = value * 10 + (s[i] - '0');
            ++i;
        }
        if (!has_digit || key.empty()) {
            continue;
        }
        if (neg) {
            value = -value;
        }

        vocab[key] = value;
    }

    return !vocab.empty();
}

} // namespace

size_t PairHash::operator()(const std::pair<std::string, std::string>& p) const noexcept {
    return std::hash<std::string>()(p.first) ^ (std::hash<std::string>()(p.second) << 1);
}

void BPE::InitByteMapping() {
    byte_to_char_.clear();
    char_to_byte_.clear();

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

    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }

    for (int i = 0; i < 256; ++i) {
        int b = bs[i];
        int cp = cs[i];
        std::string uni = utf8_from_codepoint(cp);
        byte_to_char_[b] = uni;
        char_to_byte_[uni] = b;
    }
}

std::string BPE::TextToByteTokens(const std::string& text) const {
    std::string result;
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(text.data());
    for (size_t i = 0; i < text.size(); ++i) {
        int b = static_cast<int>(bytes[i]);
        auto it = byte_to_char_.find(b);
        if (it != byte_to_char_.end()) {
            result += it->second;
        } else {
            result.push_back(static_cast<char>(bytes[i]));
        }
    }
    return result;
}

std::string BPE::ByteTokensToText(const std::string& byte_tokens) const {
    std::string result;
    size_t i = 0;
    const size_t n = byte_tokens.size();

    while (i < n) {
        bool matched = false;
        for (int len = 3; len >= 1; --len) {
            if (i + static_cast<size_t>(len) > n) {
                continue;
            }
            std::string candidate = byte_tokens.substr(i, static_cast<size_t>(len));
            auto it = char_to_byte_.find(candidate);
            if (it != char_to_byte_.end()) {
                result.push_back(static_cast<char>(it->second));
                i += static_cast<size_t>(len);
                matched = true;
                break;
            }
        }
        if (!matched) {
            result.push_back(byte_tokens[i]);
            ++i;
        }
    }

    return result;
}

std::vector<BaseTokenizer::Token> BPE::BpeTokenize(const std::string& byte_tokens) const {
    std::vector<Token> word;
    size_t i = 0;
    const size_t n = byte_tokens.size();
    while (i < n) {
        bool matched = false;
        for (int len = 3; len >= 1; --len) {
            if (i + static_cast<size_t>(len) > n) {
                continue;
            }
            std::string candidate = byte_tokens.substr(i, static_cast<size_t>(len));
            if (char_to_byte_.find(candidate) != char_to_byte_.end()) {
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

    if (word.empty() || merges_.empty()) {
        return word;
    }

    std::vector<Token> current = word;
    while (true) {
        std::vector<std::pair<Token, Token>> pairs;
        for (size_t i = 0; i + 1 < current.size(); ++i) {
            pairs.emplace_back(current[i], current[i + 1]);
        }

        if (pairs.empty()) {
            break;
        }

        std::pair<Token, Token> bigram;
        bool found = false;
        int min_rank = INT_MAX;
        for (const auto& pair : pairs) {
            auto it = merge_ranks_.find(pair);
            if (it != merge_ranks_.end() && it->second < min_rank) {
                min_rank = it->second;
                bigram = pair;
                found = true;
            }
        }

        if (!found) {
            break;
        }

        std::vector<Token> new_word;
        size_t idx = 0;
        while (idx < current.size()) {
            bool merged = false;
            if (idx + 1 < current.size() &&
                current[idx] == bigram.first &&
                current[idx + 1] == bigram.second) {
                new_word.push_back(bigram.first + bigram.second);
                merged = true;
                idx += 2;
            }
            if (!merged) {
                new_word.push_back(current[idx]);
                ++idx;
            }
        }

        current = std::move(new_word);
        if (current.size() == 1) {
            break;
        }
    }

    return current;
}

int BPE::Init() {
    if (token_to_id_.empty()) {
        return ERR_VOCAB_NOT_FOUND;
    }

    if (byte_to_char_.empty()) {
        InitByteMapping();
    }

    vocab_size_ = static_cast<int32_t>(token_to_id_.size());
    return SUCCESS;
}

int BPE::LoadVocab(const std::string& vocab_file) {
    std::unordered_map<std::string, int> vocab;
    if (!parse_vocab_json(vocab_file, vocab)) {
        return ERR_VOCAB_NOT_FOUND;
    }

    token_to_id_.clear();
    id_to_token_.clear();
    for (const auto& kv : vocab) {
        token_to_id_[kv.first] = kv.second;
        id_to_token_[kv.second] = kv.first;
    }
    vocab_size_ = static_cast<int32_t>(token_to_id_.size());

    return SUCCESS;
}

int BPE::LoadMerges(const std::string& merges_file) {
    std::ifstream file(merges_file);
    if (!file.is_open()) {
        return ERR_VOCAB_NOT_FOUND;
    }

    merges_.clear();
    merge_ranks_.clear();

    std::string line;
    std::getline(file, line);

    int rank = 0;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        std::istringstream iss(line);
        Token first;
        Token second;
        if (iss >> first >> second) {
            merges_.emplace_back(first, second);
            merge_ranks_[{first, second}] = rank;
            ++rank;
        }
    }

    if (byte_to_char_.empty()) {
        InitByteMapping();
    }

    return SUCCESS;
}

void BPE::SetVocab(const std::unordered_map<Token, TokenId>& vocab) {
    token_to_id_ = vocab;
    id_to_token_.clear();
    for (const auto& kv : vocab) {
        id_to_token_[kv.second] = kv.first;
    }
    vocab_size_ = static_cast<int32_t>(token_to_id_.size());

    if (byte_to_char_.empty()) {
        InitByteMapping();
    }
}

int BPE::Tokenize(const std::string& text, std::vector<Token>& tokens_out) const {
    if (token_to_id_.empty()) {
        return ERR_VOCAB_NOT_FOUND;
    }

    std::string byte_tokens = TextToByteTokens(text);
    std::vector<Token> bpe_tokens = BpeTokenize(byte_tokens);

    tokens_out.clear();
    for (const auto& token : bpe_tokens) {
        tokens_out.push_back(token);
    }

    return SUCCESS;
}

int BPE::Encode(const std::string& text, std::vector<TokenId>& ids_out) const {
    if (token_to_id_.empty()) {
        return ERR_VOCAB_NOT_FOUND;
    }

    std::string byte_tokens = TextToByteTokens(text);
    std::vector<Token> bpe_tokens = BpeTokenize(byte_tokens);

    ids_out.clear();
    for (const auto& token : bpe_tokens) {
        auto it = token_to_id_.find(token);
        if (it != token_to_id_.end()) {
            ids_out.push_back(it->second);
        }
    }

    return SUCCESS;
}

int BPE::Decode(const std::vector<TokenId>& ids, std::string& text_out) const {
    text_out.clear();

    std::string byte_tokens;
    for (TokenId id : ids) {
        auto it = id_to_token_.find(id);
        if (it == id_to_token_.end()) {
            continue;
        }
        byte_tokens += it->second;
    }

    text_out = ByteTokensToText(byte_tokens);
    return SUCCESS;
}

} // namespace ttsinfer::tokenizer
