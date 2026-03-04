#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ttsinfer::tokenizer {

enum ErrorCode {
    SUCCESS = 0,
    ERR_INVALID_ARGUMENT = 1,
    ERR_VOCAB_NOT_FOUND = 2,
    ERR_INTERNAL = 3,
};

class BaseTokenizer {
public:
    using Token = std::string;
    using TokenId = int;

    virtual ~BaseTokenizer() = default;

    virtual int Init() = 0;
    virtual int Tokenize(const std::string& text, std::vector<Token>& tokens_out) const = 0;
    virtual int Encode(const std::string& text, std::vector<TokenId>& ids_out) const = 0;
    virtual int Decode(const std::vector<TokenId>& ids, std::string& text_out) const = 0;

    virtual int ConvertTokensToIds(
        const std::vector<Token>& tokens,
        std::vector<TokenId>& ids_out) const {
        return SUCCESS;
    }

    virtual int ConvertIdsToTokens(
        const std::vector<TokenId>& ids,
        std::vector<Token>& tokens_out) const {
        return SUCCESS;
    }

    virtual int GetVocabSize() const noexcept {
        return vocab_size_;
    }

    virtual int AddSpecialTokens(std::vector<TokenId>& special_tokens) {
        special_tokens_ = special_tokens;
        return SUCCESS;
    }

protected:
    int32_t vocab_size_ = 0;
    std::vector<TokenId> vocab_;
    std::vector<TokenId> special_tokens_;
};

} // namespace ttsinfer::tokenizer
