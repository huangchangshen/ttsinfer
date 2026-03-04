#pragma once

#include "tokenizer.h"

#include <unordered_map>
#include <utility>
#include <vector>

namespace ttsinfer::tokenizer {

struct PairHash {
    size_t operator()(const std::pair<std::string, std::string>& p) const noexcept;
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
    std::unordered_map<Token, TokenId> token_to_id_;
    std::unordered_map<TokenId, Token> id_to_token_;
    std::vector<std::pair<Token, Token>> merges_;
    std::unordered_map<std::pair<Token, Token>, int, PairHash> merge_ranks_;

    std::unordered_map<int, std::string> byte_to_char_;
    std::unordered_map<std::string, int> char_to_byte_;

    void InitByteMapping();
    std::string TextToByteTokens(const std::string& text) const;
    std::string ByteTokensToText(const std::string& byte_tokens) const;

    std::vector<Token> BpeTokenize(const std::string& byte_tokens) const;
};

} // namespace ttsinfer::tokenizer
