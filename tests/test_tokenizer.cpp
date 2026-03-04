#include "bpe.h"

#include <cassert>
#include <iostream>
#include <vector>

using namespace ttsinfer::tokenizer;

int main() {
    BPE bpe;

    int ret = bpe.LoadVocab("./vocab.json");
    assert(ret == SUCCESS);

    ret = bpe.LoadMerges("./merges.txt");
    assert(ret == SUCCESS);

    ret = bpe.Init();
    assert(ret == SUCCESS);

    {
        const std::string text = "Hello world!";
        std::vector<int> ids;
        ret = bpe.Encode(text, ids);
        assert(ret == SUCCESS);
        assert(!ids.empty());

        std::string decoded;
        ret = bpe.Decode(ids, decoded);
        assert(ret == SUCCESS);
        assert(decoded == text);

        std::vector<std::string> tokens;
        ret = bpe.Tokenize(text, tokens);
        assert(ret == SUCCESS);
        assert(!tokens.empty());
    }

    {
        const std::string text = "你是谁！";
        std::vector<int> ids;
        ret = bpe.Encode(text, ids);
        assert(ret == SUCCESS);
        assert(!ids.empty());

        std::string decoded;
        ret = bpe.Decode(ids, decoded);
        assert(ret == SUCCESS);
        assert(decoded == text);
    }

    std::cout << "[OK] test_tokenizer passed\n";
    return 0;
}
