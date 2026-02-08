#include <cassert>
#include <iostream>
#include "json.h"

using namespace ttsinfer;

int main() {
    const char* json = R"json(
    {
        "name": "Qwen",
        "layers": 32,
        "hidden": 4096,
        "fp16": true,
        "scale": 1.25,
        "tags": ["llm", "transformer", "inference"],
        "extra": null
    }
    )json";

    JsonValue root = JsonParser::parse(json);

    assert(root.isObject());

    const auto& obj = root.as<JsonObject>();
    assert(obj.at("name").as<std::string>() == "Qwen");
    assert(obj.at("layers").as<int>() == 32);
    assert(obj.at("hidden").as<int>() == 4096);
    assert(obj.at("fp16").as<bool>() == true);
    assert(obj.at("scale").as<double>() > 1.0);

    const auto& tags = obj.at("tags").as<JsonArray>();
    assert(tags.size() == 3);
    assert(tags[0].as<std::string>() == "llm");

    assert(obj.at("extra").isNull());

    std::cout << "[OK] test_json_basic passed\n";
    return 0;
}
