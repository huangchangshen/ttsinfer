#include <iostream>
#include <cstdlib>
#include "backend.h"

#define ASSERT_TRUE(cond) \
    if (!(cond)) { \
        std::cerr << "ASSERT_TRUE failed: " #cond \
                  << " at line " << __LINE__ << "\n"; \
        std::exit(1); \
    }

#define ASSERT_EQ(a, b) \
    if (!((a) == (b))) { \
        std::cerr << "ASSERT_EQ failed: " #a " != " #b \
                  << " at line " << __LINE__ << "\n"; \
        std::exit(1); \
    }

using namespace ttsinfer;

Backend* create_test() {
    Backend* b = new Backend();
    b->name = "test";
    b->support_op = nullptr;
    b->compute = nullptr;
    return b;
}

static BackendRegistrar reg_test("test", create_test);

void test_registration() {
    auto& registry = BackendRegistry::instance();

    ASSERT_TRUE(registry.contains("test"));
    ASSERT_TRUE(registry.size() >= 1);
}

void test_create() {

    auto* backend =
        BackendRegistry::instance().create("test");

    ASSERT_TRUE(backend != nullptr);
    ASSERT_EQ(std::string(backend->name), "test");
}

void test_not_found() {

    auto* backend =
        BackendRegistry::instance().create("not_exist");

    ASSERT_TRUE(backend == nullptr);
}

int main() {
    std::cout << "Running BackendRegistry tests...\n";

    test_registration();
    test_create();
    test_not_found();

    std::cout << "All tests passed.\n";
}