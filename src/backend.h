#pragma once
#include "ttsinfer.h"
#include <unordered_set>
#include <vector>
#include <iostream>
#include <functional>

namespace ttsinfer {

const int kMaxBackends = 16;

struct Backend {
    const char* name;

    bool (*support_op)(int op);
    void (*compute)(int op);
};

class BackendRegistry {
public:
    struct Entry {
        const char* name;
        Backend* (*create)();
    };

    static BackendRegistry& instance() {
        static BackendRegistry r;
        return r;
    }

    void register_backend(const char* name, Backend* (*create)()) {
        if (count >= kMaxBackends) {
            std::cerr << "BackendRegistry full!\n";
            return;
        }
        factories[count++] = {name, create};
    }

    Backend* create(const char* name) {
        for (int i = 0; i < count; ++i) {
            if (std::strcmp(factories[i].name, name) == 0) {
                return factories[i].create();
            }
        }
        return nullptr;
    }

    int size() const { return count; }

    bool contains(const char* name) const {
        for (int i = 0; i < count; ++i) {
            if (std::strcmp(factories[i].name, name) == 0)
                return true;
        }
        return false;
    }

private:
    Entry factories[kMaxBackends];
    int count = 0;
};

struct BackendRegistrar {
    BackendRegistrar(const char* name, Backend* (*create)()) {
        BackendRegistry::instance().register_backend(name, create);
    }
};


}