#pragma once
#include "ttsinfer.h"
#include "backend.h"
#include <unordered_set>
#include <vector>
#include <iostream>
#include <functional>

namespace ttsinfer {

bool cpu_support(int op) { return true; }
void cpu_compute(int op) { std::cout << "CPU compute op " << op << "\n"; }

Backend* create_cpu() {
    Backend* b = new Backend();
    b->name = "cpu";
    b->support_op = cpu_support;
    b->compute = cpu_compute;
    return b;
}

static BackendRegistrar reg_cpu("cpu", create_cpu);

}