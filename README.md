# ttsinfer

## How to build

### Prerequisites

- A C++17-capable compiler (Apple Clang, Clang, or GCC)
- CMake 3.15+

### Build (out-of-source)

```bash
cmake -S . -B build 
cmake --build build -j
```

### Run tests

```bash
cd build
ctest --output-on-failure
```
