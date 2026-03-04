cmake_minimum_required(VERSION 3.15)
project(ttsinfer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)

add_library(ttsinfer
    src/json.cpp
    src/safetensors.cpp
    src/graph.cpp
    src/bpe.cpp
)

enable_testing()
add_subdirectory(tests)
