cmake_minimum_required(VERSION 3.28)
project(main)

set(CMAKE_CXX_STANDARD 17)

include_directories(.)

add_executable(main
        main.cpp
        tokenizer/tokenizer.hpp)
