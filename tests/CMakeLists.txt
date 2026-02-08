add_executable(test_json_basic
    test_json_basic.cpp
)


target_link_libraries(test_json_basic PRIVATE ttsinfer)

add_test(NAME test_json_basic COMMAND test_json_basic)

