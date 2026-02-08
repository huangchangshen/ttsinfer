configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/model.safetensors
    ${CMAKE_CURRENT_BINARY_DIR}/model.safetensors
    COPYONLY
)

add_executable(test_json_basic
    test_json_basic.cpp
)
target_link_libraries(test_json_basic PRIVATE ttsinfer)
add_test(NAME test_json_basic COMMAND test_json_basic)

add_executable(test_safetensors
    test_safetensors.cpp
)
target_link_libraries(test_safetensors PRIVATE ttsinfer)
add_test(NAME test_safetensors COMMAND test_safetensors)