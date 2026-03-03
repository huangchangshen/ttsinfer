function(add_tts_test target_name)
    add_executable(${target_name} ${target_name}.cpp)
    target_link_libraries(${target_name} PRIVATE ttsinfer)
    add_test(NAME ${target_name} COMMAND ${target_name})
endfunction()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/model.safetensors
    ${CMAKE_CURRENT_BINARY_DIR}/model.safetensors
    COPYONLY
)

add_tts_test(test_json_basic)
add_tts_test(test_safetensors)
add_tts_test(test_tensor)
add_tts_test(test_graph)
add_tts_test(test_module)
add_tts_test(test_backend)