if (BUILD_SHARED_LIBS OR NOT TARGET ggml-cpu OR GGML_BACKEND_DL OR GGML_CPU_ALL_VARIANTS)
    message(FATAL_ERROR "I2_S tests require a static, single ggml-cpu backend")
endif()
if (BITNET_ARM_TL1 OR BITNET_X86_TL2)
    message(FATAL_ERROR "I2_S tests require TL1 and TL2 to be disabled")
endif()

add_executable(test-i2s test-i2s.cpp)
target_compile_features(test-i2s PRIVATE cxx_std_11)
target_include_directories(test-i2s PRIVATE
    ../3rdparty/llama.cpp/ggml/include
    ../3rdparty/llama.cpp/ggml/src
    ../3rdparty/llama.cpp/ggml/src/ggml-cpu)
target_link_libraries(test-i2s PRIVATE ggml-cpu)

foreach(test_case dot gemm zero)
    add_test(NAME bitnet-i2s-${test_case} COMMAND test-i2s ${test_case})
endforeach()
