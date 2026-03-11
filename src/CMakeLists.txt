set(GGML_HEADERS_BITNET ../include/ggml-bitnet.h)
set(GGML_SOURCES_BITNET ggml-bitnet-mad.cpp)
set(GGML_SOURCES_BITNET ggml-bitnet-lut.cpp)

# Add sparse-ternary-fma adapter if enabled
if (BITNET_USE_STFMA)
    list(APPEND GGML_HEADERS_BITNET ../include/ggml-bitnet-stfma.h)
    list(APPEND GGML_HEADERS_BITNET ../include/ggml-bitnet-stfma-cache.h)
    list(APPEND GGML_HEADERS_BITNET ../include/ggml-bitnet-stfma-avx512.h)
    list(APPEND GGML_SOURCES_BITNET ggml-bitnet-stfma.cpp)
    list(APPEND GGML_SOURCES_BITNET ggml-bitnet-stfma-cache.c)
    list(APPEND GGML_SOURCES_BITNET ggml-bitnet-stfma-avx512.cpp)
    list(APPEND GGML_SOURCES_BITNET ggml-bitnet-stfma-inference.cpp)
endif()

include_directories(3rdparty/llama.cpp/ggml/include)

if (NOT (CMAKE_C_COMPILER_ID MATCHES "Clang" OR CMAKE_C_COMPILER_ID STREQUAL "GNU") OR
    NOT (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
    message(FATAL_ERROR "Clang or GCC is required for Bitnet.cpp compilation")
endif()
