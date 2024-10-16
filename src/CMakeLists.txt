set(GGML_HEADERS_BITNET ../include/ggml-bitnet.h)
set(GGML_SOURCES_BITNET ggml-bitnet-mad.cpp)
set(GGML_SOURCES_BITNET ggml-bitnet-lut.cpp)

include_directories(3rdparty/llama.cpp/ggml/include)

if ((NOT ${CMAKE_C_COMPILER_ID} MATCHES "Clang") OR
(NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang"))
    message(FATAL_ERROR "Clang is required for Bitnet.cpp compilation")
endif()