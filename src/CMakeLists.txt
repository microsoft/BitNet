set(GGML_HEADERS_BITNET ../include/ggml-bitnet.h ../3rdparty/llama.cpp/ggml/src/ggml/src/ggml-cpu/ggml-cpu-impl.h)
set(GGML_SOURCES_BITNET ggml-bitnet-mad.cpp ggml-bitnet-lut.cpp)

include_directories(3rdparty/llama.cpp/ggml/include 3rdparty/llama.cpp/ggml/src)

if (NOT (CMAKE_C_COMPILER_ID MATCHES "Clang" OR CMAKE_C_COMPILER_ID STREQUAL "GNU") OR
    NOT (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
    message(FATAL_ERROR "Clang or GCC is required for Bitnet.cpp compilation")
endif()
