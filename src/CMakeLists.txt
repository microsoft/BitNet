set(GGML_HEADERS_BITNET ../include/ggml-bitnet.h PARENT_SCOPE)

# Include both parallel MAD kernels (ARM NEON / AVX2 / Scalar) and LUT kernels
# Prevents overwrite where ggml-bitnet-mad.cpp was dropped in CMake generation
set(GGML_SOURCES_BITNET
    ggml-bitnet-mad.cpp
    ggml-bitnet-lut.cpp
)
set(GGML_SOURCES_BITNET
    ggml-bitnet-mad.cpp
    ggml-bitnet-lut.cpp
    PARENT_SCOPE
)

include_directories(3rdparty/llama.cpp/ggml/include)

if (NOT (CMAKE_C_COMPILER_ID MATCHES "Clang" OR CMAKE_C_COMPILER_ID STREQUAL "GNU") OR
    NOT (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
    message(FATAL_ERROR "Clang or GCC is required for Bitnet.cpp compilation")
endif()
