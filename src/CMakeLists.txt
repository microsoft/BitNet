set(GGML_HEADERS_BITNET ../include/ggml-bitnet.h)
set(GGML_SOURCES_BITNET ggml-bitnet-mad.cpp)
set(GGML_SOURCES_BITNET ggml-bitnet-lut.cpp)

include_directories(3rdparty/llama.cpp/ggml/include)


if(CMAKE_VERSION VERSION_LESS "3.22")
    message(FATAL_ERROR "CMake version 3.22 or higher is required for Bitnet.cpp compilation")
endif()

if (NOT (CMAKE_C_COMPILER_ID MATCHES "Clang" OR CMAKE_C_COMPILER_ID STREQUAL "GNU") OR
    NOT (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
    message(FATAL_ERROR "Clang or GCC is required for Bitnet.cpp compilation")
else()
    # Check for minimum Clang version: 
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
        # on Windows we use the Visual Studio Clang compiler (in 2022 it is version 17.)
        message(STATUS "Visual Studio Clang compiler is being used.")
        if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 17.0)
            message(FATAL_ERROR "Visual Studio Clang version 17.0 or higher is required for Bitnet.cpp compilation")
        endif()
    else()
        # on Linux Clang compiler >= 18.0 is required.
        message(STATUS "Linux/Mac Clang compiler is being used.")
        if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 18.0)
            message(FATAL_ERROR "Clang version 18.0 or higher is required for Bitnet.cpp compilation")
        endif()
    endif()
endif()

