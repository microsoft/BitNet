cmake_minimum_required(VERSION 3.14)  # for add_link_options and implicit target directories.
project("bitnet.cpp" C CXX)
include(CheckIncludeFileCXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# option list
option(BITNET_ARM_TL1    "bitnet.cpp: use tl1 on arm platform"    OFF)
option(BITNET_X86_TL2    "bitnet.cpp: use tl2 on x86 platform"    OFF)


set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)
set(THREADS_PREFER_PTHREAD_FLAG ON)

# override ggml options
set(GGML_BITNET_ARM_TL1    ${BITNET_ARM_TL1})
set(GGML_BITNET_X86_TL2    ${BITNET_X86_TL2})

if (GGML_BITNET_ARM_TL1)
    add_compile_definitions(GGML_BITNET_ARM_TL1)
endif()
if (GGML_BITNET_X86_TL2)
    add_compile_definitions(GGML_BITNET_X86_TL2)
endif()

if (CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-fpermissive)
endif()

find_package(Threads REQUIRED)

add_subdirectory(src)
set(LLAMA_BUILD_SERVER ON CACHE BOOL "Build llama.cpp server" FORCE)
add_subdirectory(3rdparty/llama.cpp)

# Local build fix (bitnet-pruebas): src/CMakeLists.txt only sets
# GGML_SOURCES_BITNET (overwritten twice, never consumed), so the ternary
# kernels are never compiled and ggml-base.dll fails to link
# (undefined quantize_i2_s / dequantize_row_i2_s; the latter is shimmed in
# src/ggml-bitnet-shim.c). Compile the I2_S kernels directly into ggml-base
# with -march=native (required by gemm-config.h).
if (TARGET ggml-base)
    target_sources(ggml-base PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ggml-bitnet-mad.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ggml-bitnet-shim.c
    )
    if (GGML_BITNET_X86_TL2 OR GGML_BITNET_ARM_TL1)
        target_sources(ggml-base PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/ggml-bitnet-lut.cpp
        )
    endif()
    target_include_directories(ggml-base PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/llama.cpp/ggml/include
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/llama.cpp/ggml/src
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/llama.cpp/ggml/src/ggml-cpu
    )
    # gemm-config.h gates ROW/COL/PARALLEL_SIZE on __AVX__/__AVX2__.
    # -march=native is fastest on the build machine but not portable (CI
    # runners differ and can hit undeclared AVX-512 intrinsics), so make it
    # overridable from CI: -DBITNET_GGML_ARCH=avx2
    set(BITNET_GGML_ARCH "native" CACHE STRING "ISA for ggml-base (native, avx2, ...)")
    target_compile_options(ggml-base PRIVATE "-march=${BITNET_GGML_ARCH}")
    target_compile_definitions(ggml-base PRIVATE GGML_BUILD)
endif()

# install

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(LLAMA_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR}
    CACHE PATH "Location of header files")
set(LLAMA_LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}
    CACHE PATH "Location of library files")
set(LLAMA_BIN_INSTALL_DIR ${CMAKE_INSTALL_BINDIR}
    CACHE PATH "Location of binary files")
set(LLAMA_BUILD_NUMBER ${BUILD_NUMBER})
set(LLAMA_BUILD_COMMIT ${BUILD_COMMIT})
set(LLAMA_INSTALL_VERSION 0.0.${BUILD_NUMBER})

get_target_property(GGML_DIRECTORY ggml SOURCE_DIR)
get_directory_property(GGML_DIR_DEFINES DIRECTORY ${GGML_DIRECTORY} COMPILE_DEFINITIONS)
get_target_property(GGML_TARGET_DEFINES ggml COMPILE_DEFINITIONS)
set(GGML_TRANSIENT_DEFINES ${GGML_TARGET_DEFINES} ${GGML_DIR_DEFINES})
get_target_property(GGML_LINK_LIBRARIES ggml LINK_LIBRARIES)

get_directory_property(LLAMA_TRANSIENT_DEFINES COMPILE_DEFINITIONS)

write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/LlamaConfigVersion.cmake
    VERSION ${LLAMA_INSTALL_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/LlamaConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/LlamaConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Llama)

set_target_properties(llama PROPERTIES PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/llama.h)
install(TARGETS llama LIBRARY PUBLIC_HEADER)
