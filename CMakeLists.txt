cmake_minimum_required(VERSION 3.14)  # for add_link_options and implicit target directories.
project("bitnet.cpp" C CXX)
include(CheckIncludeFileCXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ─── Level 1: kernel format ──────────────────────────────────────────────────
option(BITNET_ARM_TL1    "bitnet.cpp: use TL1 lookup-table kernel (ARM64)"   OFF)
option(BITNET_X86_TL2    "bitnet.cpp: use TL2 lookup-table kernel (x86_64)"  OFF)

# ─── Level 2-5: math research kernels ────────────────────────────────────────
option(BITNET_L2_WHT      "bitnet.cpp: WHT zero-mul GEMV (Level 2)"           ON)
option(BITNET_L3_ACDC     "bitnet.cpp: FWHT+ACDC O(n log n) layers (Level 3)" ON)
option(BITNET_L4_TROPICAL "bitnet.cpp: Tropical attention (max,+) (Level 4)"  ON)
option(BITNET_L5_HRR      "bitnet.cpp: Holographic memory HRR (Level 5)"      ON)
option(BITNET_BUILD_TESTS "bitnet.cpp: build kernel unit tests"               ON)

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

# ─── src/ ─────────────────────────────────────────────────────────────────────
# Compiles L2-L5 into the bitnet_math OBJECT library.
# Sets BITNET_MATH_TARGET in this scope (empty string if no levels enabled).
add_subdirectory(src)

# ─── 3rdparty/llama.cpp ───────────────────────────────────────────────────────
# Defines the ggml target (which already contains L1 kernels via hardcoded paths).
set(LLAMA_BUILD_SERVER ON CACHE BOOL "Build llama.cpp server" FORCE)
add_subdirectory(3rdparty/llama.cpp)

# ─── Wire L2-L5 into ggml ────────────────────────────────────────────────────
# After both subdirectories are processed, both `bitnet_math` and `ggml` exist.
# We add the OBJECT library to ggml so L2-L5 symbols are available in all
# llama.cpp binaries (llama-cli, llama-server, llama-bench, etc.)
# without any extra linker flags on the caller side.
if (BITNET_MATH_TARGET)
    target_link_libraries(ggml PUBLIC ${BITNET_MATH_TARGET})
    message(STATUS "BitNet: L2-L5 kernels linked into ggml target")
endif()

# ─── Tests ────────────────────────────────────────────────────────────────────
# Standalone unit tests for L2-L5 kernels. Add -DBITNET_BUILD_TESTS=OFF to skip.
if (BITNET_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
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
