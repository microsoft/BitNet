# ─── Compiler check ──────────────────────────────────────────────────────────
# Clang or GCC required; MSVC not supported for SIMD kernels.
if (NOT (CMAKE_C_COMPILER_ID   MATCHES "Clang|GNU") OR
    NOT (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU"))
    message(FATAL_ERROR "Clang or GCC is required for bitnet.cpp compilation")
endif()

# ─── L1: I2_S + LUT kernels ──────────────────────────────────────────────────
# These are hardcoded into 3rdparty/llama.cpp/ggml/src/CMakeLists.txt via
# relative paths (../../../../src/ggml-bitnet-mad.cpp, etc.) and compiled as
# part of the ggml OBJECT library there.  Nothing to do here for L1.

# ─── L2–L5: math kernels + dispatch layer ────────────────────────────────────
# Compiled as an OBJECT library linked into the ggml target.
#
# The dispatch source (ggml-bitnet-dispatch.cpp) uses ggml.h types
# (struct ggml_tensor, ggml_map_custom*).  Since dispatch.cpp compiles into
# the same OBJECT library that is then linked INTO ggml, forward references to
# ggml symbols are resolved at link time with no circular-dep issues.
#
# ggml's own headers are in 3rdparty/llama.cpp/ggml/include — added below.

set(_bitnet_math_srcs)
set(_bitnet_math_defs)
set(_bitnet_has_dispatch OFF)

if (BITNET_L2_WHT)
    list(APPEND _bitnet_math_srcs ${CMAKE_CURRENT_SOURCE_DIR}/ggml-bitnet-wht.cpp)
    list(APPEND _bitnet_math_defs BITNET_L2_WHT)
    message(STATUS "BitNet: Level 2 WHT zero-mul GEMV enabled")
endif()

if (BITNET_L3_ACDC)
    list(APPEND _bitnet_math_srcs ${CMAKE_CURRENT_SOURCE_DIR}/ggml-bitnet-fwht.cpp)
    list(APPEND _bitnet_math_defs BITNET_L3_ACDC)
    set(_bitnet_has_dispatch ON)
    message(STATUS "BitNet: Level 3 FWHT+ACDC O(n log n) enabled")
endif()

if (BITNET_L4_TROPICAL)
    list(APPEND _bitnet_math_srcs ${CMAKE_CURRENT_SOURCE_DIR}/ggml-bitnet-tropical.cpp)
    list(APPEND _bitnet_math_defs BITNET_L4_TROPICAL)
    set(_bitnet_has_dispatch ON)
    message(STATUS "BitNet: Level 4 Tropical attention (max,+) enabled")
endif()

if (BITNET_L5_HRR)
    list(APPEND _bitnet_math_srcs ${CMAKE_CURRENT_SOURCE_DIR}/ggml-bitnet-hrr.cpp)
    list(APPEND _bitnet_math_defs BITNET_L5_HRR)
    set(_bitnet_has_dispatch ON)
    message(STATUS "BitNet: Level 5 Holographic memory (HRR) enabled")
endif()

# ggml-bitnet-dispatch.cpp registers custom ops (ggml_map_custom2/3) for L3-L5.
# Compiled when at least one of L3/L4/L5 is enabled.
if (_bitnet_has_dispatch)
    list(APPEND _bitnet_math_srcs ${CMAKE_CURRENT_SOURCE_DIR}/ggml-bitnet-dispatch.cpp)
    message(STATUS "BitNet: dispatch layer (L3-L5 custom ops) enabled")
endif()

if (_bitnet_math_srcs)
    # OBJECT library: sources compiled once, objects reused by ggml and any
    # other target (e.g. standalone test binaries) without duplication.
    add_library(bitnet_math OBJECT ${_bitnet_math_srcs})

    target_include_directories(bitnet_math PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        # ggml.h needed by ggml-bitnet-dispatch.cpp (ggml_map_custom*, struct ggml_tensor)
        ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/llama.cpp/ggml/include)

    target_compile_features(bitnet_math PUBLIC cxx_std_11)

    # Propagate level defines so callers can use #ifdef BITNET_L2_WHT etc.
    target_compile_definitions(bitnet_math PUBLIC ${_bitnet_math_defs})

    # SIMD: apply per-architecture flags.
    # ggml already gates AVX2/NEON via its own detection; we mirror that here
    # so the kernel #if __AVX2__ / #if __ARM_NEON paths compile correctly.
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i686")
        target_compile_options(bitnet_math PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma>)
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        target_compile_options(bitnet_math PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-march=armv8-a+simd>)
    endif()

    # libm: required for HRR (cos, sin, sqrt) and tropical (expf) on Linux.
    # macOS and Windows link math implicitly.
    if (BITNET_L5_HRR OR BITNET_L4_TROPICAL)
        if (UNIX AND NOT APPLE)
            target_link_libraries(bitnet_math PUBLIC m)
        endif()
    endif()

    # Expose the target name to the parent scope so CMakeLists.txt can link it
    # into ggml after add_subdirectory(3rdparty/llama.cpp).
    set(BITNET_MATH_TARGET bitnet_math PARENT_SCOPE)
else()
    set(BITNET_MATH_TARGET "" PARENT_SCOPE)
    message(STATUS "BitNet: no L2-L5 math kernels enabled (use -DBITNET_L2_WHT=ON etc.)")
endif()
