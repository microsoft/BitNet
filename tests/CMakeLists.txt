# ─── Kernel unit tests for bitnet.cpp ──────────────────────────────────────────
#
# Standalone executables that link directly against the L2-L5 math kernel
# source files. No model needed; runtime < 1ms each. Tests verify the kernel
# implementations against a hand-rolled reference (no ggml runtime).
#
# Enable with -DBITNET_BUILD_TESTS=ON (default ON).
# Run all tests:    ctest --output-on-failure
# Run one test:     ctest -R test_wht --output-on-failure

if (NOT BITNET_BUILD_TESTS)
    return()
endif()

if (NOT BITNET_MATH_TARGET)
    message(STATUS "BitNet: tests skipped (no L2-L5 math kernels enabled)")
    return()
endif()

# Helper: per-arch SIMD flags. Mirrors src/CMakeLists.txt.
function(bitnet_test_set_simd_flags target)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i686")
        target_compile_options(${target} PRIVATE -mavx2 -mfma)
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        target_compile_options(${target} PRIVATE -march=armv8-a+simd)
    endif()
    if (UNIX AND NOT APPLE)
        target_link_libraries(${target} PRIVATE m)
    endif()
endfunction()

# ─── Shared kernel utilities (bitnet_next_pow2) ──────────────────────────
# 5/5 PASS: basic, aliases (fwht/hrr forward to bitnet), edge cases (0/1/-1),
# structural (no butterfly is exported — see taxonomy in the header),
# power-of-2 inputs unchanged.
# This test guards against accidental API drift in the shared utility.
if (BITNET_L2_WHT OR BITNET_L3_ACDC OR BITNET_L4_TROPICAL OR BITNET_L5_HRR)
    add_executable(test_bitnet_common
        ${CMAKE_SOURCE_DIR}/test_bitnet_common.cpp
        ${CMAKE_SOURCE_DIR}/src/ggml-bitnet-common.cpp)
    target_include_directories(test_bitnet_common PRIVATE
        ${CMAKE_SOURCE_DIR}/include)
    target_compile_definitions(test_bitnet_common PRIVATE BITNET_L2_WHT)
    bitnet_test_set_simd_flags(test_bitnet_common)
    set_target_properties(test_bitnet_common PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
    add_test(NAME test_bitnet_common COMMAND test_bitnet_common)
endif()

# Each test compiles ONLY the kernel source it needs (not the full dispatch
# path, which references ggml symbols not available outside the llama.cpp
# build).  This keeps tests self-contained and < 200KB of object code each.

# ─── L2: Walsh-Hadamard Transform (zero-multiplication GEMV) ───────────────
# 5/5 PASS: raw_dot, sum_i8, verify, dot_row, gemv.
# (Bug found + fixed: wht_dot_avx2 had g0/g3 labels inverted relative to the
#  library's own unpack_i2s_block — see src/ggml-bitnet-wht.cpp:186-189.)
if (BITNET_L2_WHT)
    add_executable(test_wht
        ${CMAKE_SOURCE_DIR}/test_wht.cpp
        ${CMAKE_SOURCE_DIR}/src/ggml-bitnet-wht.cpp)
    target_include_directories(test_wht PRIVATE
        ${CMAKE_SOURCE_DIR}/include)
    target_compile_definitions(test_wht PRIVATE BITNET_L2_WHT)
    bitnet_test_set_simd_flags(test_wht)
    set_target_properties(test_wht PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
    add_test(NAME test_wht COMMAND test_wht)
endif()

# ─── L3: ACDC (Fast WHT + diagonal scaling) ────────────────────────────────
# 5/5 PASS: fwht_f32, fwht_i8_to_i32, acdc_forward_i8, acdc_project, acdc_gemv.
# (Bug found + fixed: acdc_forward_i8 had a stray 1/n² normalization that
#  violated the spec in CLAUDE.md ("unnormalized — no 1/n² factors").
#  Spec test [4] also updated: for W=I, d*[k] = 1/n, not 1.)
if (BITNET_L3_ACDC)
    add_executable(test_acdc
        ${CMAKE_SOURCE_DIR}/test_acdc.cpp
        ${CMAKE_SOURCE_DIR}/src/ggml-bitnet-fwht.cpp)
    target_include_directories(test_acdc PRIVATE
        ${CMAKE_SOURCE_DIR}/include)
    target_compile_definitions(test_acdc PRIVATE BITNET_L3_ACDC)
    bitnet_test_set_simd_flags(test_acdc)
    set_target_properties(test_acdc PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
    add_test(NAME test_acdc COMMAND test_acdc)
endif()

# ─── L4: Tropical attention (max,+) semiring ───────────────────────────────
# 5/5 PASS: argmax, topk, attention, gemv, zero-K edge case.
if (BITNET_L4_TROPICAL)
    add_executable(test_tropical
        ${CMAKE_SOURCE_DIR}/test_tropical.cpp
        ${CMAKE_SOURCE_DIR}/src/ggml-bitnet-tropical.cpp)
    target_include_directories(test_tropical PRIVATE
        ${CMAKE_SOURCE_DIR}/include)
    target_compile_definitions(test_tropical PRIVATE BITNET_L4_TROPICAL)
    bitnet_test_set_simd_flags(test_tropical)
    set_target_properties(test_tropical PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
    add_test(NAME test_tropical COMMAND test_tropical)

    # ─── L4-alt: Float sparse top-K attention ────────────────────────────
    # 5/5 PASS: K_top=0 returns zero, K_top=n_keys equals full softmax,
    # top-1 picks argmax, top-K partial_sort picks correct keys,
    # float scoring matches a hand-rolled reference implementation.
    # Guards sparse_attention_float (the kernel behind BITNET_SPARSE_TOPK).
    add_executable(test_sparse_attention
        ${CMAKE_SOURCE_DIR}/test_sparse_attention.cpp
        ${CMAKE_SOURCE_DIR}/src/ggml-bitnet-tropical.cpp
        ${CMAKE_SOURCE_DIR}/src/ggml-bitnet-common.cpp)
    target_include_directories(test_sparse_attention PRIVATE
        ${CMAKE_SOURCE_DIR}/include)
    target_compile_definitions(test_sparse_attention PRIVATE BITNET_L4_TROPICAL)
    bitnet_test_set_simd_flags(test_sparse_attention)
    set_target_properties(test_sparse_attention PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
    add_test(NAME test_sparse_attention COMMAND test_sparse_attention)
endif()

# ─── L5: HRR (Holographic Reduced Representations) ─────────────────────────
# 5/5 PASS: FFT roundtrip, bind, phasor inv,
# RESIDUAL Frady 2021, NAIVE projection.
if (BITNET_L5_HRR)
    add_executable(test_hrr_cleanup
        ${CMAKE_SOURCE_DIR}/test_hrr_cleanup.cpp
        ${CMAKE_SOURCE_DIR}/src/ggml-bitnet-hrr.cpp)
    target_include_directories(test_hrr_cleanup PRIVATE
        ${CMAKE_SOURCE_DIR}/include)
    target_compile_definitions(test_hrr_cleanup PRIVATE BITNET_L5_HRR)
    bitnet_test_set_simd_flags(test_hrr_cleanup)
    set_target_properties(test_hrr_cleanup PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
    add_test(NAME test_hrr_cleanup COMMAND test_hrr_cleanup)

    # ─── L5: HRR attention (dispatch kernel, no ggml wrapping) ─────────────
    # 5/5 PASS: single-query finite, multi-query independent, phasor exact,
    # gaussian finite, build+retrieve consistent with hrr_attention_full.
    # This guards the kernel that bitnet_op_hrr_attn and
    # bitnet_op_hrr_attn_with_cleanup invoke — a regression here would silently
    # corrupt L5 attention in the entire inference pipeline.
    add_executable(test_hrr_attention
        ${CMAKE_SOURCE_DIR}/test_hrr_attention.cpp
        ${CMAKE_SOURCE_DIR}/src/ggml-bitnet-hrr.cpp
        ${CMAKE_SOURCE_DIR}/src/ggml-bitnet-common.cpp)
    target_include_directories(test_hrr_attention PRIVATE
        ${CMAKE_SOURCE_DIR}/include)
    target_compile_definitions(test_hrr_attention PRIVATE BITNET_L5_HRR)
    bitnet_test_set_simd_flags(test_hrr_attention)
    set_target_properties(test_hrr_attention PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
    add_test(NAME test_hrr_attention COMMAND test_hrr_attention)
endif()
