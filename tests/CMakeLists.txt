# ─── Kernel unit tests for bitnet.cpp ──────────────────────────────────────────
#
# Standalone executables that link directly against the L2-L5 math kernel
# source files. No model needed; runtime < 1ms each. Tests verify the kernel
# implementations against a hand-rolled reference (no ggml runtime).
#
# Enable with -DBITNET_BUILD_TESTS=ON (default ON).
# Run all tests:    ctest --output-on-failure
# Run one test:     ctest -R test_wht --output-on-failure
#
# NOTE (T003, 2026-06-06): Catch2 is **not** used in this project. All existing
# tests use hand-rolled `assert(...)` macros with `fprintf(stderr, ...)` for
# diagnostics and `return 1` on failure. This is intentional — it keeps the
# test runtime under 1ms and removes a heavy dependency for an already-trim
# CPU-only build. New T-actions (T005-T008) MUST follow the same convention.
# Pattern reference: test_bitnet_common.cpp in tests/ subdir (or root for older
# tests — see top-level test_wht.cpp, test_acdc.cpp etc.).

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

    # ─── L4 cache: K_i8 persistent cache for tropical attention ────────────
    # 11/11 PASS: init noop, realloc on shape change, first-call quantizes
    # all, incremental quantizes only new entries, no-new-keys is idempotent,
    # out-of-range returns NULL, capacity grows on demand, capacity capped at
    # max_n_kv, thread-safety (2 threads racing on same slot → 0 errors),
    # reset clears state, set_layer/current_layer roundtrip.
    # This guards the K_i8 cache that bitnet_op_tropical_attn uses to avoid
    # re-quantizing all K on every decode step (Phase C).
    add_executable(test_kv_i8_cache
        ${CMAKE_SOURCE_DIR}/test_kv_i8_cache.cpp
        ${CMAKE_SOURCE_DIR}/src/ggml-bitnet-kv-cache.cpp)
    target_include_directories(test_kv_i8_cache PRIVATE
        ${CMAKE_SOURCE_DIR}/include)
    target_compile_definitions(test_kv_i8_cache PRIVATE BITNET_L4_TROPICAL)
    bitnet_test_set_simd_flags(test_kv_i8_cache)
    set_target_properties(test_kv_i8_cache PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
    add_test(NAME test_kv_i8_cache COMMAND test_kv_i8_cache)
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

# ─── ACDC diagonal extraction (Python) ────────────────────────────────────
# 4/4 PASS: next_pow2 utility, exact recovery for ACDC-diagonalizable
# matrices (energy = 1.0), random W captures ~1/n energy (1/32 = 0.0312,
# actual ~0.035 within tolerance), W=I gives d*[0] = 1/n.
# This guards the closed-form d* = diag(H·W·H) / n² that
# extract_acdc_diagonal.py implements, which is the basis for the
# ACDC pretraining initialization (Phase A).
if (BITNET_L3_ACDC)
    find_package(Python3 COMPONENTS Interpreter)
    if (Python3_Interpreter_FOUND)
        add_test(NAME test_extract_acdc_diagonal
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/test_extract_acdc_diagonal.py
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
        set_tests_properties(test_extract_acdc_diagonal PROPERTIES
            LABELS "python;L3")
    else()
        message(STATUS "BitNet: skipping test_extract_acdc_diagonal (Python3 not found)")
    endif()
endif()

# ─── Property-based tests (RF-01, AC-02) — added by T024 ─────────────────
# Hand-rolled assert-based convention (see header note). Each test runs
# 100-1000 iterations with deterministic seeds. Total runtime < 1s.
# These are the "executable specification" referenced in P2
# (docs/invariants.md#p2).

# L3: ACDC properties — 4/4 PASS (T005)
if (BITNET_L3_ACDC)
    add_executable(test_acdc_properties
        ${CMAKE_SOURCE_DIR}/test_acdc_properties.cpp
        ${CMAKE_SOURCE_DIR}/src/ggml-bitnet-fwht.cpp
        ${CMAKE_SOURCE_DIR}/src/ggml-bitnet-common.cpp)
    target_include_directories(test_acdc_properties PRIVATE
        ${CMAKE_SOURCE_DIR}/include)
    target_compile_definitions(test_acdc_properties PRIVATE BITNET_L3_ACDC)
    bitnet_test_set_simd_flags(test_acdc_properties)
    set_target_properties(test_acdc_properties PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
    add_test(NAME test_acdc_properties COMMAND test_acdc_properties)
endif()

# L4: Sparse float properties — 3/3 PASS (T006)
if (BITNET_L4_TROPICAL)
    add_executable(test_l4_sparse_properties
        ${CMAKE_SOURCE_DIR}/test_l4_sparse_properties.cpp
        ${CMAKE_SOURCE_DIR}/src/ggml-bitnet-tropical.cpp)
    target_include_directories(test_l4_sparse_properties PRIVATE
        ${CMAKE_SOURCE_DIR}/include)
    target_compile_definitions(test_l4_sparse_properties PRIVATE BITNET_L4_TROPICAL)
    bitnet_test_set_simd_flags(test_l4_sparse_properties)
    set_target_properties(test_l4_sparse_properties PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
    add_test(NAME test_l4_sparse_properties COMMAND test_l4_sparse_properties)
endif()

# L5: HRR properties — 3/3 PASS (T007)
if (BITNET_L5_HRR)
    add_executable(test_hrr_properties
        ${CMAKE_SOURCE_DIR}/test_hrr_properties.cpp
        ${CMAKE_SOURCE_DIR}/src/ggml-bitnet-hrr.cpp
        ${CMAKE_SOURCE_DIR}/src/ggml-bitnet-common.cpp)
    target_include_directories(test_hrr_properties PRIVATE
        ${CMAKE_SOURCE_DIR}/include)
    target_compile_definitions(test_hrr_properties PRIVATE BITNET_L5_HRR)
    bitnet_test_set_simd_flags(test_hrr_properties)
    set_target_properties(test_hrr_properties PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
    add_test(NAME test_hrr_properties COMMAND test_hrr_properties)
endif()

# Dense-is-default (D-T-01, AC-06) — 3/3 PASS (T008)
# Static analysis (no kernel dep) — always built when tests are enabled.
add_executable(test_dense_is_default
    ${CMAKE_SOURCE_DIR}/test_dense_is_default.cpp)
target_include_directories(test_dense_is_default PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src)
target_compile_definitions(test_dense_is_default PRIVATE
    SOURCE_DIR="${CMAKE_SOURCE_DIR}")
bitnet_test_set_simd_flags(test_dense_is_default)
set_target_properties(test_dense_is_default PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
add_test(NAME test_dense_is_default COMMAND test_dense_is_default)

# ACDC rectangular (GATED by D2, T009) — only enabled if BITNET_ENABLE_ACDC_RECT=ON.
# Per requirements.md#10 (D2): only if T029 investigation confirms ACDC FFN is
# the blocker. Default OFF until gate resolves.
option(BITNET_ENABLE_ACDC_RECT "Enable ACDC rectangular shapes (gated by D2)" OFF)
if (BITNET_ENABLE_ACDC_RECT)
    if (BITNET_L3_ACDC)
        add_executable(test_acdc_rect
            ${CMAKE_SOURCE_DIR}/test_acdc_rect.cpp
            ${CMAKE_SOURCE_DIR}/src/ggml-bitnet-fwht.cpp
            ${CMAKE_SOURCE_DIR}/src/ggml-bitnet-common.cpp)
        target_include_directories(test_acdc_rect PRIVATE
            ${CMAKE_SOURCE_DIR}/include)
        target_compile_definitions(test_acdc_rect PRIVATE BITNET_L3_ACDC BITNET_ACDC_RECT)
        bitnet_test_set_simd_flags(test_acdc_rect)
        set_target_properties(test_acdc_rect PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
        add_test(NAME test_acdc_rect COMMAND test_acdc_rect)
        message(STATUS "BitNet: test_acdc_rect ENABLED (D2 gate passed)")
    endif()
else()
    message(STATUS "BitNet: test_acdc_rect DISABLED (D2 gate pending; see T029)")
endif()
