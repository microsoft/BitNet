# ─── Kernel unit tests for bitnet.cpp ──────────────────────────────────────────
#
# Standalone executables that link directly against the bitnet_math OBJECT
# library (so they exercise the same compiled objects that go into ggml).
# No model needed; runtime < 1ms each.
#
# Enable with -DBITNET_BUILD_TESTS=ON (default ON).

if (NOT BITNET_BUILD_TESTS)
    return()
endif()

if (NOT BITNET_MATH_TARGET)
    message(STATUS "BitNet: tests skipped (no L2-L5 math kernels enabled)")
    return()
endif()

# ─── HRR cleanup_iter (Frady 2021) ────────────────────────────────────────────
# 5/5 PASS: FFT roundtrip, bind vs circular_conv, phasor inverse,
# RESIDUAL Frady 2021, NAIVE projection.
#
# This test exercises L5 in isolation. We compile only ggml-bitnet-hrr.cpp
# (not the full bitnet_math OBJECT lib) to avoid pulling in ggml-bitnet-dispatch.cpp
# which references ggml symbols not available outside the llama.cpp build.
if (BITNET_L5_HRR)
    add_executable(test_hrr_cleanup
        ${CMAKE_SOURCE_DIR}/test_hrr_cleanup.cpp
        ${CMAKE_SOURCE_DIR}/src/ggml-bitnet-hrr.cpp)
    target_include_directories(test_hrr_cleanup PRIVATE
        ${CMAKE_SOURCE_DIR}/include)
    target_compile_definitions(test_hrr_cleanup PRIVATE BITNET_L5_HRR)
    # Per-arch SIMD flags mirroring src/CMakeLists.txt for the kernel.
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i686")
        target_compile_options(test_hrr_cleanup PRIVATE -mavx2 -mfma)
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        target_compile_options(test_hrr_cleanup PRIVATE -march=armv8-a+simd)
    endif()
    if (UNIX AND NOT APPLE)
        target_link_libraries(test_hrr_cleanup PRIVATE m)
    endif()
    set_target_properties(test_hrr_cleanup PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
    add_test(NAME test_hrr_cleanup COMMAND test_hrr_cleanup)
endif()
