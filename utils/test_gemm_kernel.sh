#!/bin/bash
# Unified GEMM kernel benchmark script
# Builds, tests, and benchmarks the GEMM kernel with configurable output

set -e

# Default values
BUILD_DIR="../build"
ITERATIONS=1000
OUTPUT_CSV=""
SKIP_BUILD=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Print usage
print_usage() {
    cat << EOF
Usage: $0 [options]

Options:
  -o, --output <path>     Output CSV file path (default: ../stats/gemm_kernel_test_noparal.csv)
  -i, --iterations <num>  Number of iterations per test (default: 1000)
  -s, --skip-build        Skip building the benchmark binary
  -h, --help              Show this help message

Examples:
  # Run with default settings
  $0

  # Specify custom output file
  $0 -o /path/to/my_results.csv

  # Quick test with fewer iterations
  $0 -i 100 -o quick_test.csv

  # Skip build if already compiled
  $0 -s -o results.csv
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output)
            OUTPUT_CSV="$2"
            shift 2
            ;;
        -i|--iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        -s|--skip-build)
            SKIP_BUILD=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Set default output CSV if not specified
if [ -z "$OUTPUT_CSV" ]; then
    OUTPUT_CSV="${SCRIPT_DIR}/../stats/gemm_kernel_test_noparal.csv"
fi

# Create output directory first
mkdir -p "$(dirname "$OUTPUT_CSV")"

# Convert to absolute path
if [[ "$OUTPUT_CSV" = /* ]]; then
    # Already absolute path
    OUTPUT_CSV="$OUTPUT_CSV"
else
    # Convert relative path to absolute
    OUTPUT_CSV="$(cd "$(dirname "$OUTPUT_CSV")" && pwd)/$(basename "$OUTPUT_CSV")"
fi

echo "=========================================="
echo "GEMM Kernel Benchmark Suite"
echo "=========================================="
echo "Configuration:"
echo "  Iterations: $ITERATIONS"
echo "  Output CSV: $OUTPUT_CSV"
echo "  Skip build: $SKIP_BUILD"
echo "=========================================="
echo ""

# Build the benchmark binary
if [ "$SKIP_BUILD" = false ]; then
    echo "Step 1: Building GEMM kernel benchmark..."
    echo "------------------------------------------"
    
    CXX=${CXX:-g++}
    
    # Create build directory if it doesn't exist
    mkdir -p "${SCRIPT_DIR}/${BUILD_DIR}"
    
    # Create temporary C++ source file
    TEMP_CPP="${SCRIPT_DIR}/${BUILD_DIR}/test_gemm_kernel_temp.cpp"
    
    cat > "${TEMP_CPP}" << 'EOF'
/**
 * Standalone benchmark for ggml_gemm_i2_i8_s kernel
 * 
 * This program tests the performance of the ggml_gemm_i2_i8_s kernel
 * with configurable matrix sizes and iteration counts.
 * 
 * Usage: ./test_gemm_kernel [options]
 *   -n <size>   : embedding dimension (must be divisible by 4, default: 2048)
 *   -r <rows>   : number of rows in matrix Y (default: 32)
 *   -c <cols>   : number of columns in matrix X (default: 128)
 *   -i <iters>  : number of iterations (default: 1000)
 *   -w <warmup> : number of warmup iterations (default: 10)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>

// Include necessary headers
#include "../include/gemm-config.h"

// Function declarations (from ggml-quants.h)
extern "C" void ggml_vec_dot_i2_i8_s(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);

// GEMM kernel definition
void ggml_gemm_i2_i8_s(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
#if defined(ACT_PARALLEL)
    const int64_t row_block = ROW_BLOCK_SIZE;
    const int64_t col_block = COL_BLOCK_SIZE;

    for (int64_t c0 = 0; c0 < nc; c0 += col_block) {
        int64_t cur_c = (c0 + col_block <= nc) ? col_block : (nc - c0);
        for (int64_t r0 = 0; r0 < nr; r0 += row_block) {
            int64_t cur_r = (r0 + row_block <= nr) ? row_block : (nr - r0);
            const void * vy_r = (const uint8_t *)vy + r0 * n;
            for (int64_t c = 0; c < cur_c; ++c) {
                const int64_t col = c0 + c;
                float * s_col = s + col;
                const void * vx_col = (const uint8_t *)vx + col * n / 4;
                ggml_vec_dot_i2_i8_s(n, s_col + r0 * bs, bs, vx_col, n, vy_r, n, cur_r);
            }
        }
    }
#else
    const int64_t row_block = ROW_BLOCK_SIZE;
    const int64_t col_block = COL_BLOCK_SIZE;

    for (int64_t r0 = 0; r0 < nr; r0 += row_block) {
        int64_t cur_r = (r0 + row_block <= nr) ? row_block : (nr - r0);
        for (int64_t c0 = 0; c0 < nc; c0 += col_block) {
            int64_t cur_c = (c0 + col_block <= nc) ? col_block : (nc - c0);
            const void * vx_c = (const uint8_t *)vx + c0 * n / 4;
            for (int64_t r = 0; r < cur_r; ++r) {
                const int64_t row = r0 + r;
                float * s_row = s + row * bs;
                const void * vy_row = (const uint8_t *)vy + row * n;
                ggml_vec_dot_i2_i8_s(n, s_row + c0, bs, vx_c, n, vy_row, n, cur_c);
            }
        }
    }
#endif
}

// Helper function to get current time in nanoseconds
double get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

// Initialize matrix with random i2 values (2-bit quantized)
void init_matrix_i2(uint8_t* data, int n, int cols) {
    // i2 format: 4 values per byte (2 bits each)
    int total_bytes = n * cols / 4;
    for (int i = 0; i < total_bytes; i++) {
        data[i] = rand() & 0xFF;
    }
}

// Initialize matrix with random i8 values
void init_matrix_i8(int8_t* data, int n, int rows) {
    int total_elements = n * rows;
    for (int i = 0; i < total_elements; i++) {
        data[i] = (int8_t)((rand() % 256) - 128);
    }
}

// Benchmark configuration
struct BenchmarkConfig {
    int n;           // embedding dimension (must be divisible by 4)
    int nr;          // number of rows in Y matrix
    int nc;          // number of columns in X matrix
    int iterations;  // number of benchmark iterations
    int warmup;      // number of warmup iterations
};

void print_config(const BenchmarkConfig& config) {
    printf("=" "=%.78s\n", "===============================================================================");
    printf("Benchmark Configuration:\n");
    printf("=" "=%.78s\n", "===============================================================================");
    printf("  Embedding dimension (n)    : %d\n", config.n);
    printf("  Matrix Y rows (nr)         : %d\n", config.nr);
    printf("  Matrix X columns (nc)      : %d\n", config.nc);
    printf("  Iterations                 : %d\n", config.iterations);
    printf("  Warmup iterations          : %d\n", config.warmup);
    printf("\nMatrix sizes:\n");
    printf("  X (i2): %d x %d (%.2f KB)\n", config.nc, config.n, 
           (config.nc * config.n / 4) / 1024.0);
    printf("  Y (i8): %d x %d (%.2f KB)\n", config.nr, config.n,
           (config.nr * config.n) / 1024.0);
    printf("  S (f32): %d x %d (%.2f KB)\n", config.nr, config.nc,
           (config.nr * config.nc * sizeof(float)) / 1024.0);
    printf("\nGEMM Config:\n");
#if defined(ACT_PARALLEL)
    printf("  ACT_PARALLEL              : ON\n");
#else
    printf("  ACT_PARALLEL              : OFF\n");
#endif
    printf("  ROW_BLOCK_SIZE            : %d\n", ROW_BLOCK_SIZE);
    printf("  COL_BLOCK_SIZE            : %d\n", COL_BLOCK_SIZE);
    printf("  PARALLEL_SIZE             : %d\n", PARALLEL_SIZE);
    printf("=" "=%.78s\n\n", "===============================================================================");
}

void run_benchmark(const BenchmarkConfig& config) {
    // Allocate matrices
    printf("Allocating matrices...\n");
    
    // X matrix (i2 format): nc x n, but stored as nc x (n/4) bytes
    // Align to 64 bytes for AVX-512, which is backward compatible with AVX2 (32 bytes)
    size_t x_size = config.nc * config.n / 4;
    size_t x_size_aligned = ((x_size + 63) / 64) * 64;
    uint8_t* X = (uint8_t*)aligned_alloc(64, x_size_aligned);
    
    // Y matrix (i8 format): nr x n
    size_t y_size = config.nr * config.n;
    size_t y_size_aligned = ((y_size + 63) / 64) * 64;
    int8_t* Y = (int8_t*)aligned_alloc(64, y_size_aligned);
    
    // Result matrix (float32): nr x nc
    size_t s_size = config.nr * config.nc * sizeof(float);
    size_t s_size_aligned = ((s_size + 63) / 64) * 64;
    float* S = (float*)aligned_alloc(64, s_size_aligned);
    
    if (!X || !Y || !S) {
        fprintf(stderr, "Failed to allocate memory\n");
        exit(1);
    }
    
    // Initialize matrices with random data
    printf("Initializing matrices with random data...\n");
    srand(time(NULL));
    init_matrix_i2(X, config.n, config.nc);
    init_matrix_i8(Y, config.n, config.nr);
    memset(S, 0, config.nr * config.nc * sizeof(float));
    
    // Warmup
    printf("Running %d warmup iterations...\n", config.warmup);
    for (int i = 0; i < config.warmup; i++) {
        ggml_gemm_i2_i8_s(config.n, S, config.nc, X, Y, config.nr, config.nc);
    }
    
    // Benchmark
    printf("Running %d benchmark iterations...\n", config.iterations);
    double total_time = 0.0;
    double min_time = 1e20;
    double max_time = 0.0;
    
    for (int i = 0; i < config.iterations; i++) {
        double start = get_time_ns();
        ggml_gemm_i2_i8_s(config.n, S, config.nc, X, Y, config.nr, config.nc);
        double end = get_time_ns();
        
        double elapsed = end - start;
        total_time += elapsed;
        if (elapsed < min_time) min_time = elapsed;
        if (elapsed > max_time) max_time = elapsed;
        
        if ((i + 1) % 100 == 0) {
            printf("  Progress: %d/%d iterations\n", i + 1, config.iterations);
        }
    }
    
    // Calculate statistics
    double avg_time_ns = total_time / config.iterations;
    double avg_time_ms = avg_time_ns / 1e6;
    double min_time_ms = min_time / 1e6;
    double max_time_ms = max_time / 1e6;
    
    // Calculate GFLOPS
    // For GEMM: nr x nc x n multiply-adds = 2 * nr * nc * n FLOPs
    double flops = 2.0 * config.nr * config.nc * config.n;
    double gflops = (flops / avg_time_ns);
    
    // Calculate throughput (tokens/s assuming each column is a token)
    double throughput = (config.nc * 1e9) / avg_time_ns;
    
    // Print results
    printf("\n");
    printf("=" "=%.78s\n", "===============================================================================");
    printf("Benchmark Results:\n");
    printf("=" "=%.78s\n", "===============================================================================");
    printf("  Average time  : %.3f ms\n", avg_time_ms);
    printf("  Min time      : %.3f ms\n", min_time_ms);
    printf("  Max time      : %.3f ms\n", max_time_ms);
    printf("  Std dev       : %.3f ms\n", sqrt((max_time_ms - min_time_ms) * (max_time_ms - min_time_ms) / 12));
    printf("\nPerformance:\n");
    printf("  GFLOPS        : %.2f\n", gflops);
    printf("  Throughput    : %.2f tokens/s\n", throughput);
    printf("  Latency/token : %.3f us\n", (avg_time_ms * 1000) / config.nc);
    printf("=" "=%.78s\n", "===============================================================================");
    
    // Cleanup
    free(X);
    free(Y);
    free(S);
}

void print_usage(const char* program) {
    printf("Usage: %s [options]\n", program);
    printf("Options:\n");
    printf("  -n <size>    Embedding dimension (must be divisible by 4, default: 2048)\n");
    printf("  -r <rows>    Number of rows in matrix Y (default: 32)\n");
    printf("  -c <cols>    Number of columns in matrix X (default: 128)\n");
    printf("  -i <iters>   Number of iterations (default: 1000)\n");
    printf("  -w <warmup>  Number of warmup iterations (default: 10)\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    BenchmarkConfig config = {
        .n = 2048,
        .nr = 32,
        .nc = 128,
        .iterations = 1000,
        .warmup = 10
    };
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            config.n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            config.nr = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            config.nc = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            config.iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-w") == 0 && i + 1 < argc) {
            config.warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // Validate configuration
    if (config.n % 4 != 0) {
        fprintf(stderr, "Error: Embedding dimension (-n) must be divisible by 4\n");
        return 1;
    }
    
    if (config.n <= 0 || config.nr <= 0 || config.nc <= 0 || config.iterations <= 0) {
        fprintf(stderr, "Error: All size parameters must be positive\n");
        return 1;
    }
    
    // Run benchmark
    print_config(config);
    run_benchmark(config);
    
    return 0;
}
EOF
    
    # Compiler flags
    CXXFLAGS="-O3 -march=native -mtune=native -std=c++17 -fopenmp"
    CXXFLAGS+=" -I${SCRIPT_DIR}/.. -I${SCRIPT_DIR}/../include"
    CXXFLAGS+=" -I${SCRIPT_DIR}/../3rdparty/llama.cpp/ggml/include"
    CXXFLAGS+=" -I${SCRIPT_DIR}/../3rdparty/llama.cpp/ggml/src"
    CXXFLAGS+=" -I${SCRIPT_DIR}/../3rdparty/llama.cpp/include"
    CXXFLAGS+=" -DNDEBUG -ffast-math"
    
    # Link flags
    LDFLAGS="-lm -lpthread"
    
    # Link with pre-built libraries
    GGML_LIB_DIR="${SCRIPT_DIR}/../build/3rdparty/llama.cpp/ggml/src"
    GGML_SO="${GGML_LIB_DIR}/libggml.so"
    
    if [ ! -f "${GGML_SO}" ]; then
        echo "❌ Error: Cannot find libggml.so at ${GGML_SO}"
        echo "Please build the project first with: cmake --build build"
        rm -f "${TEMP_CPP}"
        exit 1
    fi
    
    LDFLAGS+=" -L${GGML_LIB_DIR} -lggml -Wl,-rpath,${GGML_LIB_DIR}"
    
    # Output binary
    BENCHMARK_BIN="${SCRIPT_DIR}/${BUILD_DIR}/test_gemm_kernel"
    
    echo "Compiler: ${CXX}"
    echo "Building from embedded source..."
    echo ""
    
    # Build
    ${CXX} ${CXXFLAGS} "${TEMP_CPP}" -o ${BENCHMARK_BIN} ${LDFLAGS}
    
    if [ $? -eq 0 ]; then
        echo "✅ Build successful!"
        rm -f "${TEMP_CPP}"
        echo ""
    else
        echo "❌ Build failed!"
        rm -f "${TEMP_CPP}"
        exit 1
    fi
else
    echo "Step 1: Skipping build (using existing binary)"
    echo "------------------------------------------"
    BENCHMARK_BIN="${SCRIPT_DIR}/${BUILD_DIR}/test_gemm_kernel"
    
    if [ ! -f "${BENCHMARK_BIN}" ]; then
        echo "❌ Error: Benchmark binary not found at ${BENCHMARK_BIN}"
        echo "Please run without -s to build it first."
        exit 1
    fi
    echo "✅ Found existing binary"
    echo ""
fi

# Set LD_LIBRARY_PATH to include the GGML library directory
GGML_LIB_DIR="${SCRIPT_DIR}/../build/3rdparty/llama.cpp/ggml/src"
export LD_LIBRARY_PATH="${GGML_LIB_DIR}:${LD_LIBRARY_PATH}"

echo "Step 2: Running benchmark tests"
echo "------------------------------------------"
echo "Library path: ${GGML_LIB_DIR}"
echo ""

# Write CSV header
echo "test_name,n,nr,nc,time_ms,gflops,throughput_tokens_per_sec" > "$OUTPUT_CSV"
echo "Results will be saved to: $OUTPUT_CSV"
echo ""

# Function to extract metrics and append to CSV
extract_and_save() {
    local test_name="$1"
    local output="$2"
    
    # Extract values using grep and awk
    local n=$(echo "$output" | grep "Embedding dimension" | awk '{print $5}')
    local nr=$(echo "$output" | grep "Matrix Y rows" | awk '{print $6}')
    local nc=$(echo "$output" | grep "Matrix X columns" | awk '{print $6}')
    local avg_time=$(echo "$output" | grep "Average time" | awk '{print $4}')
    local min_time=$(echo "$output" | grep "Min time" | awk '{print $4}')
    local max_time=$(echo "$output" | grep "Max time" | awk '{print $4}')
    local gflops=$(echo "$output" | grep "GFLOPS" | awk '{print $3}')
    local throughput=$(echo "$output" | grep "Throughput" | awk '{print $3}')
    
    # Check if values were extracted successfully
    if [ -z "$avg_time" ] || [ -z "$min_time" ] || [ -z "$max_time" ]; then
        echo "Warning: Failed to extract timing data for ${test_name}"
        echo "${test_name},${n},${nr},${nc},N/A,N/A,N/A" >> "$OUTPUT_CSV"
        return
    fi
    
    # Calculate standard deviation estimate from range
    # Using awk with proper variable passing
    local std_time=$(awk -v min="$min_time" -v max="$max_time" 'BEGIN {printf "%.4f", (max - min) / 4}')
    
    # Format as mean±std
    local time_formatted="${avg_time}±${std_time}"
    
    # Append to CSV
    echo "${test_name},${n},${nr},${nc},${time_formatted},${gflops},${throughput}" >> "$OUTPUT_CSV"
}

# Run benchmark tests
echo "=========================================="
echo "BitNet-2B Typical Shapes Performance Test"
echo "=========================================="
echo ""

echo "Test 1: Single Token Generation (Attention QKV projection)"
echo "  Scenario: Generating 1 token at a time"
echo "  Shape: n=2048, r=1, c=2048"
OUTPUT=$($BENCHMARK_BIN -n 2048 -r 1 -c 2048 -i $ITERATIONS 2>&1)
echo "$OUTPUT"
extract_and_save "single_token_gen" "$OUTPUT"
echo ""

echo "Test 2: Small Batch Prompt Processing (Attention QKV projection)"
echo "  Scenario: Processing prompt with 128 tokens, batch size 1"
echo "  Shape: n=2048, r=128, c=2048"
OUTPUT=$($BENCHMARK_BIN -n 2048 -r 128 -c 2048 -i $ITERATIONS 2>&1)
echo "$OUTPUT"
extract_and_save "small_batch_prompt" "$OUTPUT"
echo ""

echo "Test 3: Medium Batch Prompt Processing (Attention QKV projection)"
echo "  Scenario: Processing prompt with 256 tokens or batch of 256"
echo "  Shape: n=2048, r=256, c=2048"
OUTPUT=$($BENCHMARK_BIN -n 2048 -r 256 -c 2048 -i $ITERATIONS 2>&1)
echo "$OUTPUT"
extract_and_save "medium_batch_prompt" "$OUTPUT"
echo ""

echo "Test 4: Large Batch Processing (Attention QKV projection)"
echo "  Scenario: Processing 512 tokens or batch of 512"
echo "  Shape: n=2048, r=512, c=2048"
OUTPUT=$($BENCHMARK_BIN -n 2048 -r 512 -c 2048 -i $ITERATIONS 2>&1)
echo "$OUTPUT"
extract_and_save "large_batch_prompt" "$OUTPUT"
echo ""

echo "Test 5: FFN Up-projection (Small batch)"
echo "  Scenario: Feed-forward network expansion, 128 tokens"
echo "  Shape: n=2048, r=128, c=8192"
OUTPUT=$($BENCHMARK_BIN -n 2048 -r 128 -c 8192 -i $ITERATIONS 2>&1)
echo "$OUTPUT"
extract_and_save "ffn_up_projection" "$OUTPUT"
echo ""

echo "Test 6: FFN Down-projection (Small batch)"
echo "  Scenario: Feed-forward network reduction, 128 tokens"
echo "  Shape: n=8192, r=128, c=2048"
OUTPUT=$($BENCHMARK_BIN -n 8192 -r 128 -c 2048 -i $ITERATIONS 2>&1)
echo "$OUTPUT"
extract_and_save "ffn_down_projection" "$OUTPUT"
echo ""

echo "Test 7: Long Context Processing"
echo "  Scenario: Processing very long context (2048 tokens)"
echo "  Shape: n=2048, r=2048, c=2048"
OUTPUT=$($BENCHMARK_BIN -n 2048 -r 2048 -c 2048 -i $ITERATIONS 2>&1)
echo "$OUTPUT"
extract_and_save "long_context" "$OUTPUT"
echo ""

echo "Test 8: Batched Token Generation"
echo "  Scenario: Generating tokens for 32 sequences simultaneously"
echo "  Shape: n=2048, r=32, c=2048"
OUTPUT=$($BENCHMARK_BIN -n 2048 -r 32 -c 2048 -i $ITERATIONS 2>&1)
echo "$OUTPUT"
extract_and_save "batched_token_gen" "$OUTPUT"
echo ""

echo "=========================================="
echo "All tests completed successfully!"
echo "=========================================="
echo "Results saved to: $OUTPUT_CSV"
echo ""
echo "Summary:"
wc -l "$OUTPUT_CSV" | awk '{print "  Total records:", $1 - 1}'
echo "  Output file: $OUTPUT_CSV"
echo "=========================================="
