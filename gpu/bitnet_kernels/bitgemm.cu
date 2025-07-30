#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <time.h>

template <typename T1, typename T2>
__device__ void decode_i2s_to_i8s(T1 *_i2s, T2 *_i8s, const int N = 16) {
  // convert 8 int2b_t to 8 int8b_t -> 2 int32
  uint *i8s = reinterpret_cast<uint *>(_i8s);

  // i2s = {e7,e6,e5,e4,e3,e2,e1,e0}
  // also require interleave {e7,e3,e6,e2,e5,e1,e4,e0}
  uint const i2s = *_i2s;

  // First, we extract the i4s and construct an intermediate fp16 number.
  static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa; // 0b11101010
  static constexpr uint BOTTOM_MASK = 0x03030303;      // 0xf -> 0b11 select 0,3
  static constexpr uint I4s_TO_I8s_MAGIC_NUM = 0x00000000; // 1024

#pragma unroll
  for (int i = 0; i < (N / 4); i++) {
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(i8s[i])
                 : "r"(i2s >> (2 * i)), "n"(BOTTOM_MASK),
                   "n"(I4s_TO_I8s_MAGIC_NUM), "n"(immLut));
    i8s[i] = __vsubss4(i8s[i], 0x02020202);
  }
}


template <int N, int K>
__global__ void int8_int2_gemm_tensor_core(
    const int8_t *__restrict__ A,             // M x K matrix, row-major
    const int32_t *__restrict__ B_compressed, // Compressed int2 data for N x K matrix, column-major
    int32_t *__restrict__ C,                  // M x N output matrix, row-major
    int M)
{
  // Define WMMA dimensions - all constant
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;
  
  // Define block tile dimensions - all constant
  constexpr int BLOCK_SIZE_M = 64; // Multiple of WMMA_M
  constexpr int BLOCK_SIZE_N = 64; // Multiple of WMMA_N
  constexpr int BLOCK_SIZE_K = 32; // K dimension as requested

  // Calculate thread block position
  const int blockM = blockIdx.y * BLOCK_SIZE_M;
  const int blockN = blockIdx.x * BLOCK_SIZE_N;

  // Calculate thread ID and warp IDs
  const int warpM = threadIdx.y; // 0-1 (2 warps in M dimension)
  const int warpN = threadIdx.z; // 0-1 (2 warps in N dimension)
  const int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

  // Add padding to shared memory to avoid bank conflicts
  constexpr int PAD_A = 16; // Padding for A matrix
  constexpr int PAD_B = 16; // Padding for B matrix
  
  // Allocate shared memory for A and B matrices with padding
  __shared__ int8_t shared_A[BLOCK_SIZE_M][BLOCK_SIZE_K + PAD_A];
  __shared__ int8_t shared_B[BLOCK_SIZE_N][BLOCK_SIZE_K + PAD_B];

  // Define fragments for all tiles this warp will handle - static allocation
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> c_frags[2][2];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, nvcuda::wmma::row_major> a_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, nvcuda::wmma::col_major> b_frag;

  // Initialize all accumulator fragments to zero (unrolled)
  #pragma unroll
  for (int m_iter = 0; m_iter < 2; m_iter++) {
    #pragma unroll
    for (int n_iter = 0; n_iter < 2; n_iter++) {
      nvcuda::wmma::fill_fragment(c_frags[m_iter][n_iter], 0);
    }
  }

  // Only check M bounds at the beginning
  const bool m_valid = blockM < M;
  
  // Loop over K dimension in chunks of BLOCK_SIZE_K
  #pragma unroll 4 // Partial unroll of K-dimension loop
  for (int k_block = 0; k_block < K; k_block += BLOCK_SIZE_K) {
    // Clear shared memory first
    __syncthreads();

    // Load A matrix tiles into shared memory using vectorized loads
    // Each thread handles multiple elements based on its ID
    for (int load_idx = tid; load_idx < (BLOCK_SIZE_M * BLOCK_SIZE_K / 16); load_idx += blockDim.x * blockDim.y * blockDim.z) {
      int local_m = (load_idx * 16) / BLOCK_SIZE_K;
      int local_k = (load_idx * 16) % BLOCK_SIZE_K;
      
      int global_m = blockM + local_m;
      int global_k = k_block + local_k;
      
      // Use vector loads for A - 16 bytes at a time (int4 = 4 integers = 16 bytes)
      if (m_valid && global_m < M) {
        // Vector load from A to shared memory
        *((int4*)&shared_A[local_m][local_k]) = *((int4*)&A[global_m * K + global_k]);
      } else {
        // Zero out if M is out of bounds
        *((int4*)&shared_A[local_m][local_k]) = {0};
      }
    }

    // Load B matrix tiles into shared memory (always in bounds for N and K)
    // Calculate which 16-element chunk this thread is responsible for
    int chunk_n = (tid * 16 / BLOCK_SIZE_K);
    int chunk_k = (tid * 16) % BLOCK_SIZE_K;

    if (chunk_n < BLOCK_SIZE_N) {
      int global_n = blockN + chunk_n;
      int global_k = k_block + chunk_k;

      // Calculate which compressed block this belongs to
      int n_block = global_n / 16;
      int k_block_32 = global_k / 32;
      int k_offset_in_block = chunk_k % 32;

      // Get the specific compressed tile within the 16x32 block
      int in_block_n = chunk_n % 16;
      int compressed_block_idx = n_block * (K / 32) + k_block_32;

      // Calculate which tile within the compressed block
      int tile_idx;
      tile_idx = in_block_n / 8 * 16 + in_block_n % 8 + (k_offset_in_block / 16) * 8;

      // Extract and decompress the int2 values
      int32_t compressed = B_compressed[compressed_block_idx * 32 + tile_idx];
      int8_t decompressed[16];
      decode_i2s_to_i8s(&compressed, decompressed);

      // Vector store to shared memory
      *((int4*)&shared_B[chunk_n][chunk_k]) = *((int4*)decompressed);
    }

    // Make sure all threads have finished loading into shared memory
    __syncthreads();

    // Process the 2x2 WMMA tiles for this K block
    #pragma unroll
    for (int m_iter = 0; m_iter < 2; m_iter++) {
      #pragma unroll
      for (int n_iter = 0; n_iter < 2; n_iter++) {
        // Calculate the starting positions for this WMMA tile
        #pragma unroll
        for (int wmma_k = 0; wmma_k < BLOCK_SIZE_K; wmma_k += WMMA_K) {
            // Fully unroll the m and n iterations
          const int tile_m = (warpM * 2 + m_iter) * WMMA_M;
          const int tile_n = (warpN * 2 + n_iter) * WMMA_N;

          // Load matrix A fragment from shared memory with padding
          nvcuda::wmma::load_matrix_sync(
              a_frag, &shared_A[tile_m][wmma_k], BLOCK_SIZE_K + PAD_A);
          
          // Load matrix B fragment from shared memory with padding
          nvcuda::wmma::load_matrix_sync(
              b_frag, &shared_B[tile_n][wmma_k], BLOCK_SIZE_K + PAD_B);

          // Perform matrix multiplication
          nvcuda::wmma::mma_sync(c_frags[m_iter][n_iter], a_frag, b_frag, c_frags[m_iter][n_iter]);
        }
      }
    }
  }

  // Store results back to global memory - only check M bounds
  #pragma unroll
  for (int m_iter = 0; m_iter < 2; m_iter++) {
    const int tile_m = (warpM * 2 + m_iter) * WMMA_M;
    const int global_tile_m = blockM + tile_m;
    
    if (m_valid && global_tile_m < M) {
      #pragma unroll
      for (int n_iter = 0; n_iter < 2; n_iter++) {
        const int tile_n = (warpN * 2 + n_iter) * WMMA_N;
        const int global_tile_n = blockN + tile_n;
        
        // No need to check N bounds as it's always aligned
        nvcuda::wmma::store_matrix_sync(
            &C[global_tile_m * N + global_tile_n],
            c_frags[m_iter][n_iter], N, nvcuda::wmma::mem_row_major);
      }
    }
  }
}

extern "C" void bitlinear_int8xint2(int8_t *input0, int8_t *input1,
                                    int32_t *output0, int M, int N, int K,
                                    cudaStream_t stream = 0) {
  if (N == 3840 && K == 2560) {
    int8_int2_gemm_tensor_core<3840, 2560>
        <<<dim3(60, (M + 63) / 64, 1), dim3(32, 2, 2), 0, stream>>>(
            input0, (int32_t *)input1, (int32_t *)output0, M);
  } else if (N == 2560 && K == 2560) {
    int8_int2_gemm_tensor_core<2560, 2560>
        <<<dim3(40, (M + 63) / 64, 1), dim3(32, 2, 2), 0, stream>>>(
            input0, (int32_t *)input1, (int32_t *)output0, M);
  } else if (N == 13824 && K == 2560) {
    int8_int2_gemm_tensor_core<13824, 2560>
        <<<dim3(216, (M + 63) / 64, 1), dim3(32, 2, 2), 0, stream>>>(
            input0, (int32_t *)input1, (int32_t *)output0, M);
  } else if (N == 2560 && K == 6912) {
    int8_int2_gemm_tensor_core<2560, 6912>
        <<<dim3(40, (M + 63) / 64, 1), dim3(32, 2, 2), 0, stream>>>(
            input0, (int32_t *)input1, (int32_t *)output0, M);
  } else {
    std::cerr << "Error: Unsupported matrix dimensions for bitlinear_int8xint2. "
              << "Required kernel: M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cerr << "Supported configurations:" << std::endl;
    std::cerr << "  - N=3840, K=2560" << std::endl;
    std::cerr << "  - N=2560, K=2560" << std::endl;
    std::cerr << "  - N=13824, K=2560" << std::endl;
    std::cerr << "  - N=2560, K=6912" << std::endl;
    throw std::runtime_error("Unsupported matrix dimensions for bitlinear_int8xint2");
  }
  
  // Check for CUDA launch errors
  cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(launch_error) << std::endl;
    throw std::runtime_error("CUDA kernel launch failed");
  }
  
  // Synchronize and check for execution errors
  cudaError_t sync_error = cudaStreamSynchronize(stream);
  if (sync_error != cudaSuccess) {
    std::cerr << "CUDA kernel execution failed: " << cudaGetErrorString(sync_error) << std::endl;
    throw std::runtime_error("CUDA kernel execution failed");
  }
}