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

template <int N, int K, int BLOCK_SIZE_M, int BLOCK_SIZE_N>
__global__ void int8_int2_gemm_fused_kernel(
    const int8_t *__restrict__ A,
    const int32_t *__restrict__ B_compressed,
    __nv_bfloat16 *__restrict__ C,
    int M,
    const __nv_bfloat16 *__restrict__ s,  // MODIFICATION: s is now bfloat16
    const __nv_bfloat16 *__restrict__ ws) // MODIFICATION: ws is now bfloat16
{
  // --- GEMM Calculation Stage (largely unchanged) ---
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;
  constexpr int BLOCK_SIZE_K = 32;
  constexpr int WARPS_M = 2;
  constexpr int WARPS_N = 2;
  constexpr int M_ITER = BLOCK_SIZE_M / WMMA_M / WARPS_M;
  constexpr int N_ITER = BLOCK_SIZE_N / WMMA_N / WARPS_N;

  const int blockM = blockIdx.y * BLOCK_SIZE_M;
  const int blockN = blockIdx.x * BLOCK_SIZE_N;
  const int warpM = threadIdx.y;
  const int warpN = threadIdx.z;
  const int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

  constexpr int PAD_A = 16;
  constexpr int PAD_B = 16;
  
  __shared__ int8_t shared_A[BLOCK_SIZE_M][BLOCK_SIZE_K + PAD_A];
  __shared__ int8_t shared_B[BLOCK_SIZE_N][BLOCK_SIZE_K + PAD_B];

  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> c_frags[M_ITER][N_ITER];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, nvcuda::wmma::row_major> a_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, nvcuda::wmma::col_major> b_frag;

  #pragma unroll
  for (int m_iter = 0; m_iter < M_ITER; m_iter++) {
    #pragma unroll
    for (int n_iter = 0; n_iter < N_ITER; n_iter++) {
      nvcuda::wmma::fill_fragment(c_frags[m_iter][n_iter], 0);
    }
  }

  const bool m_valid = blockM < M;
  
  for (int k_block = 0; k_block < K; k_block += BLOCK_SIZE_K) {
    __syncthreads();
    // Load A tile
    for (int load_idx = tid; load_idx < (BLOCK_SIZE_M * BLOCK_SIZE_K / 16); load_idx += blockDim.x * blockDim.y * blockDim.z) {
      int local_m = (load_idx * 16) / BLOCK_SIZE_K;
      int local_k = (load_idx * 16) % BLOCK_SIZE_K;
      int global_m = blockM + local_m;
      int global_k = k_block + local_k;
      if (m_valid && global_m < M) {
        *((int4*)&shared_A[local_m][local_k]) = *((int4*)&A[global_m * K + global_k]);
      } else {
        *((int4*)&shared_A[local_m][local_k]) = {0};
      }
    }
    // Load B tile
    int chunk_n = (tid * 16 / BLOCK_SIZE_K);
    int chunk_k = (tid * 16) % BLOCK_SIZE_K;
    if (chunk_n < BLOCK_SIZE_N) {
      int global_n = blockN + chunk_n;
      int global_k = k_block + chunk_k;
      int n_block = global_n / 16;
      int k_block_32 = global_k / 32;
      int k_offset_in_block = chunk_k % 32;
      int in_block_n = chunk_n % 16;
      int compressed_block_idx = n_block * (K / 32) + k_block_32;
      int tile_idx = in_block_n / 8 * 16 + in_block_n % 8 + (k_offset_in_block / 16) * 8;
      int32_t compressed = B_compressed[compressed_block_idx * 32 + tile_idx];
      int8_t decompressed[16];
      decode_i2s_to_i8s(&compressed, decompressed);
      *((int4*)&shared_B[chunk_n][chunk_k]) = *((int4*)decompressed);
    }
    __syncthreads();
    // Perform MMA
    #pragma unroll
    for (int m_iter = 0; m_iter < M_ITER; m_iter++) {
      #pragma unroll
      for (int n_iter = 0; n_iter < N_ITER; n_iter++) {
        #pragma unroll
        for (int wmma_k = 0; wmma_k < BLOCK_SIZE_K; wmma_k += WMMA_K) {
          const int tile_m = (warpM + m_iter * WARPS_M) * WMMA_M;
          const int tile_n = (warpN + n_iter * WARPS_N) * WMMA_N;
          nvcuda::wmma::load_matrix_sync(a_frag, &shared_A[tile_m][wmma_k], BLOCK_SIZE_K + PAD_A);
          nvcuda::wmma::load_matrix_sync(b_frag, &shared_B[tile_n][wmma_k], BLOCK_SIZE_K + PAD_B);
          nvcuda::wmma::mma_sync(c_frags[m_iter][n_iter], a_frag, b_frag, c_frags[m_iter][n_iter]);
        }
      }
    }
  }

  // --- Fused Post-Processing and Store Stage ---
  __shared__ int32_t shared_C[BLOCK_SIZE_M][BLOCK_SIZE_N];

  #pragma unroll
  for (int m_iter = 0; m_iter < M_ITER; m_iter++) {
    #pragma unroll
    for (int n_iter = 0; n_iter < N_ITER; n_iter++) {
      const int tile_m = (warpM + m_iter * WARPS_M) * WMMA_M;
      const int tile_n = (warpN + n_iter * WARPS_N) * WMMA_N;
      nvcuda::wmma::store_matrix_sync(
          &shared_C[tile_m][tile_n],
          c_frags[m_iter][n_iter],
          BLOCK_SIZE_N,
          nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();

  for (int i = tid; i < BLOCK_SIZE_M * BLOCK_SIZE_N; i += blockDim.x * blockDim.y * blockDim.z) {
    const int m = i / BLOCK_SIZE_N;
    const int n = i % BLOCK_SIZE_N;

    const int global_m = blockM + m;
    const int global_n = blockN + n;

    if (global_m < M) {
      int32_t val = shared_C[m][n];
      float float_val = static_cast<float>(val);

      // MODIFICATION: Load bfloat16 scales and convert to float for calculation
      float s_val = __bfloat162float(s[global_m]);
      float_val /= s_val;

      int ws_idx = 0;
      if (N == 3840) {
          ws_idx = global_n / (3840 / 6);
      } else if (N == 13824) {
          ws_idx = global_n / (13824 / 2);
      }
      
      float ws_val = __bfloat162float(ws[ws_idx]);
      float_val *= ws_val;
      
      __nv_bfloat16 bf16_val = __float2bfloat16(float_val);
      C[global_m * N + global_n] = bf16_val;
    }
  }
}

extern "C" void bitlinear_fused_int8xint2(
    int8_t *input0, int8_t *input1,
    __nv_bfloat16 *output0,
    int M, int N, int K,
    __nv_bfloat16 *s,  // MODIFICATION: s is now bfloat16
    __nv_bfloat16 *ws, // MODIFICATION: ws is now bfloat16
    cudaStream_t stream = 0) {
      
  constexpr int BLOCK_SIZE_M = 64;
  constexpr int BLOCK_SIZE_N = 64;

  const dim3 gridDim(N / BLOCK_SIZE_N, (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M, 1);
  const dim3 blockDim(32, 2, 2);

  // Kernel launch now passes the bfloat16 pointers
  if (N == 3840 && K == 2560) {
    int8_int2_gemm_fused_kernel<3840, 2560, BLOCK_SIZE_M, BLOCK_SIZE_N>
        <<<gridDim, blockDim, 0, stream>>>(
            input0, (int32_t *)input1, output0, M, s, ws);
  } else if (N == 2560 && K == 2560) {
    int8_int2_gemm_fused_kernel<2560, 2560, BLOCK_SIZE_M, BLOCK_SIZE_N>
        <<<gridDim, blockDim, 0, stream>>>(
            input0, (int32_t *)input1, output0, M, s, ws);
  } else if (N == 13824 && K == 2560) {
    int8_int2_gemm_fused_kernel<13824, 2560, BLOCK_SIZE_M, BLOCK_SIZE_N>
        <<<gridDim, blockDim, 0, stream>>>(
            input0, (int32_t *)input1, output0, M, s, ws);
  } else if (N == 2560 && K == 6912) {
    int8_int2_gemm_fused_kernel<2560, 6912, BLOCK_SIZE_M, BLOCK_SIZE_N>
        <<<gridDim, blockDim, 0, stream>>>(
            input0, (int32_t *)input1, output0, M, s, ws);
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
  
  cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(launch_error) << std::endl;
    throw std::runtime_error("CUDA kernel launch failed");
  }
}
