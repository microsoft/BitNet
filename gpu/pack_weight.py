import torch
import numpy as np


def B_global_16x32_to_shared_load_16x32_layout(i, j):
    """
         stride * 8 * (tx // HALF_WARP_expr)
                + (tx % 8) * stride
                + 16 * ((tx % HALF_WARP_expr) // 8)
    """
    thread_id = i * 2 + j // 16
    row = (thread_id // 16) * 8 + (thread_id % 8)
    col = (j % 16) + 16 * ((thread_id % 16) // 8)
    return row, col


def permutate_weight_fastest(weight):
    wmma_n = 16
    wmma_k = 32
    N = weight.shape[0]
    K = weight.shape[1]
    
    # Create a lookup table for the permutation
    mapping = np.zeros((wmma_n, wmma_k, 2), dtype=int)
    for ii in range(wmma_n):
        for jj in range(wmma_k):
            mapping[ii, jj] = B_global_16x32_to_shared_load_16x32_layout(ii, jj)
    
    # Reshape weight for the final format
    permutated_weight = np.zeros((N // wmma_n, K // wmma_k, wmma_n, wmma_k), dtype="int8")
    
    # Use advanced indexing for the entire operation
    i_indices = np.arange(N // wmma_n)[:, np.newaxis, np.newaxis, np.newaxis]
    j_indices = np.arange(K // wmma_k)[np.newaxis, :, np.newaxis, np.newaxis]
    
    # Create the source indices
    src_i = i_indices * wmma_n + mapping[:, :, 0]
    src_j = j_indices * wmma_k + mapping[:, :, 1]
    
    # Extract and reshape in one go
    permutated_weight = weight[src_i, src_j]
    
    return permutated_weight


def compress_int2_to_int8(int2_weight):
    int8_weight = np.zeros(
        (*int2_weight.shape[:-1], int2_weight.shape[-1] // 4), dtype=np.int8
    )
    for j in range(int2_weight.shape[-1] // 4):
        for k in range(4):
            int8_weight[:, :, :, j] |= int2_weight[:, :, :, j * 4 + k] << (k * 2)
    return int8_weight


def interleave_weight_int8(qweight, nbits=2):\
    # reinterpret the data type of qweight to int32
    # shift = [ 0,  8, 16, 24,  2, 10, 18, 26,  4, 12, 20, 28,  6, 14, 22, 30]
    # index: [ 0,  4,  8, 12,  1,  5,  9, 13,  2,  6, 10, 14,  3,  7, 11, 15]
    qweight = qweight.view(np.int32)
    new_qweight = np.zeros_like(qweight)
    bits_stride = 8
    mask = (1 << nbits) - 1  # for 4bit the val is 0x0000000f
    num_groups = 32 // bits_stride # 4
    elems_per_group = bits_stride // nbits  # 4
    for i in range(num_groups):
        for j in range(elems_per_group):
            offset = i * elems_per_group + j
            shift = (offset % num_groups) * bits_stride + (offset // num_groups) * nbits

            new_qweight |= ((qweight >> (nbits * offset)) & mask) << shift
    return new_qweight.view(np.int8)



def convert_weight_int8_to_int2(weight):
    N = weight.shape[0]
    K = weight.shape[1]

    weight = weight+2
    
    weight = weight.cpu().numpy()

    # print(weight)
    # print(torch.max(weight), torch.min(weight))

    # permutated_weight_slow = permutate_weight(weight)
    permutated_weight = permutate_weight_fastest(weight)
    # assert np.all(permutated_weight_slow == permutated_weight)
    # print("Permutation is correct")
    compressed_weight = compress_int2_to_int8(permutated_weight)
    interleaved_weight = interleave_weight_int8(compressed_weight, 2)

    ret = torch.from_numpy(interleaved_weight)

    ret = torch.reshape(ret, (N, K // 4))

    return ret
