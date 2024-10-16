import argparse
import os
from configparser import ConfigParser

def gen_ctor_code():
    kernel_code = "\n\
#include \"ggml-bitnet.h\"\n\
#define GGML_BITNET_MAX_NODES 8192\n\
static bool initialized = false;\n\
static bitnet_tensor_extra * bitnet_tensor_extras = nullptr;\n\
static size_t bitnet_tensor_extras_index = 0;\n\
static void * aligned_malloc(size_t size) {{\n\
#if defined(_WIN32)\n\
    return _aligned_malloc(size, 64);\n\
#else\n\
    void * ptr = nullptr;\n\
    posix_memalign(&ptr, 64, size);\n\
    return ptr;\n\
#endif\n\
}}\n\
static void aligned_free(void * ptr) {{\n\
#if defined(_WIN32)\n\
    _aligned_free(ptr);\n\
#else\n\
    free(ptr);\n\
#endif\n\
}}\n\
\n\
void per_tensor_quant(int k, void* lut_scales_, void* b_) {{\n\
    bitnet_float_type* lut_scales = (bitnet_float_type*)lut_scales_;\n\
    bitnet_float_type* b = (bitnet_float_type*)b_;\n\
#ifdef __ARM_NEON\n\
    float32x4_t temp_max = vdupq_n_f32(0);\n\
    for (int i=0; i < k / 4; i++) {{\n\
      float32x4_t vec_bs = vld1q_f32(b + 4 * i);\n\
      float32x4_t abssum = vabsq_f32(vec_bs);\n\
      temp_max = vmaxq_f32(abssum, temp_max);\n\
    }}\n\
    float32_t scales = 127 / vmaxvq_f32(temp_max);\n\
    *lut_scales = scales;\n\
#elif defined __AVX2__\n\
    __m256 max_vec = _mm256_set1_ps(0.f);\n\
    const __m256 vec_sign = _mm256_set1_ps(-0.0f);\n\
    // #pragma unroll\n\
    for (int i = 0; i < k / 8; i++) {{\n\
        __m256 vec_b = _mm256_loadu_ps(b + i * 8);\n\
        __m256 vec_babs = _mm256_andnot_ps(vec_sign, vec_b);\n\
        max_vec = _mm256_max_ps(vec_babs, max_vec);\n\
    }}\n\
    __m128 max1 = _mm_max_ps(_mm256_extractf128_ps(max_vec, 1), _mm256_castps256_ps128(max_vec));\n\
    max1 = _mm_max_ps(max1, _mm_movehl_ps(max1, max1));\n\
    max1 = _mm_max_ss(max1, _mm_movehdup_ps(max1));\n\
    float scales = 127 / _mm_cvtss_f32(max1);\n\
    *lut_scales = scales;\n\
#endif\n\
}}\n\
\n\
void partial_max_reset(void* lut_scales_) {{\n\
    bitnet_float_type* lut_scales = (bitnet_float_type*)lut_scales_;\n\
    *lut_scales = 0.0;\n\
}}\n\
\n\
#ifdef __ARM_NEON\n\
inline void Transpose_8_8(\n\
    int16x8_t *v0,\n\
    int16x8_t *v1,\n\
    int16x8_t *v2,\n\
    int16x8_t *v3,\n\
    int16x8_t *v4,\n\
    int16x8_t *v5,\n\
    int16x8_t *v6,\n\
    int16x8_t *v7)\n\
{{\n\
    int16x8x2_t q04 = vzipq_s16(*v0, *v4);\n\
    int16x8x2_t q15 = vzipq_s16(*v1, *v5);\n\
    int16x8x2_t q26 = vzipq_s16(*v2, *v6);\n\
    int16x8x2_t q37 = vzipq_s16(*v3, *v7);\n\
\n\
    int16x8x2_t q0246_0 = vzipq_s16(q04.val[0], q26.val[0]);\n\
    int16x8x2_t q0246_1 = vzipq_s16(q04.val[1], q26.val[1]);\n\
    int16x8x2_t q1357_0 = vzipq_s16(q15.val[0], q37.val[0]);\n\
    int16x8x2_t q1357_1 = vzipq_s16(q15.val[1], q37.val[1]);\n\
\n\
    int16x8x2_t q_fin_0 = vzipq_s16(q0246_0.val[0], q1357_0.val[0]);\n\
    int16x8x2_t q_fin_1 = vzipq_s16(q0246_0.val[1], q1357_0.val[1]);\n\
    int16x8x2_t q_fin_2 = vzipq_s16(q0246_1.val[0], q1357_1.val[0]);\n\
    int16x8x2_t q_fin_3 = vzipq_s16(q0246_1.val[1], q1357_1.val[1]);\n\
\n\
    *v0 = q_fin_0.val[0];\n\
    *v1 = q_fin_0.val[1];\n\
    *v2 = q_fin_1.val[0];\n\
    *v3 = q_fin_1.val[1];\n\
    *v4 = q_fin_2.val[0];\n\
    *v5 = q_fin_2.val[1];\n\
    *v6 = q_fin_3.val[0];\n\
    *v7 = q_fin_3.val[1];\n\
}}\n\
#endif\n\
\n\
template<int act_k>\n\
inline void lut_ctor(int8_t* qlut, bitnet_float_type* b, bitnet_float_type* lut_scales) {{\n\
#ifdef __ARM_NEON\n\
    int16x8_t vec_lut[16];\n\
    float32_t scales = *lut_scales;\n\
        uint8_t tbl_mask[16];\n\
        tbl_mask[0] = 0;\n\
        tbl_mask[1] = 2;\n\
        tbl_mask[2] = 4;\n\
        tbl_mask[3] = 6;\n\
        tbl_mask[4] = 8;\n\
        tbl_mask[5] = 10;\n\
        tbl_mask[6] = 12;\n\
        tbl_mask[7] = 14;\n\
        tbl_mask[8] = 1;\n\
        tbl_mask[9] = 3;\n\
        tbl_mask[10] = 5;\n\
        tbl_mask[11] = 7;\n\
        tbl_mask[12] = 9;\n\
        tbl_mask[13] = 11;\n\
        tbl_mask[14] = 13;\n\
        tbl_mask[15] = 15;\n\
        uint8x16_t tbl_mask_q = vld1q_u8(tbl_mask);\n\
#pragma unroll\n\
    for (int k = 0; k < act_k / 16; ++k) {{\n\
        float32x4x2_t vec_bs_x0 = vld2q_f32(b + k * 16);\n\
        float32x4x2_t vec_bs_x1 = vld2q_f32(b + k * 16 + 8);\n\
        float32x4_t vec_f_0 = vmulq_n_f32(vec_bs_x0.val[0], scales);\n\
        float32x4_t vec_f_1 = vmulq_n_f32(vec_bs_x0.val[1], scales);\n\
        float32x4_t vec_f_2 = vmulq_n_f32(vec_bs_x1.val[0], scales);\n\
        float32x4_t vec_f_3 = vmulq_n_f32(vec_bs_x1.val[1], scales);\n\
        int32x4_t vec_b_0 = vcvtnq_s32_f32(vec_f_0);\n\
        int32x4_t vec_b_1 = vcvtnq_s32_f32(vec_f_1);\n\
        int32x4_t vec_b_2 = vcvtnq_s32_f32(vec_f_2);\n\
        int32x4_t vec_b_3 = vcvtnq_s32_f32(vec_f_3);\n\
        int16x4_t vec_b16_0 = vmovn_s32(vec_b_0);\n\
        int16x4_t vec_b16_1 = vmovn_s32(vec_b_1);\n\
        int16x4_t vec_b16_2 = vmovn_s32(vec_b_2);\n\
        int16x4_t vec_b16_3 = vmovn_s32(vec_b_3);\n\
        int16x8_t vec_bs_0 = vcombine_s16(vec_b16_0, vec_b16_2);\n\
        int16x8_t vec_bs_1 = vcombine_s16(vec_b16_1, vec_b16_3);\n\
        vec_lut[0] = vdupq_n_s16(0);\n\
        vec_lut[0] = vec_lut[0] - vec_bs_0;\n\
        vec_lut[0] = vec_lut[0] - vec_bs_1;\n\
        vec_lut[1] = vdupq_n_s16(0);\n\
        vec_lut[1] = vec_lut[1] - vec_bs_0;\n\
        vec_lut[2] = vdupq_n_s16(0);\n\
        vec_lut[2] = vec_lut[2] - vec_bs_0;\n\
        vec_lut[2] = vec_lut[2] + vec_bs_1;\n\
        vec_lut[3] = vdupq_n_s16(0);\n\
        vec_lut[3] = vec_lut[3] - vec_bs_1;\n\
        vec_lut[4] = vdupq_n_s16(0);\n\
        vec_lut[5] = vec_bs_1;\n\
        vec_lut[6] = vec_bs_0;\n\
        vec_lut[6] = vec_lut[6] - vec_bs_1;\n\
        vec_lut[7] = vec_bs_0;\n\
        vec_lut[8] = vec_bs_0;\n\
        vec_lut[8] = vec_lut[8] + vec_bs_1;\n\
        Transpose_8_8(&(vec_lut[0]), &(vec_lut[1]), &(vec_lut[2]), &(vec_lut[3]),\n\
                      &(vec_lut[4]), &(vec_lut[5]), &(vec_lut[6]), &(vec_lut[7]));\n\
        Transpose_8_8(&(vec_lut[8]), &(vec_lut[9]), &(vec_lut[10]), &(vec_lut[11]),\n\
                      &(vec_lut[12]), &(vec_lut[13]), &(vec_lut[14]), &(vec_lut[15]));\n\
#pragma unroll\n\
        for (int idx = 0; idx < 8; idx++) {{\n\
            int8x16_t q0_s = vqtbl1q_s8(vreinterpretq_s8_s16(vec_lut[idx]), tbl_mask_q);\n\
            int8x8_t q0_low = vget_low_s8(q0_s);\n\
            int8x8_t q0_high = vget_high_s8(q0_s);\n\
            int8x16_t q1_s = vqtbl1q_s8(vreinterpretq_s8_s16(vec_lut[idx + 8]), tbl_mask_q);\n\
            int8x8_t q1_low = vget_low_s8(q1_s);\n\
            int8x8_t q1_high = vget_high_s8(q1_s);\n\
            vst1_s8(qlut + k * 16 * 8 * 2 + idx * 16 * 2, q0_high);\n\
            vst1_s8(qlut + k * 16 * 8 * 2 + idx * 16 * 2 + 8, q1_high);\n\
            vst1_s8(qlut + k * 16 * 8 * 2 + idx * 16 * 2 + 16, q0_low);\n\
            vst1_s8(qlut + k * 16 * 8 * 2 + idx * 16 * 2 + 24, q1_low);\n\
        }}\n\
    }}\n\
#endif\n\
}}\n\
\n\
static bool is_type_supported(enum ggml_type type) {{\n\
    if (type == GGML_TYPE_Q4_0 ||\n\
        type == GGML_TYPE_TL1) {{\n\
        return true;\n\
    }} else {{\n\
        return false;\n\
    }}\n\
}}\n\
"
    return kernel_code

def gen_body_core_code(bm, by):
    length = 4
    all_code = ""
    for i in range(length):
        core_code = "\n\
            uint8x16_t vec_a_{0} = vld1q_u8(a + i * KK / 2 + k * 32 * 2 + {0} * 16);\n\
            uint8x16_t vec_a{0}_top = vshrq_n_u8(vec_a_{0}, 4);\n\
            uint8x16_t vec_a{0}_bot = vandq_u8(vec_a_{0}, vec_mask);\n\
            int8x16_t  vec_v_{0}_left_tmp0 = vqtbl1q_s8(vec_lut[{1} * k + {2}], vec_a{0}_top);\n\
            int8x16_t  vec_v_{0}_left_tmp1 = vqtbl1q_s8(vec_lut[{1} * k + {3}], vec_a{0}_top);\n\
            int8x16_t  vec_v_{0}_right_tmp0 = vqtbl1q_s8(vec_lut[{1} * k + {4}], vec_a{0}_bot);\n\
            int8x16_t  vec_v_{0}_right_tmp1 = vqtbl1q_s8(vec_lut[{1} * k + {5}], vec_a{0}_bot);\n\
            int8x16x2_t  vec_v_left_{0} = vzipq_s8(vec_v_{0}_left_tmp1, vec_v_{0}_left_tmp0);\n\
            int8x16x2_t  vec_v_right_{0} = vzipq_s8(vec_v_{0}_right_tmp1, vec_v_{0}_right_tmp0);\n\
            vec_c[{6}] += vec_v_left_{0}.val[0];\n\
            vec_c[{6}] += vec_v_right_{0}.val[0];\n\
            vec_c[{7}] += vec_v_left_{0}.val[1];\n\
            vec_c[{7}] += vec_v_right_{0}.val[1];\n\
        ".format(i, 2 * by // 2, (4 * i) % (2 * by // 2), (4 * i + 1) % (2 * by // 2), (4 * i + 2) % (2 * by // 2), (4 * i + 3) % (2 * by // 2), (i * 2) // (by // 2) * 2 + 0, (i * 2) // (by // 2) * 2 + 1)
        
        all_code = "".join([all_code, core_code])

    all_code = "".join([all_code, "\n       }\n\n"])

    for i in range(bm // 8):
        core_code = "\
        int32x4_t vec_v_bot_low_low_{0} = vmovl_s16(vget_low_s16(vec_c[{0}]));\n\
        int32x4_t vec_v_bot_low_high_{0} = vmovl_high_s16(vec_c[{0}]);\n\
        vst1q_s32(c + i + {1}, vld1q_s32(c + i + {1}) + vec_v_bot_low_low_{0});\n\
        vst1q_s32(c + i + {2}, vld1q_s32(c + i + {2}) + vec_v_bot_low_high_{0});\n".format(i, i * 8, i * 8 + 4)
        all_code = "".join([all_code, core_code])

    return all_code

def gen_tbl_impl(pre, BM, BK, bm, k):

    kernel_code = "\
#include <arm_neon.h>\n\
\n\
#define BM{0} {1}\n\
#define BBK{0} {2}\n\
inline void tbl_impl_{0}(int32_t* c, int8_t* lut, uint8_t* a) {{\n\
#ifdef __ARM_NEON\n\
    const int KK = BBK{0} / 2;\n\
    const uint8x16_t vec_mask = vdupq_n_u8(0x0f);\n\
    const int8x16_t vec_zero = vdupq_n_s16(0x0000);\n\
    int8x16_t vec_lut[2 * KK];\n\
".format(pre, BM, BK)
    
    kernel_code = "".join([kernel_code, "    int16x8_t vec_c[{}];".format(bm // 8)])

    kernel_code = "".join([kernel_code, "\n\
#pragma unroll\n\
    for (int k = 0; k < 2 * KK; k++) {\n\
        vec_lut[k] = vld1q_s8(lut + k * 16);\n\
    }\n"])

    pre_core_code = "\n\
#pragma unroll\n\
    for (int i = 0; i < BM{}; i += {}) {{\n\
        #pragma unroll\n\
        for (int i=0; i<{}; i++) {{\n\
            vec_c[i] = vandq_s16(vec_c[i], vec_zero);\n\
        }}\n".format(pre, bm, bm // 8)

    body_core_pre_code = "\n\
#pragma unroll\n\
        for (int k = 0; k < KK / {}; k++) {{\n\
            ".format(256 // bm // 2)

    body_core_post_code = "\n\
    }\n\
\
#endif\n\
}\n"

    kernel_code = "".join([kernel_code, pre_core_code, body_core_pre_code, gen_body_core_code(bm, 256 // bm), body_core_post_code])

    kernel_code = "".join([kernel_code, "\n\
int32_t qgemm_lut_{0}(void* A, void* LUT, void* Scales, void* LUT_Scales, void* C) {{\n\
    alignas({1}) uint32_t CBits[BM{0}];\n\
    memset(&(CBits[0]), 0, BM{0} * sizeof(int32_t));\n\
#pragma unroll\n\
    for (int32_t k_outer = 0; k_outer < {2} / BBK{0}; ++k_outer) {{\n\
        tbl_impl_{0}((&(((int32_t*)CBits)[0])), (&(((int8_t*)LUT)[(k_outer * BBK{0} / 2 * 32)])), (&(((uint8_t*)A)[(k_outer * BBK{0} / 2 / 2 * BM{0})])));\n\
    }}\n\
#pragma unroll\n\
    for (int i = 0; i < BM{0}; i++) {{\n\
        ((bitnet_float_type*)C)[i] = (((int32_t*)CBits)[i]) / ((bitnet_float_type*)LUT_Scales)[0] * ((bitnet_float_type*)Scales)[0];\n\
    }}\n\
  return 0;\n\
}};\n".format(pre, min(32, BK), k)])

    return kernel_code

def gen_top_api(kernel_shapes):

    kernel_code = "void ggml_preprocessor(int m, int k, void* B, void* LUT_Scales, void* QLUT) {{\n\
    if (m == {0} && k == {1}) {{\n\
        preprocessor_k<{1}>(B, LUT_Scales, QLUT);\n\
    }}\n\
".format(kernel_shapes[0][0], kernel_shapes[0][1])
    for i in range(1, len(kernel_shapes)):
        kernel_code = "".join([kernel_code, "    else if (m == {0} && k == {1}) {{\n\
        preprocessor_k<{1}>(B, LUT_Scales, QLUT);\n\
    }}\n".format(kernel_shapes[i][0], kernel_shapes[i][1])])
    kernel_code = "".join([kernel_code, "}\n"])
    kernel_code = "".join([kernel_code, "void ggml_qgemm_lut(int m, int k, void* A, void* LUT, void* Scales, void* LUT_Scales, void* C) {{\n\
    if (m == {0} && k == {1}) {{\n\
        qgemm_lut_{0}_{1}(A, LUT, Scales, LUT_Scales, C);\n\
    }}\n\
".format(kernel_shapes[0][0], kernel_shapes[0][1])])
    for i in range(1, len(kernel_shapes)):
        kernel_code = "".join([kernel_code, "    else if (m == {0} && k == {1}) {{\n\
        qgemm_lut_{0}_{1}(A, LUT, Scales, LUT_Scales, C);\n\
    }}\n\
".format(kernel_shapes[i][0], kernel_shapes[i][1])])
    kernel_code = "".join([kernel_code, "}\n"])
    return kernel_code

def gen_preprocess_code():
    kernel_code = "\n\
template<int K>\n\
void preprocessor_k(void* B, void* LUT_Scales, void* QLUT) {{\n\
  partial_max_reset((&(((bitnet_float_type*)LUT_Scales)[0])));\n\
  per_tensor_quant(K, (&(((bitnet_float_type*)LUT_Scales)[0])), (&(((bitnet_float_type*)B)[0])));\n\
  \n\
  lut_ctor<K>((&(((int8_t*)QLUT)[0])), (&(((bitnet_float_type*)B)[0])), (&(((bitnet_float_type*)LUT_Scales)[0])));\n\
}}\n"
    return kernel_code

def gen_transform_code(kernel_shape):
    kernel_code = "\n\
void ggml_bitnet_transform_tensor(struct ggml_tensor * tensor) {\n\
    if (!(is_type_supported(tensor->type) && tensor->backend == GGML_BACKEND_TYPE_CPU && tensor->extra == nullptr)) {\n\
        return;\n\
    }\n\
\n\
    int k = tensor->ne[0];\n\
    int m = tensor->ne[1];\n\
    const int lut_scales_size = 1;\n\
    const int scales_size = 1;\n\
    int bk = 0;\n\
    int bm = 0;\n"

    kernel_code = "".join([kernel_code, "\n\
    if (m == {0} && k == {1}) {{\n\
        bm = BM{0}_{1};\n\
        bk = BBK{0}_{1};\n\
    }}\n".format(kernel_shapes[0][0], kernel_shapes[0][1])])

    for i in range(1, len(kernel_shapes)):
        kernel_code = "".join([kernel_code, "else if (m == {0} && k == {1}) {{\n\
        bm = BM{0}_{1};\n\
        bk = BBK{0}_{1};\n\
    }}\n".format(kernel_shapes[i][0], kernel_shapes[i][1])])

    kernel_code = "".join([kernel_code, "\n\
    const int n_tile_num = m / bm;\n\
    const int BK = bk;\n\
    uint8_t * qweights;\n\
    bitnet_float_type * scales;\n\
\n\
    scales = (bitnet_float_type *) aligned_malloc(sizeof(bitnet_float_type));\n\
    qweights = (uint8_t *) tensor->data;\n\
    float * i2_scales = (float * )(qweights + k * m / 4);\n\
    scales[0] = (bitnet_float_type) i2_scales[0];\n\
\n\
    tensor->extra = bitnet_tensor_extras + bitnet_tensor_extras_index;\n\
    bitnet_tensor_extras[bitnet_tensor_extras_index++] = {\n\
        /* .lut_scales_size = */ lut_scales_size,\n\
        /* .BK              = */ BK,\n\
        /* .n_tile_num      = */ n_tile_num,\n\
        /* .qweights        = */ qweights,\n\
        /* .scales          = */ scales\n\
    };\n\
}\n"])

    return kernel_code

if __name__ == "__main__":
    ModelShapeDict = {
        "bitnet_b1_58-large"                : [[1536, 4096],
                                               [1536, 1536],
                                               [4096, 1536]],
        "bitnet_b1_58-3B"                   : [[3200, 8640],
                                               [3200, 3200],
                                               [8640, 3200]],
        "Llama3-8B-1.58-100B-tokens"        : [[14336, 4096],
                                               [4096, 14336],
                                               [1024, 4096],
                                               [4096, 4096]] 
    }
    
    parser = argparse.ArgumentParser(description='gen impl')
    parser.add_argument('--model',default="input", type=str, dest="model", 
                        help="choose from bitnet_b1_58-large/bitnet_b1_58-3B/Llama3-8B-1.58-100B-tokens.")
    parser.add_argument('--BM',default="input", type=str,
                        help="block length when cutting one weight (M, K) into M / BM weights (BM, K).")
    parser.add_argument('--BK',default="input", type=str,
                        help="block length when cutting one weight (M, K) into K / BK weights (M, BK).")
    parser.add_argument('--bm',default="input", type=str,
                        help="using simd instructions to compute (bm, 256 / bm) in one block")
    args = parser.parse_args()

    kernel_shapes = ModelShapeDict[args.model]

    BM_list = [int(item) for item in args.BM.split(',')]
    BK_list = [int(item) for item in args.BK.split(',')]
    bm_list = [int(item) for item in args.bm.split(',')]

    assert(len(BM_list) == len(BK_list) == len(bm_list) == len(kernel_shapes)), "number of BM / BK / bm shoud be {}".format(len(kernel_shapes))
    
    for i in range(len(kernel_shapes)):
        assert kernel_shapes[i][0] % BM_list[i] == 0, "M %% BM should be 0"
        assert kernel_shapes[i][1] % BK_list[i] == 0, "K %% BK should be 0"
        assert bm_list[i] in [32, 64], "choose bm from [32, 64]"

    tbl_impl_code = []

    for i in range(len(kernel_shapes)):
        tbl_impl_code.append(
            gen_tbl_impl("{}_{}".format(kernel_shapes[i][0], kernel_shapes[i][1]), BM_list[i], BK_list[i], bm_list[i], kernel_shapes[i][1])
        )
    api_code = gen_top_api(kernel_shapes)
    pre_code = gen_preprocess_code()
    ctor_code = gen_ctor_code()
    trans_code = gen_transform_code(kernel_shapes)

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "include")

    with open(''.join([output_dir, "/bitnet-lut-kernels.h"]), 'w') as f:
        f.write(''.join("#if defined(GGML_BITNET_ARM_TL1)"))
        f.write(''.join(ctor_code))
        for code in tbl_impl_code:
            f.write(''.join(code))
        f.write(''.join(pre_code))
        f.write(''.join(api_code))
        f.write(''.join(trans_code))
        f.write(''.join("#endif"))

    config = ConfigParser()

    for i in range(len(kernel_shapes)):
        config.add_section('Kernels_{}'.format(i))
        config.set('Kernels_{}'.format(i), 'M'.format(i), str(kernel_shapes[i][0]))
        config.set('Kernels_{}'.format(i), 'K'.format(i), str(kernel_shapes[i][1]))
        config.set('Kernels_{}'.format(i), 'BM'.format(i), str(BM_list[i]))
        config.set('Kernels_{}'.format(i), 'BK'.format(i), str(BK_list[i]))
        config.set('Kernels_{}'.format(i), 'bmm'.format(i), str(bm_list[i]))

    with open(''.join([output_dir, "/kernel_config.ini"]), 'w') as configfile:
        config.write(configfile)