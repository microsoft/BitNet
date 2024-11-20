import argparse
import os
from configparser import ConfigParser
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

def gen_ctor_code():
    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
    )
    template = env.get_template("tl1_ctor.h")
    kernel_code = template.render()

    return f"\n{kernel_code}\n"

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
