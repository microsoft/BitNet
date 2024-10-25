import argparse
import os
from pathlib import Path
from configparser import ConfigParser
from jinja2 import Environment, FileSystemLoader

def gen_ctor_code():
    return "\n" + (Path(__file__).parent / "templates" / "tl2_ctor.h").read_text(encoding='utf-8')

def gen_tbl_impl(pre, BM, BK, bm, k_list):
    env = Environment(
                loader=FileSystemLoader(Path(__file__).parent / "templates"),
            )
    template = env.get_template("tl2_table_impl.h")
    return "\n" + template.render(pre=pre, BM=BM, BK=BK, bm=bm, k_list=k_list)

def gen_top_api(kernel_shapes, k_list):

    kernel_code = "void ggml_preprocessor(int bs, int m, int three_k, int two_k, void* B, void* LUT_Scales, void* Three_QLUT, void* Two_QLUT) {{\n\
    partial_max_reset(bs, (&(((float*)LUT_Scales)[0])));\n\
    if (m == {0} && two_k == {1} && three_k == {2}) {{\n\
        for (int32_t b = 0; b < bs; b++) {{\n\
            per_tensor_quant(two_k + three_k, (&(((float*)LUT_Scales)[b])), (&(((float*)B)[b * (two_k + three_k)])));\n\
            three_lut_ctor<{2}>((&(((int8_t*)Three_QLUT)[b * three_k / 3 * 32])), (&(((float*)B)[b * (three_k + two_k)])), (&(((float*)LUT_Scales)[b])));\n\
            two_lut_ctor<{1}>((&(((int8_t*)Two_QLUT)[b * two_k / 2 * 32])), (&(((float*)B)[b * (three_k + two_k) + {2}])), (&(((float*)LUT_Scales)[b])));\n\
        }}\n\
    }}\n\
".format(kernel_shapes[0][0], k_list[0][0], k_list[0][1])
    for i in range(1, len(kernel_shapes)):
        kernel_code = "".join([kernel_code, "    else if (m == {0} && two_k == {1} && three_k == {2}) {{\n\
        for (int32_t b = 0; b < bs; b++) {{\n\
            per_tensor_quant(two_k + three_k, (&(((float*)LUT_Scales)[b])), (&(((float*)B)[b * (two_k + three_k)])));\n\
            three_lut_ctor<{2}>((&(((int8_t*)Three_QLUT)[b * three_k / 3 * 32])), (&(((float*)B)[b * (three_k + two_k)])), (&(((float*)LUT_Scales)[b])));\n\
            two_lut_ctor<{1}>((&(((int8_t*)Two_QLUT)[b * two_k / 2 * 32])), (&(((float*)B)[b * (three_k + two_k) + {2}])), (&(((float*)LUT_Scales)[b])));\n\
        }}\n\
    }}\n".format(kernel_shapes[i][0], k_list[i][0], k_list[i][1])])
    kernel_code = "".join([kernel_code, "}\n"])


    kernel_code = "".join([kernel_code, "void ggml_qgemm_lut(int bs, int m, int k, int BK, void* A, void* sign, void* LUT, void* Scales, void* LUT_Scales, void* C) {{\n\
    if (m == {0} && k == {1}) {{\n\
        if (BK == {2}) {{\n\
            if (bs == 1) {{\n\
                two_qgemm_lut_{4}<1>(A, LUT, Scales, LUT_Scales, C);\n\
            }} else if (bs == 8) {{\n\
                two_qgemm_lut_{4}<8>(A, LUT, Scales, LUT_Scales, C);\n\
            }} else if (bs == 32) {{\n\
                two_qgemm_lut_{4}<32>(A, LUT, Scales, LUT_Scales, C);\n\
            }} else if (bs == 128) {{\n\
                two_qgemm_lut_{4}<128>(A, LUT, Scales, LUT_Scales, C);\n\
            }} else if (bs == 256) {{\n\
                two_qgemm_lut_{4}<256>(A, LUT, Scales, LUT_Scales, C);\n\
            }} else if (bs == 512) {{\n\
                two_qgemm_lut_{4}<512>(A, LUT, Scales, LUT_Scales, C);\n\
            }}\n\
        }}\n\
        else if (BK == {3}) {{\n\
            if (bs == 1) {{\n\
                three_qgemm_lut_{4}<1>(A, sign, LUT, Scales, LUT_Scales, C);\n\
            }}else if (bs == 8) {{\n\
                three_qgemm_lut_{4}<8>(A, sign, LUT, Scales, LUT_Scales, C);\n\
            }}else if (bs == 32) {{\n\
                three_qgemm_lut_{4}<32>(A, sign, LUT, Scales, LUT_Scales, C);\n\
            }}else if (bs == 128) {{\n\
                three_qgemm_lut_{4}<128>(A, sign, LUT, Scales, LUT_Scales, C);\n\
            }}else if (bs == 256) {{\n\
                three_qgemm_lut_{4}<256>(A, sign, LUT, Scales, LUT_Scales, C);\n\
            }}else if (bs == 512) {{\n\
                three_qgemm_lut_{4}<512>(A, sign, LUT, Scales, LUT_Scales, C);\n\
            }}\n\
        }}\n\
    }}\n\
".format(kernel_shapes[0][0], kernel_shapes[0][1], k_list[0][0], k_list[0][1], "{}_{}".format(kernel_shapes[0][0], kernel_shapes[0][1]))])
    for i in range(1, len(kernel_shapes)):
        kernel_code = "".join([kernel_code, "    else if (m == {0} && k == {1}) {{\n\
        if (BK == {2}) {{\n\
            if (bs == 1) {{\n\
                two_qgemm_lut_{4}<1>(A, LUT, Scales, LUT_Scales, C);\n\
            }} else if (bs == 8) {{\n\
                two_qgemm_lut_{4}<8>(A, LUT, Scales, LUT_Scales, C);\n\
            }} else if (bs == 32) {{\n\
                two_qgemm_lut_{4}<32>(A, LUT, Scales, LUT_Scales, C);\n\
            }} else if (bs == 128) {{\n\
                two_qgemm_lut_{4}<128>(A, LUT, Scales, LUT_Scales, C);\n\
            }} else if (bs == 256) {{\n\
                two_qgemm_lut_{4}<256>(A, LUT, Scales, LUT_Scales, C);\n\
            }} else if (bs == 512) {{\n\
                two_qgemm_lut_{4}<512>(A, LUT, Scales, LUT_Scales, C);\n\
            }}\n\
        }}\n\
        else if (BK == {3}) {{\n\
            if (bs == 1) {{\n\
                three_qgemm_lut_{4}<1>(A, sign, LUT, Scales, LUT_Scales, C);\n\
            }}else if (bs == 8) {{\n\
                three_qgemm_lut_{4}<8>(A, sign, LUT, Scales, LUT_Scales, C);\n\
            }}else if (bs == 32) {{\n\
                three_qgemm_lut_{4}<32>(A, sign, LUT, Scales, LUT_Scales, C);\n\
            }}else if (bs == 128) {{\n\
                three_qgemm_lut_{4}<128>(A, sign, LUT, Scales, LUT_Scales, C);\n\
            }}else if (bs == 256) {{\n\
                three_qgemm_lut_{4}<256>(A, sign, LUT, Scales, LUT_Scales, C);\n\
            }}else if (bs == 512) {{\n\
                three_qgemm_lut_{4}<512>(A, sign, LUT, Scales, LUT_Scales, C);\n\
            }}\n\
        }}\n\
    }}\n\
".format(kernel_shapes[i][0], kernel_shapes[i][1], k_list[i][0], k_list[i][1], "{}_{}".format(kernel_shapes[i][0], kernel_shapes[i][1]))])
    kernel_code = "".join([kernel_code, "}\n"])
    return kernel_code

def gen_transform_code(kernel_shapes):
    kernel_code = "\n\
void ggml_bitnet_transform_tensor(struct ggml_tensor * tensor) {\n\
    if (!(is_type_supported(tensor->type) && tensor->backend == GGML_BACKEND_TYPE_CPU && tensor->extra == nullptr)) {\n\
        return;\n\
    }\n\
\n\
    int k = tensor->ne[0];\n\
    int m = tensor->ne[1];\n\
    const int lut_scales_size = 1;\n\
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
    int nbytes = (k - 256) * m / 3 * 5 / 8 + 256 * m / 2 * 4 / 8;\n\
    if (nbytes % 32 != 0) nbytes = 32 - nbytes % 32 + nbytes;\n\
    float * i2_scales = (float * )(qweights + nbytes);\n\
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

def get_three_k_two_k(K, bk):
    bk_num = K // bk
    three_k = bk_num * bk
    two_k = K - three_k
    return two_k, three_k

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
                        help="using simd instructions to compute (bm, 192 / bm) in one block")
    args = parser.parse_args()

    kernel_shapes = ModelShapeDict[args.model]

    BM_list = [int(item) for item in args.BM.split(',')]
    BK_list = [int(item) for item in args.BK.split(',')]
    bm_list = [int(item) for item in args.bm.split(',')]

    tbl_impl_code = []
    k_list = []

    for i in range(len(kernel_shapes)):
        k_list.append(get_three_k_two_k(kernel_shapes[i][1], BK_list[i]))

    for i in range(len(kernel_shapes)):
        tbl_impl_code.append(
            gen_tbl_impl("{}_{}".format(kernel_shapes[i][0], kernel_shapes[i][1]), BM_list[i], BK_list[i], bm_list[i], k_list[i])
        )

    assert(len(BM_list) == len(BK_list) == len(bm_list) == len(kernel_shapes)), "number of BM / BK / bm shoud be {}".format(len(kernel_shapes))
    
    for i in range(len(kernel_shapes)):
        assert kernel_shapes[i][0] % BM_list[i] == 0, "M %% BM should be 0"
        assert (kernel_shapes[i][1] % BK_list[i]) % 32 == 0, "K %% BK %% 32 should be 0"
        assert bm_list[i] in [32], "choose bm from [32]"

    ctor_code = gen_ctor_code()
    api_code = gen_top_api(kernel_shapes, k_list)
    trans_code = gen_transform_code(kernel_shapes)

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "include")

    with open(''.join([output_dir, "/bitnet-lut-kernels.h"]), 'w') as f:
        f.write(''.join("#if defined(GGML_BITNET_X86_TL2)"))
        f.write(''.join(ctor_code))
        for code in tbl_impl_code:
            f.write(''.join(code))
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