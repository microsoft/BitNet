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


def gen_tbl_impl(kernel_shapes, BM_list, BK_list, bm_list, k_list):
    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
    )

    kernel_template = env.get_template("tl1_table.h")

    kernel_code = kernel_template.render(bm_list=bm_list, kernel_shapes=kernel_shapes, BM_list=BM_list, BK_list=BK_list, range=range, length=4, k_list=k_list, min=min)

    return kernel_code

def gen_top_api(kernel_shapes):
    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
    )

    kernel_template = env.get_template("tl1_top_api.h")

    kernel_code = kernel_template.render(kernel_shapes=kernel_shapes)

    return kernel_code

def gen_preprocess_code():
    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
    )

    kernel_template = env.get_template("tl1_preprocess.h")

    kernel_code = kernel_template.render()

    return f"\n{kernel_code}\n"

def gen_transform_code(kernel_shapes):
    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
    )

    kernel_template = env.get_template("tl1_transform.h")

    kernel_code = kernel_template.render(kernel_shapes=kernel_shapes)

    return f"\n{kernel_code}\n"

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


    tbl_impl_code = gen_tbl_impl(kernel_shapes=kernel_shapes, BM_list=BM_list, BK_list=BK_list, bm_list=bm_list, k_list=[item[1] for item in kernel_shapes])
    api_code = gen_top_api(kernel_shapes)
    pre_code = gen_preprocess_code()
    ctor_code = gen_ctor_code()
    trans_code = gen_transform_code(kernel_shapes)

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "include")

    with open(''.join([output_dir, "/bitnet-lut-kernels.h"]), 'w') as f:
        f.write(''.join(ctor_code))
        f.write(''.join(tbl_impl_code))
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
