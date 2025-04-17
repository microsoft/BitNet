import argparse
from pathlib import Path
from configparser import ConfigParser
from jinja2 import Environment, FileSystemLoader

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

    k_list = []

    for i in range(len(kernel_shapes)):
        k_list.append(get_three_k_two_k(kernel_shapes[i][1], BK_list[i]))

    assert(len(BM_list) == len(BK_list) == len(bm_list) == len(kernel_shapes)), "number of BM / BK / bm shoud be {}".format(len(kernel_shapes))
    
    for i in range(len(kernel_shapes)):
        assert kernel_shapes[i][0] % BM_list[i] == 0, "M %% BM should be 0"
        assert (kernel_shapes[i][1] % BK_list[i]) % 32 == 0, "K %% BK %% 32 should be 0"
        assert bm_list[i] in [32], "choose bm from [32]"

    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
    )
    template = env.get_template("tl2.h")
    contents = template.render(
        kernel_shapes=kernel_shapes,
        k_list=k_list,
        BM_list=BM_list,
        BK_list=BK_list,
        bm_list=bm_list,
    )

    output_dir = Path(__file__).resolve().parent.parent / "include"

    header_file = output_dir / "bitnet-lut-kernels.h"
    header_file.write_text(contents, encoding='utf-8')

    config = ConfigParser()

    for i in range(len(kernel_shapes)):
        config.add_section('Kernels_{}'.format(i))
        config.set('Kernels_{}'.format(i), 'M'.format(i), str(kernel_shapes[i][0]))
        config.set('Kernels_{}'.format(i), 'K'.format(i), str(kernel_shapes[i][1]))
        config.set('Kernels_{}'.format(i), 'BM'.format(i), str(BM_list[i]))
        config.set('Kernels_{}'.format(i), 'BK'.format(i), str(BK_list[i]))
        config.set('Kernels_{}'.format(i), 'bmm'.format(i), str(bm_list[i]))

    config_file = output_dir / "kernel_config.ini"
    with open(config_file, 'w', encoding='utf-8') as configfile:
        config.write(configfile)
