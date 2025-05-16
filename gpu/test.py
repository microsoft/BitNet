import torch
from torch.utils import benchmark
from torch import nn

from pack_weight import convert_weight_int8_to_int2
from torch.profiler import profile, record_function, ProfilerActivity
import ctypes
import numpy as np
# set all seed
torch.manual_seed(42)
np.random.seed(42)

bitnet_lib = ctypes.CDLL('bitnet_kernels/libbitnet.so')

def bitnet_int8xint2_linear(input0, input1, s, ws, ret):
    out_shape = list(input0.shape)
    out_shape[-1] = input1.shape[0]

    stream = torch.cuda.current_stream()

    M = input0.shape[0]
    if len(out_shape) == 3: 
        M *= input0.shape[1]
    N = input1.shape[0]
    K = input1.shape[1] * 4

    bitnet_lib.bitlinear_int8xint2(*[ctypes.c_void_p(input0.data_ptr()), ctypes.c_void_p(input1.data_ptr()), ctypes.c_void_p(ret.data_ptr()), ctypes.c_void_p(s.data_ptr()), ctypes.c_void_p(ws.data_ptr()), ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K), ctypes.c_void_p(stream.cuda_stream)])

    return ret

if __name__ == '__main__':
    test_list = [
        (2560,  2560), 
        (3840,  2560), 
        (13824, 2560),
        (2560,  6912) ,
        (3200, 3200), 
        (4800, 3200), 
        (3200, 10240),
        (20480, 3200),
    ]
    for N,K in test_list:
        weight = torch.randint(-1, 2, (N, K), dtype=torch.int8, device='cuda')
        weight_scale = torch.ones(1, dtype=torch.bfloat16, device='cuda')
        weight_compressed = convert_weight_int8_to_int2(weight).to('cuda')

        for i in range(1):
            input0 = torch.randint(-128,127,(1, K),dtype=torch.int8, device='cuda')
            input0_bf16 = input0.to(torch.bfloat16)
            input_np = input0.cpu().to(torch.int32).numpy()
            weight_np = weight.cpu().to(torch.int32).T.numpy()
            out_np = np.matmul(input_np,weight_np)
            out_np = torch.tensor(out_np).cuda().to(torch.bfloat16)

            s = torch.ones(1, dtype=torch.bfloat16, device='cuda')
            ws = torch.ones(6, dtype=torch.bfloat16, device='cuda')

            ret = torch.empty((1,N), dtype=torch.bfloat16, device=input0.device)
            out = bitnet_int8xint2_linear(input0, weight_compressed, s, ws, ret)

            print(f'custom == np {torch.all(out==out_np)}')

        input0 = torch.randint(-128,127,(1, K),dtype=torch.int8, device='cuda')
        input0_fp16 = input0.to(torch.float16)
        input0_bf16 = input0.to(torch.bfloat16)
        weight_fp16 = weight.to(torch.float16).T
        weight_bf16 = weight.to(torch.bfloat16).T
        ret = torch.empty((1,N), dtype=torch.bfloat16, device=input0.device)
        s = torch.ones(1, dtype=torch.bfloat16, device='cuda')
        ws = torch.ones(6, dtype=torch.bfloat16, device='cuda')
        t0 = benchmark.Timer(
            stmt="bitnet_int8xint2_linear(input0, weight_compressed, s, ws, ret)",
            setup="from __main__ import input0, weight_compressed, s, ws, ret, bitnet_int8xint2_linear",
            num_threads=1,
        )

        t1 = benchmark.Timer(
            stmt="torch.matmul(input0_bf16,weight_bf16)",
            setup="from __main__ import input0_bf16, weight_bf16",
            num_threads=1,
        )

        time0 = t0.timeit(50)
        time1 = t1.timeit(50)

        print(f'Shape{N,K}, W2A8: {time0.mean * 1e6:.2f}us, torch BF16: {time1.mean * 1e6:.2f}us')
        # activities = [ ProfilerActivity.CUDA, 
        #             #   ProfilerActivity.CPU
        #               ]
        # sort_by_keyword = 'cuda' + "_time_total"
        # with profile(activities=activities, record_shapes=True) as prof:
        #     with record_function("model_inference1"):
        #         for _ in range(10):
        #             bitnet_int8xint2_linear(input0, weight_compressed, s, ws, ret)
        #             torch.matmul(input0_fp16,weight_fp16)
        #             torch.matmul(input0_bf16,weight_bf16)

        # print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=15))
        
