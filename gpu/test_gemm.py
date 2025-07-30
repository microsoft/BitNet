import torch
from pack_weight import convert_weight_int8_to_int2
from torch.profiler import profile, record_function, ProfilerActivity
import ctypes
import numpy as np
from torch.utils import benchmark

gemm_lib = ctypes.CDLL('bitnet_kernels/libgemm.so')
# set all seed
torch.manual_seed(42)
np.random.seed(42)

def bit_linear_int8xint2(input0, weight, out, M, N, K):
    stream = torch.cuda.current_stream()
    gemm_lib.bitlinear_int8xint2(*[
        ctypes.c_void_p(input0.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int(M),
        ctypes.c_int(N),
        ctypes.c_int(K),
        ctypes.c_void_p(stream.cuda_stream),])

M = 512
test_list = [
    (2560,  2560), 
    (3840,  2560), 
    (13824, 2560),
    (2560,  6912),
]
for N,K in test_list:
    weight = torch.randint(-1, 2, (N, K), dtype=torch.int8, device='cuda')
    weight_scale = torch.ones(1, dtype=torch.bfloat16, device='cuda')
    weight_compressed = convert_weight_int8_to_int2(weight).to('cuda')
    weight_np = weight.cpu().to(torch.int32).T.numpy()
    stream = torch.cuda.current_stream()
    input0 = torch.randint(-128,127,(M, K),dtype=torch.int8, device='cuda')
    input0_np = input0.cpu().to(torch.int32).numpy()
    out_np = np.matmul(input0_np, weight_np)
    weight_bf16 = weight.to(torch.bfloat16).T
    input0_bf16 = input0.to(torch.bfloat16)
    s = torch.ones(1, dtype=torch.bfloat16, device='cuda')
    ws = torch.ones(6, dtype=torch.bfloat16, device='cuda')
    out = torch.empty(M, N, dtype=torch.int32, device='cuda')
    t0 = benchmark.Timer(
        stmt="bit_linear_int8xint2(input0, weight_compressed, out, M, N, K)",
        setup="from __main__ import input0, weight_compressed, s, ws, out, bit_linear_int8xint2, M, N, K",
        num_threads=1,
    )

    t1 = benchmark.Timer(
        stmt="out_bf16 = torch.matmul(input0_bf16, weight_bf16)",
        setup="from __main__ import input0_bf16, weight_bf16",
        num_threads=1,
    )

    time0 = t0.timeit(50)
    time1 = t1.timeit(50)

    print(f'Shape{M,N,K}, W2A8: {time0.mean * 1e6:.2f}us, torch BF16: {time1.mean * 1e6:.2f}us')
    out_np = torch.tensor(out_np).cuda()

    print(f'custom == np {torch.all(out==out_np)}')
