from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='bitlinear_cpp',
    ext_modules=[
        CUDAExtension('bitlinear_cuda', [
            'bitnet_kernels.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })