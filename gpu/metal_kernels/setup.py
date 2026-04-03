# Copyright (c) Microsoft. All rights reserved.
# Metal backend for BitNet inference on Apple Silicon

from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension
import torch
import os


def get_include_dirs():
    """Get include directories for PyTorch."""
    include_dirs = []

    # PyTorch include directories
    torch_include = os.path.join(os.path.dirname(torch.__file__), "include")
    include_dirs.append(torch_include)

    # PyTorch API include
    torch_api_include = os.path.join(torch_include, "torch", "csrc", "api", "include")
    if os.path.exists(torch_api_include):
        include_dirs.append(torch_api_include)

    return include_dirs


def get_metal_compile_args():
    """Get Metal compiler arguments."""
    # Metal shaders are compiled at runtime, so we just need to package them
    return []


def get_metal_link_args():
    """Get Metal linker arguments."""
    # Link against Metal framework
    return ["-framework", "Metal", "-framework", "Foundation"]


# Get PyTorch include directories
include_dirs = get_include_dirs()

setup(
    name="bitnet_metal",
    version="0.1.0",
    ext_modules=[
        Extension(
            "bitnet_metal",
            sources=["metal_backend.mm"],  # Objective-C++ wrapper
            include_dirs=include_dirs,
            extra_compile_args=["-std=c++17", "-ObjC++"] + get_metal_compile_args(),
            extra_link_args=get_metal_link_args(),
            language="objc++",
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    package_data={
        "": ["*.metal"],  # Include Metal shader files
    },
)
