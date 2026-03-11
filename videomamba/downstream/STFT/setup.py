# Copyright (c) SenseTime Research and its affiliates. All Rights Reserved.
#!/usr/bin/env python

import glob
import os

import torch
from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

requirements = ["torch", "torchvision"]


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "stft_core", "csrc")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv("FORCE_CUDA", "0") == "1":
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            #1.额外添加
            "-gencode=arch=compute_60,code=sm_60",
            "-gencode=arch=compute_70,code=sm_70",
            "-gencode=arch=compute_75,code=sm_75",
            "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_86,code=sm_86",
            # # 强制架构
            # "-arch=sm_60",                # 直接指定目标架构
            # "--ptxas-options=-v",         # 输出汇编信息，便于确认
            # "-Xcompiler=-Wno-deprecated-declarations",
        ]
        # #2.修改内容
        # include_dirs = [extensions_dir, os.path.join(CUDA_HOME, "include")]
    else:
        include_dirs = [extensions_dir]   
    include_dirs = [extensions_dir]
    
    sources = [os.path.join(extensions_dir, s) for s in sources]


    ext_modules = [
        extension(
            "stft_core._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            # #3.额外添加
            # library_dirs=[os.path.join(CUDA_HOME, "lib64")],  # 添加库路径
            # libraries=["cudart"],  # 显式链接CUDA运行时库
        )
    ]

    return ext_modules


setup(
    name="stft_core",
    version="0.1",
    author="Lingyun",
    url="https://github.com/lingyunwu14/STFT",
    description="video object detection in pytorch",
    packages=find_packages(exclude=("configs", "tests",)),
    # install_requires=requirements,
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
