from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="hyper_attn_cpp",
    ext_modules=[
        CppExtension(
            name="hyper_attn_cpp",
            sources=["cpp/hyper_attn_cpp.cpp"],
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
) 