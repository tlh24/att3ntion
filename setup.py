from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='hyper_attn_extensions',
    ext_modules=[
        CppExtension(
            name='hyper_attn_cpp_reference',
            sources=['cpp/hyper_attn_cpp_reference.cpp']
        ),
        CppExtension(
            name='hyper_attn_cpp_manual',
            sources=['cpp/hyper_attn_cpp_manual.cpp']
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
) 