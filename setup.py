from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='hyper_attn_extensions',
    ext_modules=[
        CppExtension(
            name='hyper_attn_cpp_reference',
            sources=['cpp/torch_att3ntion.cpp']
        ),
        CppExtension(
            name='hyper_attn_cpp_manual',
            sources=['cpp/manual_att3ntion.cpp']
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
) 