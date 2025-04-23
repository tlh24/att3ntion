from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

setup(
    name='hyper_attn_extensions',
    ext_modules=[
        CppExtension(
            name='hyper_attn_cpp_reference',
            sources=['cpp/torch_att3ntion.cpp'],
        ),
        CUDAExtension(
            name='hyper_attn_cpp_manual',
            sources=[
                'cpp/manual_att3ntion.cpp',
                'cuda/manual_att3ntion.cu'
            ]
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
) 