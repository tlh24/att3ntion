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
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-gencode=arch=compute_86,code=sm_86',
                    '-lineinfo'
                ]
            }
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
) 