from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

setup(
    name='hyper_attn_extensions',
    ext_modules=[
        CppExtension(
            name='hyper_attn_cpp_reference',
            sources=['cpp/torch_reference.cpp'],
        ),
        CUDAExtension(
            name='hyper_attn_cpp_manual',
            sources=[
                'cpp/cuda_bindings.cpp',
                'cuda/forward.cu',
                'cuda/backward.cu'
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-gencode=arch=compute_89,code=sm_89',
                    # '-lineinfo' # uncomment for debugging
                ]
            }
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
) 