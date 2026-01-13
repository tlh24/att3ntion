from setuptools import setup
import torch
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

def get_cuda_arch_flags():
	# return flags for the minimum capability seen
	arch_flags = []
	min_arch = 999

	for i in range(torch.cuda.device_count()):
		major, minor = torch.cuda.get_device_capability(i)
		decimal = major + (minor / 100)
		if decimal < min_arch:
			arch = f'{major}{minor}'

	return f'-gencode=arch=compute_{arch},code=sm_{arch}'

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
                    get_cuda_arch_flags(),
                    # '-lineinfo' # uncomment for debugging
                ]
            }
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
) 
