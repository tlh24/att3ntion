from setuptools import setup
import torch
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

def get_cuda_arch_flags():
	"""Generate nvcc -gencode flags for all detected GPU architectures.

	Produces SASS for each detected arch AND embeds PTX for the highest
	detected arch to enable forward compatibility with newer GPUs
	(e.g., code compiled on sm_89 can JIT-compile for sm_120 at load time).
	"""
	# Default to sm_89 (Ada Lovelace / RTX 4080) if no GPU detected
	seen_archs = set()
	device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
	for i in range(device_count):
		major, minor = torch.cuda.get_device_capability(i)
		seen_archs.add(f'{major}{minor}')

	if not seen_archs:
		seen_archs.add('89')

	flags = []
	max_arch = max(seen_archs)
	for arch in sorted(seen_archs):
		# Emit SASS for each detected architecture
		flags.append(f'-gencode=arch=compute_{arch},code=sm_{arch}')

	# Also embed PTX for the highest arch for forward compatibility
	# This allows the CUDA driver to JIT-compile for future architectures
	flags.append(f'-gencode=arch=compute_{max_arch},code=compute_{max_arch}')

	return flags

setup(
    name='hyper_attn_extensions',
    version='0.2.0',
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
                    *get_cuda_arch_flags(),
                    # '-DTORCH_USE_CUDA_DSA', # for debugging
                    # '-lineinfo' # for debugging
                ]
            }
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
) 
