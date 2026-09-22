import os
from setuptools import setup, find_packages
from pathlib import Path
import subprocess
import torch
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension, CUDA_HOME

def get_cuda_arch_flags():
	"""Generate nvcc -gencode flags for all detected GPU architectures.

	Produces SASS for each detected arch AND embeds PTX for the highest
	detected arch to enable forward compatibility with newer GPUs
	(e.g., code compiled on sm_89 can JIT-compile for sm_120 at load time).
	PTX only JITs forward, so a build cannot run on an arch older than the
	oldest one it was compiled for.

	Returns [] when TORCH_CUDA_ARCH_LIST is set, handing control to
	torch.utils.cpp_extension -- the standard way to target cards that are not
	present at build time, e.g. TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0+PTX".
	sm_120 (Blackwell consumer, RTX 50xx) additionally needs CUDA >= 12.8.
	"""
	if os.environ.get('TORCH_CUDA_ARCH_LIST'):
		return []

	seen_archs = set()
	device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
	for i in range(device_count):
		seen_archs.add(torch.cuda.get_device_capability(i))

	# No GPU at build time (containers, CI): cover every arch the tensor-core
	# path supports rather than guessing one and stranding the rest.
	if not seen_archs:
		seen_archs = {(8, 0), (8, 6), (8, 9), (9, 0)}

	flags = [f'-gencode=arch=compute_{M}{m},code=sm_{M}{m}'
	         for M, m in sorted(seen_archs)]

	# Embed PTX for the newest arch so future cards JIT at load time. Compared
	# as tuples, not strings: '120' < '89' lexically but sm_120 is newer.
	M, m = max(seen_archs)
	flags.append(f'-gencode=arch=compute_{M}{m},code=compute_{M}{m}')

	return flags


CUTLASS_INCLUDE = os.environ.get('ATT3NTION_CUTLASS_INCLUDE')
HOPPER_ENABLED = bool(CUTLASS_INCLUDE)

class Att3BuildExtension(BuildExtension):
    def build_extensions(self):
        if HOPPER_ENABLED:
            include = Path(CUTLASS_INCLUDE).resolve()
            if not (include / 'cute/tensor.hpp').is_file():
                raise RuntimeError('ATT3NTION_CUTLASS_INCLUDE must contain CuTe headers (validated: CUTLASS 3.5.1)')
            objects=[]
            for source in ('shared_hopper','shared_hopper_dq','shared_hopper_rs','shared_hopper_rs64'):
                target = Path(self.build_temp).resolve() / (source+'.o')
                target.parent.mkdir(parents=True,exist_ok=True)
                subprocess.run([str(Path(CUDA_HOME)/'bin/nvcc'),'-std=c++17','--expt-relaxed-constexpr',
                    '-O3','-lineinfo','-arch=sm_90a','-Xcompiler=-fPIC,-fvisibility=hidden',
                    '-I'+str(include),'-c',str(Path('cuda',source+'.cu').resolve()),'-o',str(target)],check=True)
                objects.append(str(target))
            for extension in self.extensions:
                if extension.name == 'att3ntion._cuda_kernels':
                    extension.extra_objects = list(extension.extra_objects or []) + objects
        super().build_extensions()

setup(
    name='att3ntion',
    packages=find_packages(include=['att3ntion','att3ntion.*']),
    version='0.2.0',
    ext_modules=[
        CppExtension(
            name='att3ntion._torch_kernels',
            sources=['cpp/torch_reference.cpp'],
        ),
        CUDAExtension(
            name='att3ntion._cuda_kernels',
            define_macros=[('ATT3NTION_WITH_HOPPER',None)] if HOPPER_ENABLED else [],
            sources=[
                'cpp/cuda_bindings.cpp',
                'cuda/forward.cu',
                'cuda/backward.cu',
                'cuda/single_gather_shared.cu'
            ],
            extra_compile_args={
                'cxx': ['-O3','-fvisibility=hidden'],
                'nvcc': [
                    '-Xcompiler=-fvisibility=hidden',
                    '-O3',
                    *get_cuda_arch_flags(),
                    # '-DTORCH_USE_CUDA_DSA',  #for debugging
                    '-lineinfo',  # source lines in sanitizer/ncu reports
                ]
            }
        ),
    ],
    cmdclass={
        "build_ext": Att3BuildExtension
    }
) 
