"""Compile and exercise the actual bounds/mask helpers; no GPU or torch needed."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


def test_bounds_and_mask_packing_without_gpu(tmp_path):
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not shutil.which(compiler[0]):
        pytest.skip("a C++ compiler is required for the host helper test")
    include_flags = []
    cuda_home = os.environ.get("CUDA_HOME")
    nvcc = shutil.which("nvcc")
    if not cuda_home and nvcc:
        cuda_home = str(Path(nvcc).resolve().parent.parent)
    if cuda_home:
        include_flags = ["-I", str(Path(cuda_home) / "include")]
    if not cuda_home and not Path("/usr/include/cuda_runtime.h").exists():
        pytest.skip("CUDA headers are required; set CUDA_HOME to the toolkit")
    binary = tmp_path / "single_gather_bounds_host"
    source = Path(__file__).with_name("single_gather_bounds_host.cpp")
    subprocess.run(
        [*compiler, "-std=c++17", "-O2", *include_flags, str(source), "-o", str(binary)],
        check=True, capture_output=True, text=True,
    )
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    assert "PASS: 208 coverage configurations; 4224 mask alignment/tail cases" in result.stdout
    print(result.stdout, end="")
