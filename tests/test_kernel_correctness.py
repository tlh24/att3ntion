#!/usr/bin/env python3
"""
Comprehensive CUDA kernel test suite.

Tests all forward and backward pass kernels against PyTorch autograd reference
across various configurations to ensure mathematical correctness.

Forward kernels: Y_q, Y_r, Y_s (gather), Y_q_, Y_r_, Y_s_ (scatter)
Backward kernels: grad_Q, grad_R, grad_S, grad_Vq_1, grad_Vq_2, grad_Vr_1, grad_Vr_2, grad_Vs_1, grad_Vs_2

Constraints (optimized kernels):
    - N must be a multiple of 16
    - D must be one of 16, 32, or 64
    - I == J == K (enforced by using N for all)

Config groups are pytest markers: `-m quick`, `-m "quick or standard"`,
`-m large`, `-m stress`, `-m edge`. Bare `pytest` runs every group.
"""
import os
import sys
from dataclasses import dataclass

import pytest
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import att3ntion._cuda_kernels as cuda_ext
import att3ntion._torch_kernels as ref_ext

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

FORWARD_KERNEL_NAMES = ['Y_q', 'Y_r', 'Y_s', 'Y_q_', 'Y_r_', 'Y_s_']
BACKWARD_KERNEL_NAMES = [
    'grad_Q', 'grad_R', 'grad_S',
    'grad_Vq_1', 'grad_Vq_2',
    'grad_Vr_1', 'grad_Vr_2',
    'grad_Vs_1', 'grad_Vs_2'
]
INPUT_NAMES = ['Q', 'R', 'S', 'Vq_1', 'Vq_2', 'Vr_1', 'Vr_2', 'Vs_1', 'Vs_2']

# Backward at larger N costs more than it catches; the shapes below it already
# cover every dispatch branch.
MAX_N_BACKWARD = 256


@dataclass
class TestConfig:
    """Configuration for a single test case."""
    __test__ = False  # data, not a test class -- keeps pytest from collecting it

    name: str
    B: int
    H: int
    N: int
    D: int
    input_scale: float = 1.0

    @property
    def I(self) -> int:
        return self.N

    @property
    def J(self) -> int:
        return self.N

    @property
    def K(self) -> int:
        return self.N

    def __str__(self) -> str:
        scale_str = f" scale={self.input_scale}" if self.input_scale != 1.0 else ""
        return f"{self.name}: B={self.B} H={self.H} N={self.N} D={self.D}{scale_str}"


# N=16 pads the forward tensor-core resident dim to 64; N=128 is the smallest
# shape reaching the unmasked Y_gather_tc variant.
QUICK_CONFIGS = [
    TestConfig("small_N16_D64",     B=1, H=2, N=16,  D=64),
    TestConfig("small_N128_D64",    B=1, H=2, N=128, D=64),
    TestConfig("small_N32_D32",     B=1, H=2, N=32, D=32),
    TestConfig("small_N64_D32",     B=1, H=2, N=64, D=32),
    TestConfig("small_N32_D64",     B=1, H=2, N=32, D=64),
]

STANDARD_CONFIGS = [
    # Varying N (multiples of 32, full sweep)
    TestConfig("N32_D32",           B=1, H=2, N=32,  D=32),
    TestConfig("N64_D32",           B=1, H=2, N=64,  D=32),
    TestConfig("N96_D32",           B=1, H=2, N=96,  D=32),
    TestConfig("N128_D32",          B=1, H=2, N=128, D=32),
    TestConfig("N160_D32",          B=1, H=2, N=160, D=32),
    TestConfig("N192_D32",          B=1, H=2, N=192, D=32),
    TestConfig("N224_D32",          B=1, H=2, N=224, D=32),
    TestConfig("N256_D32",          B=1, H=2, N=256, D=32),
    TestConfig("N288_D32",          B=1, H=2, N=288, D=32),
    TestConfig("N320_D32",          B=1, H=2, N=320, D=32),
    TestConfig("N352_D32",          B=1, H=2, N=352, D=32),
    TestConfig("N384_D32",          B=1, H=2, N=384, D=32),
    TestConfig("N416_D32",          B=1, H=2, N=416, D=32),
    TestConfig("N448_D32",          B=1, H=2, N=448, D=32),
    TestConfig("N480_D32",          B=1, H=2, N=480, D=32),
    TestConfig("N512_D32",          B=1, H=2, N=512, D=32),
    # Varying D (N x D cross-product, D max 64)
    TestConfig("N32_D64",           B=1, H=2, N=32,  D=64),
    TestConfig("N64_D64",           B=1, H=2, N=64,  D=64),
    TestConfig("N96_D64",           B=1, H=2, N=96,  D=64),
    TestConfig("N128_D64",          B=1, H=2, N=128, D=64),
    # Varying B (every value 1-8)
    TestConfig("B1_N32_D32",        B=1, H=2, N=32, D=32),
    TestConfig("B2_N32_D32",        B=2, H=2, N=32, D=32),
    TestConfig("B3_N32_D32",        B=3, H=2, N=32, D=32),
    TestConfig("B4_N32_D32",        B=4, H=2, N=32, D=32),
    TestConfig("B5_N32_D32",        B=5, H=2, N=32, D=32),
    TestConfig("B6_N32_D32",        B=6, H=2, N=32, D=32),
    TestConfig("B7_N32_D32",        B=7, H=2, N=32, D=32),
    TestConfig("B8_N32_D32",        B=8, H=2, N=32, D=32),
    # Varying H (every value 1-8, plus 12,16)
    TestConfig("H1_N32_D32",        B=1, H=1, N=32, D=32),
    TestConfig("H2_N32_D32",        B=1, H=2, N=32, D=32),
    TestConfig("H3_N32_D32",        B=1, H=3, N=32, D=32),
    TestConfig("H4_N32_D32",        B=1, H=4, N=32, D=32),
    TestConfig("H5_N32_D32",        B=1, H=5, N=32, D=32),
    TestConfig("H6_N32_D32",        B=1, H=6, N=32, D=32),
    TestConfig("H7_N32_D32",        B=1, H=7, N=32, D=32),
    TestConfig("H8_N32_D32",        B=1, H=8, N=32, D=32),
    TestConfig("H12_N32_D32",       B=1, H=12, N=32, D=32),
    TestConfig("H16_N32_D32",       B=1, H=16, N=32, D=32),
    # Combined B*H
    TestConfig("B2_H4_N32_D32",     B=2, H=4, N=32, D=32),
    TestConfig("B3_H3_N32_D32",     B=3, H=3, N=32, D=32),
    TestConfig("B2_H4_N32_D64",     B=2, H=4, N=32, D=64),
    TestConfig("B4_H2_N64_D32",     B=4, H=2, N=64, D=32),
    TestConfig("B2_H8_N32_D64",     B=2, H=8, N=32, D=64),
    TestConfig("B4_H4_N64_D64",     B=4, H=4, N=64, D=64),
    TestConfig("B8_H2_N32_D32",     B=8, H=2, N=32, D=32),
    # Medium-large with higher B*H
    TestConfig("med_N64_H4_D32",    B=1, H=4, N=64,  D=32),
    TestConfig("med_N96_D32",       B=1, H=2, N=96,  D=32),
    TestConfig("med_N128_D32",      B=1, H=2, N=128, D=32),
    TestConfig("med_N160_D32",      B=1, H=2, N=160, D=32),
    TestConfig("med_N192_D32",      B=1, H=2, N=192, D=32),
    TestConfig("med_N224_D32",      B=1, H=2, N=224, D=32),
    TestConfig("med_N256_D32",      B=1, H=2, N=256, D=32),
]

LARGE_CONFIGS = [
    TestConfig("large_N128_B2H6",   B=2, H=6, N=128, D=32),
    TestConfig("large_N128_D64",    B=2, H=4, N=128, D=64),
    TestConfig("large_N160_D32",    B=2, H=4, N=160, D=32),
    TestConfig("large_N160_D64",    B=2, H=2, N=160, D=64),
    TestConfig("large_N192_D32",    B=2, H=2, N=192, D=32),
    TestConfig("large_N192_D64",    B=1, H=2, N=192, D=64),
    TestConfig("large_N224_D32",    B=2, H=2, N=224, D=32),
    TestConfig("large_N224_D64",    B=1, H=2, N=224, D=64),
    TestConfig("large_N256_D32",    B=2, H=2, N=256, D=32),
    TestConfig("large_N256_D64",    B=1, H=2, N=256, D=64),
    TestConfig("large_N288_D32",    B=1, H=2, N=288, D=32),
    TestConfig("large_N320_D32",    B=1, H=1, N=320, D=32),
    TestConfig("large_N384_D32",    B=1, H=1, N=384, D=32),
    TestConfig("large_N448_D32",    B=1, H=1, N=448, D=32),
    TestConfig("large_N512_D32",    B=1, H=1, N=512, D=32),
    # Cross-product (N x D interactions)
    TestConfig("cross_N64_D64",     B=1, H=2, N=64,  D=64),
    TestConfig("cross_N128_D64",    B=1, H=1, N=128, D=64),
    TestConfig("cross_N192_D32",    B=1, H=1, N=192, D=32),
    TestConfig("cross_N256_D32",    B=1, H=1, N=256, D=32),
    TestConfig("cross_N256_D64",    B=1, H=1, N=256, D=64),
    # Production-like configurations
    TestConfig("prod_B4H8_N64",     B=4, H=8, N=64,  D=32),
    TestConfig("prod_B2H12_N64",    B=2, H=12, N=64, D=32),
    TestConfig("prod_B8H4_N32",     B=8, H=4, N=32,  D=64),
    TestConfig("prod_B1H16_N128",   B=1, H=16, N=128, D=32),
]

STRESS_CONFIGS = [
    TestConfig("stress_scale2",     B=1, H=2, N=32, D=32, input_scale=2.0),
    TestConfig("stress_scale3",     B=1, H=2, N=32, D=32, input_scale=3.0),
    TestConfig("stress_scale5",     B=1, H=2, N=32, D=32, input_scale=5.0),
    TestConfig("stress_N64_s2",     B=1, H=2, N=64, D=32, input_scale=2.0),
    TestConfig("stress_D64_s2",     B=1, H=2, N=32, D=64, input_scale=2.0),
    TestConfig("stress_N64_D64_s2", B=1, H=2, N=64, D=64, input_scale=2.0),
]

EDGE_CONFIGS = [
    TestConfig("B1_H1_N32_D32",     B=1, H=1, N=32, D=32),
    TestConfig("B1_H1_N64_D64",     B=1, H=1, N=64, D=64),
    TestConfig("B1_H1_N96_D32",     B=1, H=1, N=96, D=32),
    TestConfig("B1_H1_N128_D64",    B=1, H=1, N=128, D=64),
]

CONFIG_GROUPS = {
    'quick': QUICK_CONFIGS,
    'standard': STANDARD_CONFIGS,
    'large': LARGE_CONFIGS,
    'stress': STRESS_CONFIGS,
    'edge': EDGE_CONFIGS,
}

# The D=64 tensor-core kernels round Q_i (*) R_j to bf16 for the MMA, leaving
# scores ~2^-9 relative where the scalar D=16/32 path is exact. The softmax
# exponentiates that, so the output error runs ~exp(2.1e-3 * |score|) - 1: 1% at
# input_scale=1, but 6% by input_scale=2. cuda_docs/gather_readme.md section 5
# has the envelope; test_tc_paths.py checks the TC path against the scalar
# path's own error instead, which is the criterion that isolates the kernel.
TC_SCORE_PRECISION = pytest.mark.xfail(
    reason="D=64 TC path: bf16 score rounding amplified by exp at input_scale>1",
    strict=False)


def _config_params():
    """Dedup configs by shape across groups; a shape listed in several groups
    keeps all their markers so `-m quick` and `-m standard` both select it."""
    seen = {}
    for group, configs in CONFIG_GROUPS.items():
        for c in configs:
            key = (c.B, c.H, c.N, c.D, c.input_scale)
            seen.setdefault(key, (c, []))[1].append(group)
    return [
        pytest.param(c, id=c.name,
                     marks=[getattr(pytest.mark, g) for g in groups]
                     + ([TC_SCORE_PRECISION] if c.D == 64 and c.input_scale > 1.0
                        else []))
        for c, groups in seen.values()
    ]


CONFIG_PARAMS = _config_params()


@pytest.fixture(scope="module", params=CONFIG_PARAMS)
def case(request):
    """One CUDA pass and one reference pass per config, shared by every kernel
    assertion below. Module scope makes pytest group a config's tests together,
    so the passes run once instead of once per kernel name.

    Seeded from the shape so a single selected test draws the same inputs it
    would have drawn in a full run.
    """
    config = request.param
    B, H, N, D = config.B, config.H, config.N, config.D
    torch.manual_seed(hash((B, H, N, D, config.input_scale)) % (2 ** 31))

    def rnd():
        return torch.randn(B, H, N, D, device='cuda') * config.input_scale

    bf16 = {n: rnd().to(torch.bfloat16) for n in INPUT_NAMES}
    grad_Y = [rnd().to(torch.bfloat16) for _ in range(6)]
    # The reference runs in FP32 but on the kernel's own BF16 inputs. Handing it
    # the unrounded FP32 draw instead measures the dtype conversion rather than
    # the kernel: scores here are a triple product, so input_scale=s scales them
    # by s^3, and by s=2 the softmax is nearly a hard argmax over I*J*K -- BF16's
    # 2^-9 rounding then moves which cell wins, and the quantization term alone
    # swamps the kernel's own error (which is flat in input_scale).
    inputs = {n: v.float() for n, v in bf16.items()}

    cuda_fwd = cuda_ext.forward(*[bf16[n] for n in INPUT_NAMES], 0.0)

    # Grad on the reference only where the backward kernels are actually
    # checked. The reference builds I*J*K intermediates, so retaining the graph
    # for a shape that never calls backward is what pushed the largest configs
    # into OOM.
    need_backward = N <= MAX_N_BACKWARD
    with torch.set_grad_enabled(need_backward):
        ref_inputs = {n: v.detach().clone().requires_grad_(need_backward)
                      for n, v in inputs.items()}
        ref_fwd = ref_ext.forward(*[ref_inputs[n] for n in INPUT_NAMES], 0.0)

    cuda_grads = ref_grads = None
    if need_backward:
        # fwd outputs 6..11 are the softmax stats (m_i, l_i, m_j, l_j, m_k, l_k)
        cuda_grads = cuda_ext.backward(
            *[g.to(torch.bfloat16) for g in grad_Y],
            *[bf16[n] for n in INPUT_NAMES],
            *cuda_fwd[6:12],
            0.0,
        )
        sum((y * g).sum() for y, g in zip(ref_fwd, grad_Y)).backward()
        ref_grads = [
            ref_inputs[n].grad if ref_inputs[n].grad is not None
            else torch.zeros_like(ref_inputs[n])
            for n in INPUT_NAMES
        ]

    return config, cuda_fwd, ref_fwd, cuda_grads, ref_grads

    del inputs, bf16, grad_Y, cuda_fwd, ref_fwd, cuda_grads, ref_grads
    torch.cuda.empty_cache()


def _atol(base, ref):
    """Scale the absolute floor by the tensor's magnitude. `allclose` applies
    atol per element, so a fixed floor holds near-zero cells to the same
    absolute error whatever the rest of the tensor is doing -- and outputs here
    grow with input_scale^2, up to |Y| ~ 188. A fixed 5e-2 there is 20x below one
    BF16 ULP at that magnitude, so no kernel could pass it.
    """
    return base * max(1.0, ref.abs().max().item())


@pytest.mark.parametrize("kernel_name", FORWARD_KERNEL_NAMES)
def test_forward_kernel(case, kernel_name):
    config, cuda_fwd, ref_fwd, _, _ = case
    idx = FORWARD_KERNEL_NAMES.index(kernel_name)
    cuda_tensor, ref_tensor = cuda_fwd[idx], ref_fwd[idx]

    assert cuda_tensor.dtype == torch.bfloat16, \
        f"{kernel_name} ({config}): expected BF16 CUDA output, got {cuda_tensor.dtype}"
    assert cuda_tensor.shape == ref_tensor.shape, \
        f"{kernel_name} ({config}): shape {cuda_tensor.shape} vs {ref_tensor.shape}"
    assert torch.isfinite(cuda_tensor).all(), \
        f"{kernel_name} ({config}): CUDA output contains NaN or Inf"

    cuda_compare, ref_compare = cuda_tensor.float(), ref_tensor.float()
    assert torch.allclose(cuda_compare, ref_compare, rtol=5e-2,
                          atol=_atol(5e-2, ref_compare)), (
        f"{kernel_name} ({config}): max diff "
        f"{(cuda_compare - ref_compare).abs().max().item():.3e}")


@pytest.mark.parametrize("kernel_name", BACKWARD_KERNEL_NAMES)
def test_backward_kernel(case, kernel_name):
    config, _, _, cuda_grads, ref_grads = case
    if cuda_grads is None:
        pytest.skip(f"backward skipped for N={config.N} (> {MAX_N_BACKWARD})")

    idx = BACKWARD_KERNEL_NAMES.index(kernel_name)
    ref_tensor = ref_grads[idx]
    cuda_tensor = cuda_grads[idx]
    if cuda_tensor is None:
        cuda_tensor = torch.zeros_like(ref_tensor)

    assert cuda_tensor.shape == ref_tensor.shape, \
        f"{kernel_name} ({config}): shape {cuda_tensor.shape} vs {ref_tensor.shape}"
    assert torch.isfinite(cuda_tensor).all(), \
        f"{kernel_name} ({config}): CUDA gradient contains NaN or Inf"

    # grad_Q/R/S integrate the longest reduction chains and are most
    # sensitive to BF16 input quantization + atomic accumulation order.
    if kernel_name in {'grad_Q', 'grad_R', 'grad_S'}:
        rtol, atol = 5e-2, 1e-1
    else:
        rtol, atol = 2e-2, 2e-2

    cuda_compare, ref_compare = cuda_tensor.float(), ref_tensor.float()
    assert torch.allclose(cuda_compare, ref_compare, rtol=rtol,
                          atol=_atol(atol, ref_compare)), (
        f"{kernel_name} ({config}): max diff "
        f"{(cuda_compare - ref_compare).abs().max().item():.3e}")


CONFIGS_BY_NAME = {p.values[0].name: p.values[0] for p in CONFIG_PARAMS}
