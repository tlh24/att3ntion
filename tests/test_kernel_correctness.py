#!/usr/bin/env python3
"""
Comprehensive CUDA kernel test suite.

Tests all forward and backward pass kernels against PyTorch autograd reference
across various configurations to ensure mathematical correctness.

Forward kernels: Y_q, Y_r, Y_s (gather), Y_q_, Y_r_, Y_s_ (scatter)
Backward kernels: grad_Q, grad_R, grad_S, grad_Vq_1, grad_Vq_2, grad_Vr_1, grad_Vr_2, grad_Vs_1, grad_Vs_2

Constraints (optimized kernels):
    - N must be a multiple of 32
    - D must be a multiple of 32 and <= 64 (shared memory limit)
    - I == J == K (enforced by using N for all)
"""
import os
import sys
import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import defaultdict

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

FORWARD_KERNEL_NAMES = ['Y_q', 'Y_r', 'Y_s', 'Y_q_', 'Y_r_', 'Y_s_']
BACKWARD_KERNEL_NAMES = [
    'grad_Q', 'grad_R', 'grad_S',
    'grad_Vq_1', 'grad_Vq_2',
    'grad_Vr_1', 'grad_Vr_2',
    'grad_Vs_1', 'grad_Vs_2'
]
ALL_KERNEL_NAMES = FORWARD_KERNEL_NAMES + BACKWARD_KERNEL_NAMES


@dataclass
class TestConfig:
    """Configuration for a single test case."""
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


# All configs use N and D as multiples of 32 for optimized kernel requirements
QUICK_CONFIGS = [
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


def get_test_configs(
    quick: bool = False,
    standard: bool = True,
    large: bool = False,
    stress: bool = False,
    edge: bool = False
) -> List[TestConfig]:
    """Get test configurations based on flags."""
    configs = []
    if quick:
        configs.extend(QUICK_CONFIGS)
    if standard:
        configs.extend(STANDARD_CONFIGS)
    if large:
        configs.extend(LARGE_CONFIGS)
    if stress:
        configs.extend(STRESS_CONFIGS)
    if edge:
        configs.extend(EDGE_CONFIGS)

    seen = set()
    unique_configs = []
    for c in configs:
        key = (c.B, c.H, c.N, c.D, c.input_scale)
        if key not in seen:
            seen.add(key)
            unique_configs.append(c)
    return unique_configs


@dataclass
class TestResult:
    """Result of a single kernel test."""
    kernel_name: str
    config: TestConfig
    passed: bool
    max_diff: float
    error: Optional[str] = None
    duration_ms: float = 0.0


class KernelTester:
    """Tester for CUDA attention kernels against PyTorch reference."""

    def __init__(
        self,
        device: torch.device,
        rtol: float = 1e-4,
        atol: float = 1e-5,
        verbose: bool = False
    ):
        self.device = device
        self.rtol = rtol
        self.atol = atol
        self.verbose = verbose

        try:
            import att3ntion._cuda_kernels as cuda_ext
            self.cuda_ext = cuda_ext
        except ImportError as e:
            raise ImportError(
                f"Could not import 'att3ntion._cuda_kernels': {e}\n"
                "Run: pip install -e ."
            )

        try:
            import att3ntion._torch_kernels as ref_ext
            self.ref_ext = ref_ext
        except ImportError as e:
            raise ImportError(
                f"Could not import 'att3ntion._torch_kernels': {e}\n"
                "Run: pip install -e ."
            )

    def _create_inputs(self, config: TestConfig, requires_grad: bool = False) -> Dict[str, torch.Tensor]:
        B, H, N, D = config.B, config.H, config.N, config.D
        scale = config.input_scale

        inputs = {
            'Q':    torch.randn(B, H, N, D, device=self.device, dtype=torch.float32) * scale,
            'R':    torch.randn(B, H, N, D, device=self.device, dtype=torch.float32) * scale,
            'S':    torch.randn(B, H, N, D, device=self.device, dtype=torch.float32) * scale,
            'Vq_1': torch.randn(B, H, N, D, device=self.device, dtype=torch.float32) * scale,
            'Vq_2': torch.randn(B, H, N, D, device=self.device, dtype=torch.float32) * scale,
            'Vr_1': torch.randn(B, H, N, D, device=self.device, dtype=torch.float32) * scale,
            'Vr_2': torch.randn(B, H, N, D, device=self.device, dtype=torch.float32) * scale,
            'Vs_1': torch.randn(B, H, N, D, device=self.device, dtype=torch.float32) * scale,
            'Vs_2': torch.randn(B, H, N, D, device=self.device, dtype=torch.float32) * scale,
        }

        if requires_grad:
            for t in inputs.values():
                t.requires_grad_(True)
        return inputs

    def _clone_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in inputs.items()}

    def _detach_clone_inputs(self, inputs: Dict[str, torch.Tensor], requires_grad: bool = True) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone().requires_grad_(requires_grad) for k, v in inputs.items()}
    
    def test_forward_kernel(self, kernel_name: str, config: TestConfig) -> TestResult:
        """Test a single forward kernel against reference."""
        if kernel_name not in FORWARD_KERNEL_NAMES:
            raise ValueError(f"Unknown forward kernel: {kernel_name}")

        kernel_idx = FORWARD_KERNEL_NAMES.index(kernel_name)
        start_time = time.time()

        try:
            inputs = self._create_inputs(config)

            cuda_out = self.cuda_ext.forward(
                inputs['Q'].clone().to(torch.bfloat16),
                inputs['R'].clone().to(torch.bfloat16),
                inputs['S'].clone().to(torch.bfloat16),
                inputs['Vq_1'].clone().to(torch.bfloat16),
                inputs['Vq_2'].clone().to(torch.bfloat16),
                inputs['Vr_1'].clone().to(torch.bfloat16),
                inputs['Vr_2'].clone().to(torch.bfloat16),
                inputs['Vs_1'].clone().to(torch.bfloat16),
                inputs['Vs_2'].clone().to(torch.bfloat16),
                0.0
            )

            ref_out = self.ref_ext.forward(
                inputs['Q'].clone(), inputs['R'].clone(), inputs['S'].clone(),
                inputs['Vq_1'].clone(), inputs['Vq_2'].clone(),
                inputs['Vr_1'].clone(), inputs['Vr_2'].clone(),
                inputs['Vs_1'].clone(), inputs['Vs_2'].clone(),
                0.0
            )

            cuda_tensor = cuda_out[kernel_idx]
            ref_tensor = ref_out[kernel_idx]

            if cuda_tensor.dtype != torch.bfloat16:
                return TestResult(
                    kernel_name=kernel_name, config=config, passed=False,
                    max_diff=float('nan'),
                    error=f"Expected BF16 CUDA output, got {cuda_tensor.dtype}",
                    duration_ms=(time.time() - start_time) * 1000
                )

            if cuda_tensor.shape != ref_tensor.shape:
                return TestResult(
                    kernel_name=kernel_name, config=config, passed=False,
                    max_diff=float('nan'),
                    error=f"Shape mismatch: {cuda_tensor.shape} vs {ref_tensor.shape}",
                    duration_ms=(time.time() - start_time) * 1000
                )

            if torch.isnan(cuda_tensor).any() or torch.isinf(cuda_tensor).any():
                return TestResult(
                    kernel_name=kernel_name, config=config, passed=False,
                    max_diff=float('nan'), error="CUDA output contains NaN or Inf",
                    duration_ms=(time.time() - start_time) * 1000
                )

            cuda_compare = cuda_tensor.float()
            ref_compare = ref_tensor.float()
            max_diff = (cuda_compare - ref_compare).abs().max().item()
            passed = torch.allclose(cuda_compare, ref_compare, rtol=max(self.rtol, 5e-2), atol=max(self.atol, 5e-2))

            return TestResult(
                kernel_name=kernel_name, config=config, passed=passed,
                max_diff=max_diff, duration_ms=(time.time() - start_time) * 1000
            )

        except Exception as e:
            return TestResult(
                kernel_name=kernel_name, config=config, passed=False,
                max_diff=float('nan'), error=str(e),
                duration_ms=(time.time() - start_time) * 1000
            )
    
    def test_backward_kernel(self, kernel_name: str, config: TestConfig) -> TestResult:
        """Test a single backward kernel against PyTorch autograd."""
        if kernel_name not in BACKWARD_KERNEL_NAMES:
            raise ValueError(f"Unknown backward kernel: {kernel_name}")

        kernel_idx = BACKWARD_KERNEL_NAMES.index(kernel_name)
        start_time = time.time()

        try:
            N = config.N
            inputs = self._create_inputs(config)
            grad_output = torch.randn(config.B, config.H, N, config.D, device=self.device) * config.input_scale

            # Run forward pass first to get softmax stats needed by backward
            fwd_out = self.cuda_ext.forward(
                inputs['Q'].clone().to(torch.bfloat16),
                inputs['R'].clone().to(torch.bfloat16),
                inputs['S'].clone().to(torch.bfloat16),
                inputs['Vq_1'].clone().to(torch.bfloat16),
                inputs['Vq_2'].clone().to(torch.bfloat16),
                inputs['Vr_1'].clone().to(torch.bfloat16),
                inputs['Vr_2'].clone().to(torch.bfloat16),
                inputs['Vs_1'].clone().to(torch.bfloat16),
                inputs['Vs_2'].clone().to(torch.bfloat16),
                0.0
            )
            # Unpack: Y_q, Y_r, Y_s, Y_q_, Y_r_, Y_s_, m_i, l_i, m_j, l_j, m_k, l_k
            m_i, l_i, m_j, l_j, m_k, l_k = fwd_out[6], fwd_out[7], fwd_out[8], fwd_out[9], fwd_out[10], fwd_out[11]

            cuda_grads = self.cuda_ext.backward(
                grad_output.clone(),
                inputs['Q'].clone(), inputs['R'].clone(), inputs['S'].clone(),
                inputs['Vq_1'].clone(), inputs['Vq_2'].clone(),
                inputs['Vr_1'].clone(), inputs['Vr_2'].clone(),
                inputs['Vs_1'].clone(), inputs['Vs_2'].clone(),
                m_i, l_i, m_j, l_j, m_k, l_k,
                0.0
            )

            ref_inputs = self._detach_clone_inputs(inputs, requires_grad=True)
            ref_out = self.ref_ext.forward(
                ref_inputs['Q'], ref_inputs['R'], ref_inputs['S'],
                ref_inputs['Vq_1'], ref_inputs['Vq_2'],
                ref_inputs['Vr_1'], ref_inputs['Vr_2'],
                ref_inputs['Vs_1'], ref_inputs['Vs_2'],
                0.0
            )

            Y_q, Y_r, Y_s, Y_q_, Y_r_, Y_s_ = ref_out
            T = grad_output[:, :, :N, :]

            loss = (
                (Y_q * T).sum() + (Y_r * T).sum() + (Y_s * T).sum() +
                (Y_q_ * T).sum() + (Y_r_ * T).sum() + (Y_s_ * T).sum()
            )
            loss.backward()

            ref_grad_map = {
                'grad_Q': ref_inputs['Q'].grad,
                'grad_R': ref_inputs['R'].grad,
                'grad_S': ref_inputs['S'].grad,
                'grad_Vq_1': ref_inputs['Vq_1'].grad,
                'grad_Vq_2': ref_inputs['Vq_2'].grad,
                'grad_Vr_1': ref_inputs['Vr_1'].grad,
                'grad_Vr_2': ref_inputs['Vr_2'].grad,
                'grad_Vs_1': ref_inputs['Vs_1'].grad,
                'grad_Vs_2': ref_inputs['Vs_2'].grad,
            }

            cuda_tensor = cuda_grads[kernel_idx]
            ref_tensor = ref_grad_map[kernel_name]

            if cuda_tensor is None or ref_tensor is None:
                return TestResult(
                    kernel_name=kernel_name, config=config, passed=False,
                    max_diff=float('nan'), error="One of the gradients is None",
                    duration_ms=(time.time() - start_time) * 1000
                )

            if cuda_tensor.shape != ref_tensor.shape:
                return TestResult(
                    kernel_name=kernel_name, config=config, passed=False,
                    max_diff=float('nan'),
                    error=f"Shape mismatch: {cuda_tensor.shape} vs {ref_tensor.shape}",
                    duration_ms=(time.time() - start_time) * 1000
                )

            if torch.isnan(cuda_tensor).any() or torch.isinf(cuda_tensor).any():
                return TestResult(
                    kernel_name=kernel_name, config=config, passed=False,
                    max_diff=float('nan'), error="CUDA gradient contains NaN or Inf",
                    duration_ms=(time.time() - start_time) * 1000
                )

            max_diff = (cuda_tensor - ref_tensor).abs().max().item()
            passed = torch.allclose(
                cuda_tensor,
                ref_tensor,
                rtol=max(self.rtol, 2e-2),
                atol=max(self.atol, 2e-2),
            )

            return TestResult(
                kernel_name=kernel_name, config=config, passed=passed,
                max_diff=max_diff, duration_ms=(time.time() - start_time) * 1000
            )

        except RuntimeError as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower():
                error_msg = "OOM"
            return TestResult(
                kernel_name=kernel_name, config=config, passed=False,
                max_diff=float('nan'), error=error_msg,
                duration_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            return TestResult(
                kernel_name=kernel_name, config=config, passed=False,
                max_diff=float('nan'), error=str(e),
                duration_ms=(time.time() - start_time) * 1000
            )
    
    def run_tests(
        self,
        configs: List[TestConfig],
        kernel_names: Optional[List[str]] = None,
        forward_only: bool = False,
        backward_only: bool = False,
        continue_on_failure: bool = True,
        max_n_for_backward: int = 256
    ) -> List[TestResult]:
        """Run tests across all specified configurations and kernels."""
        if kernel_names is None:
            if forward_only:
                kernel_names = FORWARD_KERNEL_NAMES
            elif backward_only:
                kernel_names = BACKWARD_KERNEL_NAMES
            else:
                kernel_names = ALL_KERNEL_NAMES

        forward_kernels = [k for k in kernel_names if k in FORWARD_KERNEL_NAMES]
        backward_kernels = [k for k in kernel_names if k in BACKWARD_KERNEL_NAMES]

        results = []
        total_tests = len(configs) * len(kernel_names)
        test_idx = 0

        for config in configs:
            for kernel_name in forward_kernels:
                test_idx += 1
                if self.verbose:
                    print(f"[{test_idx}/{total_tests}] {kernel_name} on {config.name}...", end=" ", flush=True)

                torch.cuda.empty_cache()
                result = self.test_forward_kernel(kernel_name, config)
                results.append(result)

                if self.verbose:
                    status = "PASS" if result.passed else "FAIL"
                    diff_str = f"{result.max_diff:.2e}" if not result.error else f"ERROR: {result.error[:40]}"
                    print(f"{status} | diff={diff_str}")
                else:
                    print("." if result.passed else "x", end="", flush=True)

                if not result.passed and not continue_on_failure:
                    return results

            if config.N <= max_n_for_backward:
                for kernel_name in backward_kernels:
                    test_idx += 1
                    if self.verbose:
                        print(f"[{test_idx}/{total_tests}] {kernel_name} on {config.name}...", end=" ", flush=True)

                    torch.cuda.empty_cache()
                    result = self.test_backward_kernel(kernel_name, config)
                    results.append(result)

                    if self.verbose:
                        status = "PASS" if result.passed else "FAIL"
                        diff_str = f"{result.max_diff:.2e}" if not result.error else f"ERROR: {result.error[:40]}"
                        print(f"{status} | diff={diff_str}")
                    else:
                        print("." if result.passed else "x", end="", flush=True)

                    if not result.passed and not continue_on_failure:
                        return results
            else:
                if self.verbose:
                    print(f"  Skipping backward tests for N={config.N} (> {max_n_for_backward})")
                test_idx += len(backward_kernels)

        if not self.verbose:
            print()
        return results


def print_summary(results: List[TestResult], verbose: bool = False):
    """Print test summary."""
    if not results:
        print("No test results to report.")
        return

    kernel_results = defaultdict(list)
    for r in results:
        kernel_results[r.kernel_name].append(r)

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed

    print("\n" + "=" * 80)
    print("KERNEL TEST SUMMARY")
    print("=" * 80)

    print(f"\n  Total: {total}  |  Passed: {passed} ({100*passed/total:.1f}%)  |  Failed: {failed}")

    print(f"\n{'PER-KERNEL BREAKDOWN':^80}")
    print("-" * 80)
    print(f"  {'Kernel':<12} | {'Pass':>5} | {'Fail':>5} | {'Rate':>7} | {'Max Diff':>12} | {'Status'}")
    print(f"  {'-'*12}-+-{'-'*5}-+-{'-'*5}-+-{'-'*7}-+-{'-'*12}-+-{'-'*15}")

    forward_pass = forward_fail = 0
    backward_pass = backward_fail = 0

    for kernel_name in ALL_KERNEL_NAMES:
        if kernel_name not in kernel_results:
            continue

        kresults = kernel_results[kernel_name]
        kpassed = sum(1 for r in kresults if r.passed)
        kfailed = len(kresults) - kpassed
        ktotal = len(kresults)

        if kernel_name in FORWARD_KERNEL_NAMES:
            forward_pass += kpassed
            forward_fail += kfailed
        else:
            backward_pass += kpassed
            backward_fail += kfailed

        rate = 100 * kpassed / ktotal if ktotal > 0 else 0
        max_diff = max((r.max_diff for r in kresults if r.passed), default=float('nan'))

        if kfailed == 0:
            status = "OK"
        elif kpassed == 0:
            status = "ALL FAIL"
        else:
            status = "PARTIAL"

        max_diff_str = f"{max_diff:.2e}" if max_diff == max_diff else "N/A"
        print(f"  {kernel_name:<12} | {kpassed:>5} | {kfailed:>5} | {rate:>5.0f}%  | {max_diff_str:>12} | {status}")

    print(f"  {'-'*12}-+-{'-'*5}-+-{'-'*5}-+-{'-'*7}-+-{'-'*12}-+-{'-'*15}")

    forward_total = forward_pass + forward_fail
    backward_total = backward_pass + backward_fail
    forward_rate = 100 * forward_pass / forward_total if forward_total > 0 else 0
    backward_rate = 100 * backward_pass / backward_total if backward_total > 0 else 0

    print(f"  {'FORWARD':<12} | {forward_pass:>5} | {forward_fail:>5} | {forward_rate:>5.0f}%  | {'':>12} | {'OK' if forward_fail == 0 else 'ISSUES'}")
    print(f"  {'BACKWARD':<12} | {backward_pass:>5} | {backward_fail:>5} | {backward_rate:>5.0f}%  | {'':>12} | {'OK' if backward_fail == 0 else 'ISSUES'}")

    failures = [r for r in results if not r.passed]
    if failures:
        print(f"\n{'FAILURE ANALYSIS':^80}")
        print("-" * 80)

        oom_failures = [r for r in failures if r.error and "OOM" in r.error]
        nan_failures = [r for r in failures if r.error and "NaN" in r.error]
        precision_failures = [r for r in failures if r.error is None]
        other_failures = [r for r in failures if r not in oom_failures + nan_failures + precision_failures]

        print(f"  OOM: {len(oom_failures)}  |  NaN/Inf: {len(nan_failures)}  |  Precision: {len(precision_failures)}  |  Other: {len(other_failures)}")

        if verbose and failures:
            print(f"\n  Failed tests (first 20):")
            for r in failures[:20]:
                diff_str = f"{r.max_diff:.2e}" if r.max_diff == r.max_diff else "N/A"
                print(f"    {r.kernel_name:<12} | {r.config.name:<25} | diff={diff_str}")
                if r.error:
                    print(f"      Error: {r.error[:60]}")
            if len(failures) > 20:
                print(f"    ... and {len(failures) - 20} more")

        print(f"\n  By N:")
        n_failures = defaultdict(int)
        n_totals = defaultdict(int)
        for r in results:
            n_failures[r.config.N] += 0 if r.passed else 1
            n_totals[r.config.N] += 1
        for n in sorted(n_totals.keys()):
            fails = n_failures[n]
            total = n_totals[n]
            if fails > 0:
                print(f"    N={n:>3}: {total-fails}/{total} passed")

        print(f"\n  By D:")
        d_failures = defaultdict(int)
        d_totals = defaultdict(int)
        for r in results:
            d_failures[r.config.D] += 0 if r.passed else 1
            d_totals[r.config.D] += 1
        for d in sorted(d_totals.keys()):
            fails = d_failures[d]
            total = d_totals[d]
            if fails > 0:
                print(f"    D={d:>2}: {total-fails}/{total} passed")

        print(f"\n  By B*H:")
        bh_failures = defaultdict(int)
        bh_totals = defaultdict(int)
        for r in results:
            bh = r.config.B * r.config.H
            bh_failures[bh] += 0 if r.passed else 1
            bh_totals[bh] += 1
        for bh in sorted(bh_totals.keys()):
            fails = bh_failures[bh]
            total = bh_totals[bh]
            if fails > 0:
                print(f"    B*H={bh:>2}: {total-fails}/{total} passed")

    print("=" * 80)
    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="CUDA kernel test suite")

    parser.add_argument('--quick', action='store_true', help='Quick smoke tests only')
    parser.add_argument('--large', action='store_true', help='Include large N configs')
    parser.add_argument('--stress', action='store_true', help='Include stress tests')
    parser.add_argument('--edge', action='store_true', help='Include edge cases')
    parser.add_argument('--all', action='store_true', help='Run all configurations')

    parser.add_argument('--forward-only', action='store_true', help='Forward kernels only')
    parser.add_argument('--backward-only', action='store_true', help='Backward kernels only')
    parser.add_argument('--kernels', nargs='+', choices=ALL_KERNEL_NAMES, help='Specific kernels')

    parser.add_argument('--rtol', type=float, default=1e-4, help='Relative tolerance')
    parser.add_argument('--atol', type=float, default=1e-5, help='Absolute tolerance')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--continue-on-failure', action='store_true', help='Continue after failures')
    parser.add_argument('--max-n-backward', type=int, default=256, help='Max N for backward tests')

    parser.add_argument('--filter', type=str, default=None, help='Filter configs by name substring')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--json', type=str, default=None, help='Save results to JSON')

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("WARNING: CUDA not available")

    if args.quick:
        configs = get_test_configs(quick=True, standard=False)
    elif args.all:
        configs = get_test_configs(quick=True, standard=True, large=True, stress=True, edge=True)
    else:
        configs = get_test_configs(
            quick=True, standard=True,
            large=args.large, stress=args.stress, edge=args.edge
        )

    if args.filter:
        configs = [c for c in configs if args.filter.lower() in c.name.lower()]
        if not configs:
            print(f"No configs match filter '{args.filter}'")
            sys.exit(1)

    print(f"Configs: {len(configs)} | Device: {device} | rtol={args.rtol}, atol={args.atol}")
    print("-" * 80)

    tester = KernelTester(device=device, rtol=args.rtol, atol=args.atol, verbose=args.verbose)

    results = tester.run_tests(
        configs=configs,
        kernel_names=args.kernels,
        forward_only=args.forward_only,
        backward_only=args.backward_only,
        continue_on_failure=args.continue_on_failure or True,
        max_n_for_backward=args.max_n_backward
    )

    all_passed = print_summary(results, verbose=args.verbose)

    if args.json:
        import json
        json_results = [
            {
                'kernel': r.kernel_name,
                'config': r.config.name,
                'passed': r.passed,
                'max_diff': r.max_diff if r.max_diff == r.max_diff else None,
                'error': r.error,
                'duration_ms': r.duration_ms
            }
            for r in results
        ]
        with open(args.json, 'w') as f:
            json.dump({
                'summary': {
                    'total': len(results),
                    'passed': sum(1 for r in results if r.passed),
                    'failed': sum(1 for r in results if not r.passed)
                },
                'results': json_results
            }, f, indent=2)
        print(f"\nResults saved to {args.json}")

    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
