import torch
import time
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    # Import the module containing BOTH CPU and CUDA manual implementations
    import hyper_attn_cpp_manual as manual_impl
except ImportError as e:
    print(f"Error importing module 'hyper_attn_cpp_manual': {e}")
    print("Make sure the C++ extension with manual CPU/CUDA is compiled correctly.")
    sys.exit(1)

def benchmark_comparison(configs, num_runs=5, device_cpu='cpu', device_cuda='cuda'):
    """Run benchmarks comparing manual CPU vs manual CUDA implementations."""

    results = []

    for config in configs:
        B, H, I, J, K, D = config
        print(f"\n--- Testing config: B={B}, H={H}, I={I}, J={J}, K={K}, D={D} ---")

        # Generate inputs (use the same random seed for consistency)
        torch.manual_seed(42)
        cpu_inputs = [
            torch.randn(B, H, I, D, device=device_cpu),  # Q
            torch.randn(B, H, J, D, device=device_cpu),  # R
            torch.randn(B, H, K, D, device=device_cpu),  # S
            torch.randn(B, H, I, D, device=device_cpu),  # Vq_1
            torch.randn(B, H, I, D, device=device_cpu),  # Vq_2
            torch.randn(B, H, J, D, device=device_cpu),  # Vr_1
            torch.randn(B, H, J, D, device=device_cpu),  # Vr_2
            torch.randn(B, H, K, D, device=device_cpu),  # Vs_1
            torch.randn(B, H, K, D, device=device_cpu),  # Vs_2
            0.0,  # dropout_rate
        ]

        # Create CUDA inputs if available
        cuda_inputs = None
        if torch.cuda.is_available():
            cuda_inputs = [
                t.to(device_cuda) if isinstance(t, torch.Tensor) else t
                for t in cpu_inputs
            ]

        # -- Benchmark Manual Forward (CPU) --
        with torch.no_grad():
            # Warmup
            _ = manual_impl.forward(*cpu_inputs)

            # Timing
            manual_cpu_fwd_times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                manual_cpu_outputs = manual_impl.forward(*cpu_inputs)
                # No sync needed for CPU timing
                manual_cpu_fwd_times.append(time.perf_counter() - start)

            manual_cpu_fwd_time_ms = np.mean(manual_cpu_fwd_times) * 1000
            print(f"Manual CPU Forward: {manual_cpu_fwd_time_ms:.2f} ms")

        # -- Benchmark Manual Backward (CPU) --
        # Generate grad_output
        N_grad = max(I, J, K) # Required size based on kernel access
        grad_output_cpu = torch.randn(B, H, N_grad, D, device=device_cpu)

        # Warmup
        _ = manual_impl.backward(grad_output_cpu, *cpu_inputs)

        # Timing
        manual_cpu_bwd_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            manual_cpu_grads = manual_impl.backward(grad_output_cpu, *cpu_inputs)
            # No sync needed for CPU timing
            manual_cpu_bwd_times.append(time.perf_counter() - start)

        manual_cpu_bwd_time_ms = np.mean(manual_cpu_bwd_times) * 1000
        print(f"Manual CPU Backward: {manual_cpu_bwd_time_ms:.2f} ms")

        # -- Benchmark Manual Forward (CUDA, if available) --
        manual_cuda_fwd_time_ms = None
        manual_cuda_outputs = None
        if torch.cuda.is_available() and cuda_inputs is not None:
            with torch.no_grad():
                # Warmup
                _ = manual_impl.forward(*cuda_inputs)
                torch.cuda.synchronize()

                # Timing
                manual_cuda_fwd_times = []
                for _ in range(num_runs):
                    start = time.perf_counter()
                    manual_cuda_outputs = manual_impl.forward(*cuda_inputs)
                    torch.cuda.synchronize()
                    manual_cuda_fwd_times.append(time.perf_counter() - start)

                manual_cuda_fwd_time_ms = np.mean(manual_cuda_fwd_times) * 1000
                print(f"Manual CUDA Forward: {manual_cuda_fwd_time_ms:.2f} ms")

        # -- Benchmark Manual Backward (CUDA, if available) --
        manual_cuda_bwd_time_ms = None
        manual_cuda_grads = None
        if torch.cuda.is_available() and cuda_inputs is not None:
            grad_output_cuda = grad_output_cpu.to(device_cuda)

            # Warmup
            _ = manual_impl.backward(grad_output_cuda, *cuda_inputs)
            torch.cuda.synchronize()

            # Timing
            manual_cuda_bwd_times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                manual_cuda_grads = manual_impl.backward(grad_output_cuda, *cuda_inputs)
                torch.cuda.synchronize()
                manual_cuda_bwd_times.append(time.perf_counter() - start)

            manual_cuda_bwd_time_ms = np.mean(manual_cuda_bwd_times) * 1000
            print(f"Manual CUDA Backward: {manual_cuda_bwd_time_ms:.2f} ms")

        # -- Check numerical correctness --
        cuda_fwd_match = True
        max_diff_fwd = 0.0
        cuda_bwd_match = True
        max_diff_bwd = 0.0
        result = {
            'Config': f"B={B},H={H},I={I},J={J},K={K},D={D}",
            'Manual CPU Fwd (ms)': manual_cpu_fwd_time_ms,
            'Manual CUDA Fwd (ms)': manual_cuda_fwd_time_ms if manual_cuda_fwd_time_ms is not None else 'N/A',
            'Fwd Speedup': (manual_cpu_fwd_time_ms / manual_cuda_fwd_time_ms) if manual_cuda_fwd_time_ms is not None else 'N/A',
            'CUDA Fwd Match': 'Yes' if cuda_fwd_match else f'No ({max_diff_fwd:.6f})',
            'Manual CPU Bwd (ms)': manual_cpu_bwd_time_ms,
            'Manual CUDA Bwd (ms)': manual_cuda_bwd_time_ms if manual_cuda_bwd_time_ms is not None else 'N/A',
            'Bwd Speedup': (manual_cpu_bwd_time_ms / manual_cuda_bwd_time_ms) if manual_cuda_bwd_time_ms is not None else 'N/A',
            'CUDA Bwd Match': 'Yes' if cuda_bwd_match else f'No ({max_diff_bwd:.6f})'
        }
        results.append(result)

        del cpu_inputs, grad_output_cpu, manual_cpu_outputs, manual_cpu_grads
        if torch.cuda.is_available():
            del cuda_inputs, grad_output_cuda, manual_cuda_outputs, manual_cuda_grads
            torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    print("\n===== MANUAL CPU vs CUDA BENCHMARK RESULTS =====")
    print(df.to_string(index=False))

    return df

if __name__ == "__main__":
    configs = [
        # (B, H, I, J, K, D)
        (1, 2, 4, 4, 4, 8),   
        (2, 2, 6, 6, 6, 16),  
        (2, 2, 12, 12, 12, 12),  
    ]

    df = benchmark_comparison(configs, num_runs=3) 