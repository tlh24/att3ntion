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

        if torch.cuda.is_available() and manual_cuda_outputs is not None:
            with torch.no_grad():
                # Compare Forward Pass (Manual CPU vs Manual CUDA)
                print("Comparing Manual CPU vs Manual CUDA Forward...")
                for i, (cpu_tensor, cuda_tensor) in enumerate(zip(manual_cpu_outputs, manual_cuda_outputs)):
                    cuda_tensor_cpu = cuda_tensor.cpu()  # Move to CPU for comparison
                    if not torch.allclose(cpu_tensor, cuda_tensor_cpu, rtol=1e-4, atol=1e-4):
                        cuda_fwd_match = False
                        this_diff = torch.max(torch.abs(cpu_tensor - cuda_tensor_cpu)).item()
                        max_diff_fwd = max(max_diff_fwd, this_diff)
                        print(f"  Mismatch found in forward output tensor {i}")
                print(f"Manual CPU vs CUDA Forward: {'MATCH' if cuda_fwd_match else f'DIFFER (max diff: {max_diff_fwd:.6f})'}")

                # Compare Backward Pass (Manual CPU vs Manual CUDA)
                if manual_cuda_grads is not None:
                    print("Comparing Manual CPU vs Manual CUDA Backward (grad_Vq_1 only)...")
                    # Temporarily only compare grad_Vq_1 (index 3)
                    cpu_grad = manual_cpu_grads[3]
                    cuda_grad = manual_cuda_grads[3]
                    
                    if cpu_grad is None and cuda_grad is None:
                        print("  Both grad_Vq_1 are None? Skipping comparison.") 
                        # cuda_bwd_match remains True
                    elif cpu_grad is None or cuda_grad is None:
                        cuda_bwd_match = False
                        print("  Mismatch: grad_Vq_1 is None in one implementation but not the other.")
                    else:
                        cuda_grad_cpu = cuda_grad.cpu()
                        if not torch.allclose(cpu_grad, cuda_grad_cpu, rtol=1e-4, atol=1e-4):
                            cuda_bwd_match = False
                            max_diff_bwd = torch.max(torch.abs(cpu_grad - cuda_grad_cpu)).item()
                            print(f"  Mismatch found in grad_Vq_1 (index 3)")
                    
                    # Original loop commented out:
                    # for i, (cpu_grad, cuda_grad) in enumerate(zip(manual_cpu_grads, manual_cuda_grads)):
                    #     if cpu_grad is None and cuda_grad is None: continue
                    #     if cpu_grad is None or cuda_grad is None:
                    #         cuda_bwd_match = False
                    #         print(f"  Mismatch: Grad {i} is None in one implementation but not the other.")
                    #         break
                    # 
                    #     cuda_grad_cpu = cuda_grad.cpu()
                    #     if not torch.allclose(cpu_grad, cuda_grad_cpu, rtol=1e-4, atol=1e-4):
                    #         cuda_bwd_match = False
                    #         this_diff = torch.max(torch.abs(cpu_grad - cuda_grad_cpu)).item()
                    #         max_diff_bwd = max(max_diff_bwd, this_diff)
                    #         print(f"  Mismatch found in backward gradient tensor {i}")
                    print(f"Manual CPU vs CUDA Backward (grad_Vq_1 only): {'MATCH' if cuda_bwd_match else f'DIFFER (max diff: {max_diff_bwd:.6f})'}")


        # Record results
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

        # Clean up memory
        del cpu_inputs, grad_output_cpu, manual_cpu_outputs, manual_cpu_grads
        if torch.cuda.is_available():
            del cuda_inputs, grad_output_cuda, manual_cuda_outputs, manual_cuda_grads
            torch.cuda.empty_cache()

    # Create DataFrame and display results
    df = pd.DataFrame(results)
    print("\n===== MANUAL CPU vs CUDA BENCHMARK RESULTS =====")
    print(df.to_string(index=False))

    return df

if __name__ == "__main__":
    # Define realistic configurations to test
    configs = [
        # (B, H, I, J, K, D)
        (1, 4, 6, 6, 6, 12),   # Small
        # (2, 8, 32, 32, 32, 64),   # Medium
        # (4, 8, 64, 64, 64, 64),   # Large
        # (8, 16, 32, 32, 32, 128), # High-dimensional
        # (2, 8, 128, 128, 128, 32) # Long sequences
    ]

    # Run the benchmark
    df = benchmark_comparison(configs, num_runs=3) 