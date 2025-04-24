import torch
import time
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import hyper_attn_cpp_reference
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure the C++ extensions are compiled correctly.")
    sys.exit(1)

def benchmark_comparison(configs, num_runs=5, device_cpu='cpu', device_cuda='cuda'):
    """Run benchmarks comparing PyTorch vs CUDA implementations on realistic configurations"""
    
    results = []
    
    for config in configs:
        B, H, I, J, K, D = config
        print(f"\nTesting config: B={B}, H={H}, I={I}, J={J}, K={K}, D={D}")
        
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
        if torch.cuda.is_available():
            cuda_inputs = [
                t.to(device_cuda) if isinstance(t, torch.Tensor) else t
                for t in cpu_inputs
            ]
        
        # -- Benchmark PyTorch Reference (CPU) --
        with torch.no_grad():
            # Warmup
            _ = hyper_attn_cpp_reference.forward(*cpu_inputs)
            
            # Timing
            torch_times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                torch_outputs = hyper_attn_cpp_reference.forward(*cpu_inputs)
                torch_times.append(time.perf_counter() - start)
            
            torch_time_ms = np.mean(torch_times) * 1000
            print(f"PyTorch CPU: {torch_time_ms:.2f} ms")
        
        # -- Benchmark CUDA (if available) --
        cuda_time_ms = None
        cuda_outputs = None
        if torch.cuda.is_available():
            with torch.no_grad():
                # Warmup
                _ = hyper_attn_cpp_reference.forward(*cuda_inputs)
                torch.cuda.synchronize()
                
                # Timing
                cuda_times = []
                for _ in range(num_runs):
                    start = time.perf_counter()
                    cuda_outputs = hyper_attn_cpp_reference.forward(*cuda_inputs)
                    torch.cuda.synchronize()
                    cuda_times.append(time.perf_counter() - start)
                
                cuda_time_ms = np.mean(cuda_times) * 1000
                print(f"CUDA: {cuda_time_ms:.2f} ms")
        
        # -- Check numerical correctness --
        if torch.cuda.is_available() and cuda_outputs is not None:
            with torch.no_grad():
                # Compare PyTorch (CPU) vs CUDA
                cuda_match = True
                max_diff_cuda = 0.0
                
                for i, (ref_tensor, cuda_tensor) in enumerate(zip(torch_outputs, cuda_outputs)):
                    cuda_tensor_cpu = cuda_tensor.cpu()  # Move to CPU for comparison
                    if not torch.allclose(ref_tensor, cuda_tensor_cpu, rtol=1e-4, atol=1e-4):
                        cuda_match = False
                        this_diff = torch.max(torch.abs(ref_tensor - cuda_tensor_cpu)).item()
                        max_diff_cuda = max(max_diff_cuda, this_diff)
                
                print(f"CPU PyTorch vs CUDA: {'MATCH' if cuda_match else f'DIFFER (max diff: {max_diff_cuda:.6f})'}")
        
        # Record results
        result = {
            'Config': f"B={B},H={H},I={I},J={J},K={K},D={D}",
            'Total Elements': B * H * I * J * K * D,
            'PyTorch CPU (ms)': torch_time_ms,
            'CUDA (ms)': cuda_time_ms if cuda_time_ms is not None else 'N/A',
            'Speedup (CUDA/PyTorch)': (torch_time_ms / cuda_time_ms) if cuda_time_ms is not None else 'N/A',
            'CUDA Match': 'Yes' if torch.cuda.is_available() and cuda_match else f'No ({max_diff_cuda:.6f})' if torch.cuda.is_available() else 'N/A'
        }
        results.append(result)
        
        # Clean up memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Create DataFrame and display results
    df = pd.DataFrame(results)
    print("\n===== BENCHMARK RESULTS =====")
    print(df.to_string(index=False))
    
    return df

if __name__ == "__main__":
    # Define realistic configurations to test
    configs = [
        # (B, H, I, J, K, D)
        (1, 4, 16, 16, 16, 32),   # Small
        (2, 8, 32, 32, 32, 64),   # Medium
        (4, 8, 64, 64, 64, 64),   # Large
        (8, 16, 32, 32, 32, 128), # High-dimensional
        (2, 8, 128, 128, 128, 32) # Long sequences
    ]
    
    # Run the benchmark
    df = benchmark_comparison(configs, num_runs=3) 