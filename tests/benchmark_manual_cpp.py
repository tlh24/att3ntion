import torch
import time
import psutil
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import hyper_attn_cpp_manual
    import hyper_attn_cpp_reference
    from hyper_attn_cpp_wrapper import HyperAttentionAutograd
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure the C++ extensions are compiled correctly.")
    sys.exit(1)

def make_inputs(B, H, L, D, seed=42, device='cpu', dtype=torch.float32, requires_grad=False):
    torch.manual_seed(seed)
    Q = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=requires_grad)
    R = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=requires_grad)
    S = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=requires_grad)
    Vq_1 = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=requires_grad)
    Vq_2 = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=requires_grad)
    Vr_1 = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=requires_grad)
    Vr_2 = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=requires_grad)
    Vs_1 = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=requires_grad)
    Vs_2 = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=requires_grad)
    return Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2

def get_memory_usage_mb():
    """Returns the current process's RSS memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2) # Convert bytes to MB

def benchmark_forward_pass(func, *args, **kwargs):
    # Warm-up run (important for GPU timing)
    _ = func(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    mem_before = get_memory_usage_mb()
    num_runs = 5  
    times = []
    
    for _ in range(num_runs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize() # Ensure GPU operations complete
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)  # ms
    
    exec_time_ms = np.mean(times)
    exec_time_std = np.std(times)

    mem_after = get_memory_usage_mb()

    mem_increase_mb = mem_after - mem_before

    # clear memory, useful for gpu
    del result
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return exec_time_ms, exec_time_std, mem_increase_mb, mem_after

def benchmark_backward_pass(func, *args, **kwargs):
    output = func(*args, **kwargs)
    
    #instantiate grad for backward pass
    if isinstance(output, tuple) or isinstance(output, list):
        grad_output = torch.ones_like(sum(output))
    else:
        grad_output = torch.ones_like(output)
    
    output = func(*args, **kwargs) 
    if isinstance(output, tuple) or isinstance(output, list):
        sum_output = sum(output)
    else:
        sum_output = output
    sum_output.backward(grad_output)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Reset gradients after warm-up
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.grad is not None:
            arg.grad.zero_() 
    
    mem_before = get_memory_usage_mb()

    num_runs = 5  
    times = []
    
    for _ in range(num_runs): #reset grads
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.grad is not None:
                arg.grad.zero_()
        
        start_time = time.perf_counter()
        
        # Forward pass
        output = func(*args, **kwargs)
        if isinstance(output, tuple) or isinstance(output, list):
            sum_output = sum(output)
        else:
            sum_output = output
        
        # Backward pass
        sum_output.backward(grad_output)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)  # ms
    
    exec_time_ms = np.mean(times)
    exec_time_std = np.std(times)

    mem_after = get_memory_usage_mb()
    mem_increase_mb = mem_after - mem_before

    # clean up 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return exec_time_ms, exec_time_std, mem_increase_mb, mem_after

def run_benchmarks():
    """Runs benchmarks for different configurations."""
    # --- Configuration ---
    configs = [
        (1, 2, 4, 4),    
        (1, 2, 8, 4),    
        (1, 2, 16, 4),   
        # (1, 2, 32, 4),   
        # (1, 2, 64, 4), #still takes long time on mac cpu  
        # (1, 2, 128, 4), # Huge 
        # (1, 2, 256, 4), # don't even think about it 
    ]
    

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        print("Using device: cpu")

    dtype = torch.float32 

    print("\n===== Benchmarking Forward Pass (Large Inputs) ====")
    forward_results = run_forward_benchmarks(configs, device, dtype)
    
    print("\n===== Benchmarking Backward Pass (Large Inputs) ====")
    backward_results = run_backward_benchmarks(configs, device, dtype)
    
    print("\n===== BENCHMARK RESULTS =====")
    print_summary(forward_results, "Forward Pass")
    print_summary(backward_results, "Backward Pass")

def run_forward_benchmarks(configs, device, dtype):
    """Runs forward pass benchmarks for different configurations."""
    results = []

    for i, config in enumerate(configs):
        B, H, L, D = config
        seed = 42 + i
        print(f"\n--- Config {i+1}/{len(configs)}: B={B}, H={H}, L={L}, D={D} ---")

        try:
            inputs = make_inputs(B, H, L, D, seed=seed, device=device, dtype=dtype)
        except Exception as e:
            print(f"ERROR generating inputs (likely OOM): {e}")
            results.append({
                'Config': f"({B},{H},{L},{D})",
                'Implementation': 'N/A',
                'Time (ms)': 'OOM',
                'Time Std (ms)': 'N/A',
                'Mem Increase (MB)': 'OOM',
                'Mem Final (MB)': 'OOM'
            })
            continue

        print("Benchmarking Reference (torch)...")
        try:
            with torch.no_grad(): # Ensure no gradients are computed
                ref_time, ref_std, ref_mem_inc, ref_mem_final = benchmark_forward_pass(
                    lambda *args: sum(hyper_attn_cpp_reference.forward(*args)), *inputs
                )
            print(f"  Ref Time: {ref_time:.2f}±{ref_std:.2f}ms, Mem Inc: {ref_mem_inc:.4f} MB")
            results.append({
                'Config': f"({B},{H},{L},{D})",
                'Implementation': 'Reference (torch)',
                'Time (ms)': f"{ref_time:.2f}",
                'Time Std (ms)': f"{ref_std:.2f}",
                'Mem Increase (MB)': f"{ref_mem_inc:.4f}",
                'Mem Final (MB)': f"{ref_mem_final:.4f}"
            })
        except Exception as e:
            print(f"  ERROR during Reference benchmark: {e}")
            results.append({
                'Config': f"({B},{H},{L},{D})",
                'Implementation': 'Reference (torch)',
                'Time (ms)': 'Error',
                'Time Std (ms)': 'N/A',
                'Mem Increase (MB)': 'Error',
                'Mem Final (MB)': f"{get_memory_usage_mb():.4f}"
            })

        # Benchmark Manual C++
        print("Benchmarking Manual (manual)...")
        try:
            with torch.no_grad():
                man_time, man_std, man_mem_inc, man_mem_final = benchmark_forward_pass(
                    hyper_attn_cpp_manual.forward, *inputs
                )
            print(f"  Manual Time: {man_time:.2f}±{man_std:.2f}ms, Mem Inc: {man_mem_inc:.4f} MB")
            
            if 'Error' not in results[-1]['Time (ms)'] and 'OOM' not in results[-1]['Time (ms)']:
                ref_time_val = float(results[-1]['Time (ms)'])
                speedup = ref_time / man_time if man_time > 0 else float('inf')
                speedup_str = f"{speedup:.2f}x"
                print(f"  Speedup: {speedup_str}")
            else:
                speedup_str = 'N/A'
                
            results.append({
                'Config': f"({B},{H},{L},{D})",
                'Implementation': 'Manual (loops)',
                'Time (ms)': f"{man_time:.2f}",
                'Time Std (ms)': f"{man_std:.2f}",
                'Mem Increase (MB)': f"{man_mem_inc:.4f}",
                'Mem Final (MB)': f"{man_mem_final:.4f}",
                'Speedup vs Ref': speedup_str
            })
        except Exception as e:
            print(f"  ERROR during Manual benchmark: {e}")
            results.append({
                'Config': f"({B},{H},{L},{D})",
                'Implementation': 'Manual (loops)',
                'Time (ms)': 'Error',
                'Time Std (ms)': 'N/A',
                'Mem Increase (MB)': 'Error',
                'Mem Final (MB)': f"{get_memory_usage_mb():.4f}",
                'Speedup vs Ref': 'N/A'
            })

        try:
            with torch.no_grad():
                ref_result = sum(hyper_attn_cpp_reference.forward(*inputs))
                man_result = hyper_attn_cpp_manual.forward(*inputs)
                
                if torch.allclose(ref_result, man_result, rtol=1e-4, atol=1e-4):
                    max_diff = torch.max(torch.abs(ref_result - man_result)).item()
                    print(f"  ✅ Results match (diff: {max_diff:.6f})")
                else:
                    max_diff = torch.max(torch.abs(ref_result - man_result)).item()
                    print(f"  ❌ Results differ (diff: {max_diff:.6f})")
        except Exception as e:
            print(f"  ERROR during result comparison: {e}")

        # Clean up inputs to free memory before next config
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results

def run_backward_benchmarks(configs, device, dtype):
    """Runs backward pass benchmarks for different configurations."""
    results = []

    for i, config in enumerate(configs):
        B, H, L, D = config
        seed = 42 + i
        print(f"\n--- Config {i+1}/{len(configs)}: B={B}, H={H}, L={L}, D={D} ---")

        try:
            # Create tensors with requires_grad=True for backward pass
            inputs = make_inputs(B, H, L, D, seed=seed, device=device, dtype=dtype, requires_grad=True)
        except Exception as e:
            print(f"ERROR generating inputs (likely OOM): {e}")
            results.append({
                'Config': f"({B},{H},{L},{D})",
                'Implementation': 'N/A',
                'Time (ms)': 'OOM',
                'Time Std (ms)': 'N/A',
                'Mem Increase (MB)': 'OOM',
                'Mem Final (MB)': 'OOM'
            })
            continue

        # Benchmark Reference (PyTorch-based C++) - Backward
        print("Benchmarking Reference (torch) backward pass...")
        try:
            ref_time, ref_std, ref_mem_inc, ref_mem_final = benchmark_backward_pass(
                hyper_attn_cpp_reference.forward, *inputs
            )
            print(f"  Ref Time: {ref_time:.2f}±{ref_std:.2f}ms, Mem Inc: {ref_mem_inc:.4f} MB")
            results.append({
                'Config': f"({B},{H},{L},{D})",
                'Implementation': 'Reference (torch)',
                'Time (ms)': f"{ref_time:.2f}",
                'Time Std (ms)': f"{ref_std:.2f}",
                'Mem Increase (MB)': f"{ref_mem_inc:.4f}",
                'Mem Final (MB)': f"{ref_mem_final:.4f}"
            })
        except Exception as e:
            print(f"  ERROR during Reference backward benchmark: {e}")
            results.append({
                'Config': f"({B},{H},{L},{D})",
                'Implementation': 'Reference (torch)',
                'Time (ms)': 'Error',
                'Time Std (ms)': 'N/A',
                'Mem Increase (MB)': 'Error',
                'Mem Final (MB)': f"{get_memory_usage_mb():.4f}"
            })

        # Reset gradients before the next benchmark
        for arg in inputs:
            if arg.grad is not None:
                arg.grad.zero_()

        # Benchmark Manual C++ - Backward
        print("Benchmarking Manual (manual) backward pass...")
        try:
            man_time, man_std, man_mem_inc, man_mem_final = benchmark_backward_pass(
                lambda *args: HyperAttentionAutograd.apply(*args), *inputs
            )
            print(f"  Manual Time: {man_time:.2f}±{man_std:.2f}ms, Mem Inc: {man_mem_inc:.4f} MB")
            
            if 'Error' not in results[-1]['Time (ms)'] and 'OOM' not in results[-1]['Time (ms)']:
                ref_time_val = float(results[-1]['Time (ms)'])
                speedup = ref_time / man_time if man_time > 0 else float('inf')
                speedup_str = f"{speedup:.2f}x"
                print(f"  Speedup: {speedup_str}")
            else:
                speedup_str = 'N/A'
                
            results.append({
                'Config': f"({B},{H},{L},{D})",
                'Implementation': 'Manual (loops)',
                'Time (ms)': f"{man_time:.2f}",
                'Time Std (ms)': f"{man_std:.2f}",
                'Mem Increase (MB)': f"{man_mem_inc:.4f}",
                'Mem Final (MB)': f"{man_mem_final:.4f}",
                'Speedup vs Ref': speedup_str
            })
        except Exception as e:
            print(f"  ERROR during Manual backward benchmark: {e}")
            results.append({
                'Config': f"({B},{H},{L},{D})",
                'Implementation': 'Manual (loops)',
                'Time (ms)': 'Error',
                'Time Std (ms)': 'N/A',
                'Mem Increase (MB)': 'Error',
                'Mem Final (MB)': f"{get_memory_usage_mb():.4f}",
                'Speedup vs Ref': 'N/A'
            })

        try:
            for arg in inputs:
                if arg.grad is not None:
                    arg.grad.zero_()
            
            ref_output = sum(hyper_attn_cpp_reference.forward(*inputs))
            ref_grad = torch.ones_like(ref_output)
            ref_output.backward(ref_grad)
            
            ref_grads = [inp.grad.clone() for inp in inputs]
            
            for arg in inputs:
                if arg.grad is not None:
                    arg.grad.zero_()
            
            man_output = HyperAttentionAutograd.apply(*inputs)
            man_grad = torch.ones_like(man_output)
            man_output.backward(man_grad)
            
            all_match = True
            max_diff = 0.0
            
            for i, (ref_g, inp) in enumerate(zip(ref_grads, inputs)):
                if not torch.allclose(ref_g, inp.grad, rtol=1e-4, atol=1e-4):
                    all_match = False
                current_diff = torch.max(torch.abs(ref_g - inp.grad)).item()
                max_diff = max(max_diff, current_diff)
            
            if all_match:
                print(f"  ✅ Gradients match (diff: {max_diff:.6f})")
            else:
                print(f"  ❌ Gradients differ (diff: {max_diff:.6f})")
                
        except Exception as e:
            print(f"  ERROR during gradient comparison: {e}")

        # Clean up inputs to free memory before next config
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results

def print_summary(results, title):
    """Prints a summary of benchmark results."""
    summary = []
    for i in range(0, len(results), 2):
        if i+1 < len(results):  # Ensure there's a pair
            config = results[i]['Config']
            ref_time = results[i]['Time (ms)']
            man_time = results[i+1]['Time (ms)']
            ref_mem = results[i]['Mem Increase (MB)']
            man_mem = results[i+1]['Mem Increase (MB)']
            speedup = results[i+1].get('Speedup vs Ref', 'N/A')
            
            summary.append({
                'Config': config,
                'Ref Time (ms)': ref_time,
                'Manual Time (ms)': man_time,
                'Speedup': speedup,
                'Ref Mem (MB)': ref_mem,
                'Manual Mem (MB)': man_mem
            })
    
    if summary:
        print(f"\n===== {title} Performance Comparison =====")
        df_summary = pd.DataFrame(summary)
        print(df_summary.to_string(index=False))

if __name__ == "__main__":
    try:
        import pandas
    except ImportError:
        print("Install pandas for formatted results.")
        exit()

    try:
        import psutil
    except ImportError:
        print("Please install psutil to track memory usage: pip install psutil")
        exit()

    run_benchmarks() 