# tests/backward_cuda_comparison.py
import torch
import time
import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

try:
    import hyper_attn_cpp_manual
    print("Successfully imported hyper_attn_cpp_manual")
except ImportError as e:
    print(f"Error: Could not import 'hyper_attn_cpp_manual': {e}")
    print("Please ensure the extension is compiled via 'python setup.py develop'")
    sys.exit(1)

def get_grad_output(Y_q, Y_r, Y_s, Y_q_, Y_r_, Y_s_):
    """Create gradient tensor for backward pass"""
    B, H, I, D = Y_q.shape
    _, _, J, _ = Y_r.shape
    _, _, K, _ = Y_s.shape
    
    max_len = max(I, J, K)
    grad_output = torch.ones(B, H, max_len, D, device=Y_q.device, dtype=Y_q.dtype)
    
    return grad_output

def benchmark_with_memory(B, H, I, J, K, D, num_iterations=10):
    """Benchmark forward and backward passes with memory tracking"""
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return None
    
    dropout_rate = 0.0
    
    try:
        # Initialize tensors on GPU
        Q = torch.rand(B, H, I, D, device='cuda', dtype=torch.float32)
        R = torch.rand(B, H, J, D, device='cuda', dtype=torch.float32)
        S = torch.rand(B, H, K, D, device='cuda', dtype=torch.float32)
        Vq_1 = torch.rand(B, H, I, D, device='cuda', dtype=torch.float32)
        Vq_2 = torch.rand(B, H, I, D, device='cuda', dtype=torch.float32)
        Vr_1 = torch.rand(B, H, J, D, device='cuda', dtype=torch.float32)
        Vr_2 = torch.rand(B, H, J, D, device='cuda', dtype=torch.float32)
        Vs_1 = torch.rand(B, H, K, D, device='cuda', dtype=torch.float32)
        Vs_2 = torch.rand(B, H, K, D, device='cuda', dtype=torch.float32)
        
        torch.cuda.synchronize()
        
        # Measure FORWARD pass memory
        torch.cuda.reset_peak_memory_stats()
        forward_times = []
        
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            Y_q, Y_r, Y_s, Y_q_, Y_r_, Y_s_ = hyper_attn_cpp_manual.forward(
                Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2, dropout_rate
            )
            
            torch.cuda.synchronize()
            forward_times.append(time.perf_counter() - start)
        
        forward_peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        forward_mean_time = sum(forward_times) / len(forward_times) * 1000  # ms
        
        # Measure BACKWARD pass memory
        torch.cuda.reset_peak_memory_stats()
        backward_times = []
        backward_success = True
        backward_peak_mem = 0
        backward_mean_time = 0
        
        try:
            for _ in range(num_iterations):
                # Run forward pass (not timed)
                Y_q, Y_r, Y_s, Y_q_, Y_r_, Y_s_ = hyper_attn_cpp_manual.forward(
                    Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2, dropout_rate
                )
                grad_output = get_grad_output(Y_q, Y_r, Y_s, Y_q_, Y_r_, Y_s_)
                
                # Time only backward pass
                torch.cuda.synchronize()
                start = time.perf_counter()
                
                hyper_attn_cpp_manual.backward(
                    grad_output, Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2, dropout_rate
                )
                
                torch.cuda.synchronize()
                backward_times.append(time.perf_counter() - start)
            
            backward_peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
            backward_mean_time = sum(backward_times) / len(backward_times) * 1000  # ms
            
        except torch.cuda.OutOfMemoryError:
            backward_success = False
        
        # Clean up
        del Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2
        if 'Y_q' in locals():
            del Y_q, Y_r, Y_s, Y_q_, Y_r_, Y_s_
        if 'grad_output' in locals():
            del grad_output
        torch.cuda.empty_cache()
        
        return {
            'forward_time': forward_mean_time,
            'forward_mem': forward_peak_mem,
            'backward_time': backward_mean_time if backward_success else None,
            'backward_mem': backward_peak_mem if backward_success else None,
            'backward_success': backward_success
        }
        
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {
            'forward_time': None,
            'forward_mem': None,
            'backward_time': None,
            'backward_mem': None,
            'backward_success': False,
            'forward_oom': True
        }
    except Exception as e:
        torch.cuda.empty_cache()
        print(f"Error: {e}")
        return None

def main():
    print("\n" + "=" * 100)
    print("FORWARD vs BACKWARD: TIME & MEMORY COMPARISON")
    print("=" * 100)
    
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        return
    
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"Total GPU Memory: {total_mem:.2f} GB")
    print(f"Number of benchmark iterations per test: 10")
    
    # Fixed parameters
    B, H, D = 8, 4, 32
    
    # Progressive N values - start small and increase until OOM
    N_values = [16, 32, 64, 96, 128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 
                1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096]
    
    print(f"\nFixed Config: B={B}, H={H}, D={D}")
    print(f"Testing progressive N values (I=J=K=N) until OOM...")
    
    print("\n" + "-" * 100)
    header = f"{'N':<8} | {'Fwd Time':<12} | {'Bwd Time':<12} | {'Fwd Mem':<12} | {'Bwd Mem':<12} | {'Mem Ratio':<10} | {'Status':<15}"
    print(header)
    print("-" * 100)
    
    results = []
    forward_oom_N = None
    backward_oom_N = None
    
    for N in N_values:
        result = benchmark_with_memory(B, H, N, N, N, D)
        
        if result is None:
            print(f"{N:<8} | {'Error':<12} | {'Error':<12} | {'Error':<12} | {'Error':<12} | {'N/A':<10} | {'Error':<15}")
            break
        
        # Check if forward pass OOM
        if result.get('forward_oom', False):
            forward_oom_N = N
            print(f"{N:<8} | {'OOM':<12} | {'N/A':<12} | {'N/A':<12} | {'N/A':<12} | {'N/A':<10} | {'Fwd OOM':<15}")
            break
        
        # Format output
        fwd_time_str = f"{result['forward_time']:.2f} ms" if result['forward_time'] else "N/A"
        fwd_mem_str = f"{result['forward_mem']:.1f} MB" if result['forward_mem'] else "N/A"
        
        if result['backward_success']:
            bwd_time_str = f"{result['backward_time']:.2f} ms"
            bwd_mem_str = f"{result['backward_mem']:.1f} MB"
            mem_ratio = result['backward_mem'] / result['forward_mem'] if result['forward_mem'] > 0 else 0
            mem_ratio_str = f"{mem_ratio:.2f}x"
            status = "OK"
        else:
            backward_oom_N = N
            bwd_time_str = "OOM"
            bwd_mem_str = "OOM"
            mem_ratio_str = "N/A"
            status = "Bwd OOM"
        
        print(f"{N:<8} | {fwd_time_str:<12} | {bwd_time_str:<12} | {fwd_mem_str:<12} | {bwd_mem_str:<12} | {mem_ratio_str:<10} | {status:<15}")
        
        results.append({
            'N': N,
            'result': result
        })
        
        # Stop if backward OOM
        if not result['backward_success']:
            break
    
    print("-" * 100)
    
    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    if backward_oom_N:
        print(f"✗ Backward pass OOM at N = {backward_oom_N}")
        successful_results = [r for r in results if r['result']['backward_success']]
        if successful_results:
            max_successful_N = successful_results[-1]['N']
            print(f"✓ Maximum successful N for backward: {max_successful_N}")
    else:
        print(f"✓ All tested sizes completed successfully (up to N = {N_values[len(results)-1]})")
    
    if forward_oom_N:
        print(f"✗ Forward pass OOM at N = {forward_oom_N}")
    
    # Memory ratio analysis for successful runs
    successful_with_backward = [r for r in results if r['result']['backward_success']]
    if successful_with_backward:
        print("\nMemory Usage Analysis (successful runs):")
        mem_ratios = [r['result']['backward_mem'] / r['result']['forward_mem'] 
                     for r in successful_with_backward]
        avg_mem_ratio = sum(mem_ratios) / len(mem_ratios)
        min_mem_ratio = min(mem_ratios)
        max_mem_ratio = max(mem_ratios)
        
        print(f"  Average Backward/Forward Memory Ratio: {avg_mem_ratio:.2f}x")
        print(f"  Min Memory Ratio: {min_mem_ratio:.2f}x")
        print(f"  Max Memory Ratio: {max_mem_ratio:.2f}x")
        
        if avg_mem_ratio > 3:
            print(f"  ⚠ Backward uses {avg_mem_ratio:.1f}x more memory than forward (high)")
        elif avg_mem_ratio > 2:
            print(f"  ~ Backward uses {avg_mem_ratio:.1f}x more memory than forward (moderate)")
        else:
            print(f"  ✓ Backward uses {avg_mem_ratio:.1f}x more memory than forward (good)")

if __name__ == '__main__':
    main()