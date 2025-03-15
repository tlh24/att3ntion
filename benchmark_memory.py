import torch
import time
import hyper_attn_cpp
import hyper_attn_pytorch
import self_attn_pytorch
import gc
import tracemalloc  # Add tracemalloc for CPU memory tracking

# Remove debug print - no longer needed
# print("Available in hyper_attn_cpp module:", dir(hyper_attn_cpp))

def measure_memory(func, *args, **kwargs):
    """Measure peak memory usage and execution time of a function."""
    # Clear cache and collect garbage
    torch.cuda.empty_cache()
    gc.collect()
    
    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()
    
    # Start CPU memory tracking
    tracemalloc.start()
    
    # Record starting memory
    start_mem = torch.cuda.memory_allocated()
    start_cpu_mem = tracemalloc.get_traced_memory()[0]
    
    # Time execution
    start_time = time.time()
    output = func(*args, **kwargs)
    end_time = time.time()
    
    # Record peak memory
    peak_mem = torch.cuda.max_memory_allocated()
    curr_mem = torch.cuda.memory_allocated()
    
    # Get CPU memory stats
    current_cpu_mem, peak_cpu_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Memory actually used during computation
    used_mem = peak_mem - start_mem
    used_cpu_mem = peak_cpu_mem - start_cpu_mem
    
    return {
        "output": output,
        "execution_time": end_time - start_time,
        "peak_memory": peak_mem / (1024 ** 2),    # MB
        "used_memory": used_mem / (1024 ** 2),    # MB
        "peak_cpu_memory": peak_cpu_mem / (1024 ** 2)    # MB
    }


def benchmark_implementations(batch_size=1, seq_len=64, d_model=256, n_heads=4):
    """Benchmark implementations with the same input."""
    # Create random input tensor
    x = torch.randn(batch_size, seq_len, d_model, device="cuda")
    
    results = {}
    
    # Measure Self-Attention implementation (first)
    self_attn_model = self_attn_pytorch.SelfAttention(d_model, n_heads).cuda()
    with torch.no_grad():
        results["self attention pytorch"] = measure_memory(self_attn_model.forward, x)
    
    # Measure PyTorch implementation of Hypergraph Attention (second)
    torch_model = hyper_attn_pytorch.HypergraphAttention(d_model, n_heads).cuda()
    with torch.no_grad():
        results["3-attention pytorch"] = measure_memory(torch_model.forward, x)
    
    # Measure C++ implementation (third)
    # Now directly call the forward function with the right parameters
    with torch.no_grad():
        results["3-attention c++"] = measure_memory(
            lambda x: hyper_attn_cpp.forward(x, d_model, n_heads), 
            x
        )
    
    return results


def print_results_table(results, config):
    """Print results in a concise table format."""
    bs, seq_len, dim, heads = config
    
    print(f"\n----- batch={bs}, seq_len={seq_len}, dim={dim}, heads={heads} -----")
    
    # Print header
    print(f"{'Implementation':<22} {'Time (s)':<10} {'GPU Mem (MB)':<15} {'CPU Mem (MB)':<15}")
    print("-" * 62)
    
    # Print results in the specified order
    ordered_keys = ["self attention pytorch", "3-attention pytorch", "3-attention c++"]
    for name in ordered_keys:
        if name in results:
            result = results[name]
            print(f"{name:<22} {result['execution_time']:.4f}     {result['used_memory']:.2f}         {result['peak_cpu_memory']:.2f}")
    
    # Print comparisons
    if "3-attention pytorch" in results and "self attention pytorch" in results:
        hyper = results["3-attention pytorch"]
        self_attn = results["self attention pytorch"]
        
        time_diff = (1 - self_attn["execution_time"] / hyper["execution_time"]) * 100
        mem_diff = (1 - self_attn["used_memory"] / hyper["used_memory"]) * 100
        
        print("\nself attention pytorch vs 3-attention pytorch:")
        print(f"Time: {'faster' if time_diff > 0 else 'slower'} by {abs(time_diff):.1f}%")
        print(f"Memory: {'less' if mem_diff > 0 else 'more'} by {abs(mem_diff):.1f}%")
    
    if "3-attention pytorch" in results and "3-attention c++" in results:
        hyper = results["3-attention pytorch"]
        cpp = results["3-attention c++"]
        
        time_diff = (1 - cpp["execution_time"] / hyper["execution_time"]) * 100
        mem_diff = (1 - cpp["used_memory"] / hyper["used_memory"]) * 100
        
        print("\n3-attention c++ vs 3-attention pytorch:")
        print(f"Time: {'faster' if time_diff > 0 else 'slower'} by {abs(time_diff):.1f}%")
        print(f"Memory: {'less' if mem_diff > 0 else 'more'} by {abs(mem_diff):.1f}%")


if __name__ == "__main__":
    # Test with smaller sizes to avoid CUDA OOM
    sizes = [
        (1, 32, 128, 2),     # Tiny
        (1, 64, 256, 4),     # Small
        (2, 64, 256, 4),     # Medium
    ]
    
    print("=== Memory Benchmark for Attention Mechanisms ===")
    
    for config in sizes:
        try:
            results = benchmark_implementations(*config)
            print_results_table(results, config)
        except RuntimeError as e:
            error_msg = str(e).split('.')[0]
            print(f"Error with batch={config[0]}, seq_len={config[1]}: {error_msg}...")
            print("Skipping to next size...") 