import torch
import time
import hyper_attn_cpp
import hyper_attn_pytorch
import self_attn_pytorch
import gc
import tracemalloc  # Add tracemalloc for CPU memory tracking
import csv
import os
from datetime import datetime

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
    """Print results in a concise table format with detailed analysis."""
    bs, seq_len, dim, heads = config
    
    print(f"\n----- batch={bs}, seq_len={seq_len}, dim={dim}, heads={heads} -----")
    
    # Print header
    print(f"{'Implementation':<22} {'Time (s)':<10} {'GPU Mem (MB)':<15} {'CPU Mem (MB)':<15}")
    print("-" * 62)
    
    # Print results in the specified order with cleaner display names
    ordered_keys = ["self attention pytorch", "3-attention pytorch", "3-attention c++"]
    display_names = {
        "self attention pytorch": "self attention pytorch",
        "3-attention pytorch": "pytorch",
        "3-attention c++": "cpp"
    }
    
    # Store values for comparison
    all_times = {}
    all_memories = {}
    
    for name in ordered_keys:
        if name in results:
            result = results[name]
            display_name = display_names[name]
            print(f"{display_name:<22} {result['execution_time']:.4f}     {result['used_memory']:.2f}         {result['peak_cpu_memory']:.2f}")
            all_times[name] = result['execution_time']
            all_memories[name] = result['used_memory']
    
    # Print comparisons between PyTorch and C++ implementations
    if "3-attention pytorch" in results and "3-attention c++" in results:
        pytorch = results["3-attention pytorch"]
        cpp = results["3-attention c++"]
        
        # C++ compared to PyTorch
        time_diff_cpp_vs_pytorch = (1 - cpp["execution_time"] / pytorch["execution_time"]) * 100
        mem_diff_cpp_vs_pytorch = (1 - cpp["used_memory"] / pytorch["used_memory"]) * 100
        
        print("\ncpp vs pytorch:")
        print(f"Time: {'faster' if time_diff_cpp_vs_pytorch > 0 else 'slower'} by {abs(time_diff_cpp_vs_pytorch):.1f}%")
        print(f"Memory: {'less' if mem_diff_cpp_vs_pytorch > 0 else 'more'} by {abs(mem_diff_cpp_vs_pytorch):.1f}%")

            
        # Add visual representation of memory savings
        mem_ratio = cpp["used_memory"] / pytorch["used_memory"]
        print("\nMemory usage (cpp vs pytorch):")
        print("cpp:     " + "█" * int(mem_ratio * 40))
        print("pytorch: " + "█" * 40)
        
        # Add summary for this configuration
        print("\nConfiguration summary:")
        if time_diff_cpp_vs_pytorch > 0:
            print(f"✓ C++ implementation is {abs(time_diff_cpp_vs_pytorch):.1f}% faster than PyTorch")
        else:
            print(f"× C++ implementation is {abs(time_diff_cpp_vs_pytorch):.1f}% slower than PyTorch")
            
        print(f"✓ Memory usage is {abs(mem_diff_cpp_vs_pytorch):.1f}% less than PyTorch")
        
        # Compute the total operation count (rough approximation)
        total_ops = bs * seq_len * seq_len * seq_len * dim * heads
        ops_in_billions = total_ops / 1e9
        print(f"Total operations: ~{ops_in_billions:.2f} billion ops")


def save_results_to_csv(all_results):
    """Save benchmark results to a CSV file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_results_{timestamp}.csv"
    
    # Prepare CSV headers
    headers = [
        "Batch Size", "Sequence Length", "Hidden Dimension", "Heads",
        "Self-Attn Time (s)", "Self-Attn Memory (MB)",
        "PyTorch Time (s)", "PyTorch Memory (MB)",
        "C++ Time (s)", "C++ Memory (MB)",
        "C++ vs PyTorch Time (%)", "C++ vs PyTorch Memory (%)",
        "C++ vs Self-Attn Time (%)", "C++ vs Self-Attn Memory (%)",
        "Total Operations (B)"
    ]
    
    rows = []
    
    for config, results in all_results.items():
        bs, seq_len, dim, heads = config
        
        row = [bs, seq_len, dim, heads]
        
        # Add Self-Attention results
        if "self attention pytorch" in results:
            self_attn = results["self attention pytorch"]
            row.extend([self_attn["execution_time"], self_attn["used_memory"]])
        else:
            row.extend(["N/A", "N/A"])
            
        # Add PyTorch results
        if "3-attention pytorch" in results:
            pytorch = results["3-attention pytorch"]
            row.extend([pytorch["execution_time"], pytorch["used_memory"]])
        else:
            row.extend(["N/A", "N/A"])
            
        # Add C++ results
        if "3-attention c++" in results:
            cpp = results["3-attention c++"]
            row.extend([cpp["execution_time"], cpp["used_memory"]])
        else:
            row.extend(["N/A", "N/A"])
        
        # Add comparisons
        if "3-attention pytorch" in results and "3-attention c++" in results:
            pytorch = results["3-attention pytorch"]
            cpp = results["3-attention c++"]
            
            time_diff = (1 - cpp["execution_time"] / pytorch["execution_time"]) * 100
            mem_diff = (1 - cpp["used_memory"] / pytorch["used_memory"]) * 100
            row.extend([time_diff, mem_diff])
        else:
            row.extend(["N/A", "N/A"])
            
        # Add comparison with self-attention
        if "self attention pytorch" in results and "3-attention c++" in results:
            self_attn = results["self attention pytorch"]
            cpp = results["3-attention c++"]
            
            time_diff = (1 - cpp["execution_time"] / self_attn["execution_time"]) * 100
            mem_diff = (1 - cpp["used_memory"] / self_attn["used_memory"]) * 100
            row.extend([time_diff, mem_diff])
        else:
            row.extend(["N/A", "N/A"])
            
        # Add total operations
        total_ops = bs * seq_len * seq_len * seq_len * dim * heads / 1e9
        row.append(total_ops)
        
        rows.append(row)
    
    # Write to CSV
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows)
        
    print(f"\nResults saved to {filename}")
    return filename


if __name__ == "__main__":
    # Test with various configurations to better evaluate optimizations
    sizes = [
        # Small configurations
        (1, 32, 128, 2),     # Tiny
        (1, 64, 128, 2),     # Small with longer sequence
        (1, 64, 256, 4),     # Small standard config
        
        # Medium configurations
        (2, 64, 256, 4),     # Medium batch
        (2, 128, 256, 4),    # Medium with longer sequence
        (2, 64, 512, 8),     # Medium with higher dimensions
        
        # Large configurations (may cause OOM on smaller GPUs)
        (4, 64, 256, 4),     # Larger batch
        (4, 128, 256, 4),    # Larger batch, longer sequence
        (2, 64, 768, 12),    # BERT-base like config
    ]
    
    print("=== Memory Benchmark for Attention Mechanisms ===")
    
    all_results = {}
    
    for config in sizes:
        try:
            print(f"\nTesting configuration: batch={config[0]}, seq_len={config[1]}, dim={config[2]}, heads={config[3]}")
            results = benchmark_implementations(*config)
            print_results_table(results, config)
            all_results[config] = results
        except RuntimeError as e:
            error_msg = str(e).split('.')[0]
            print(f"Error with batch={config[0]}, seq_len={config[1]}, dim={config[2]}, heads={config[3]}: {error_msg}...")
            print("Skipping to next size...")
            
    # Save results to CSV
    csv_file = save_results_to_csv(all_results)