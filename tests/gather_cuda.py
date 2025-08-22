import torch
import os
import sys
import time
from pynvml import *

# Import extensions (assuming they are compiled and accessible)
# The order of imports matters for shared library loading, ensure pytorch-related ones are first if there are conflicts.
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

try:
    import hyper_attn_cpp_manual as manual_att3ntion
    import hyper_attn_cpp_reference
    print("Successfully imported C++/CUDA extensions.")
except ImportError:
    print("\nError: Failed to import the C++/CUDA extension 'hyper_attn_cpp_manual' or 'hyper_attn_cpp_reference'.")
    print("Please ensure the extensions have been compiled successfully and the names match setup.py.")
    sys.exit(1)


# --- Configuration ---
B = 2
H = 2
I = 4
J = 4
K = 4
D = 64
dtype = torch.bfloat16
device_cpu = torch.device("cpu")
device_cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rtol = 2e-2 if dtype == torch.bfloat16 else 1e-4
atol = 1e-2 if dtype == torch.bfloat16 else 1e-5

print(f"Testing Gather Equivalence (CUDA vs Torch)")
print(f"Parameters: B={B}, H={H}, I={I}, J={J}, K={K}, D={D}")
print(f"CPU Device: {device_cpu}")
print(f"CUDA Device: {device_cuda}")
print(f"Testing dtype: {dtype}")

# ---> Verify LD_LIBRARY_PATH before import <---
print(f"LD_LIBRARY_PATH before import: {os.environ.get('LD_LIBRARY_PATH')}")


def generate_inputs(device):
    """Generates random input tensors on the specified device."""
    Q = torch.randn(B, H, I, D, dtype=dtype, device=device)
    R = torch.randn(B, H, J, D, dtype=dtype, device=device)
    S = torch.randn(B, H, K, D, dtype=dtype, device=device)
    Vq_1 = torch.randn(B, H, I, D, dtype=dtype, device=device)
    Vq_2 = torch.randn(B, H, I, D, dtype=dtype, device=device)
    Vr_1 = torch.randn(B, H, J, D, dtype=dtype, device=device)
    Vr_2 = torch.randn(B, H, J, D, dtype=dtype, device=device)
    Vs_1 = torch.randn(B, H, K, D, dtype=dtype, device=device)
    Vs_2 = torch.randn(B, H, K, D, dtype=dtype, device=device)
    dropout_rate = 0.0
    return Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2, dropout_rate


def print_table(headers, rows):
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, item in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(item)))
    
    header = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    separator = "-+-".join("-" * w for w in col_widths)
    print(f"| {header} |")
    print(f"|-{separator}-|")
    
    for row in rows:
        row_str = " | ".join(str(item).ljust(col_widths[i]) for i, item in enumerate(row))
        print(f"| {row_str} |")


def run_test():
    print("\nRunning forward pass on CPU...")
    Q_cpu, R_cpu, S_cpu, Vq_1_cpu, Vq_2_cpu, Vr_1_cpu, Vr_2_cpu, Vs_1_cpu, Vs_2_cpu, dr_cpu = generate_inputs(device_cpu)
    assert Q_cpu.device.type == 'cpu', "Input tensors not on CPU!"

    if dtype == torch.bfloat16:
        print("Skipping CPU forward pass for bf16.")
    else:
        try:
            Y_tuple_cpu = manual_att3ntion.forward(Q_cpu, R_cpu, S_cpu, Vq_1_cpu, Vq_2_cpu, Vr_1_cpu, Vr_2_cpu, Vs_1_cpu, Vs_2_cpu, dr_cpu)
            Y_q_cpu, Y_r_cpu, Y_s_cpu, _, _, _ = Y_tuple_cpu
            print("CPU forward pass completed.")
        except Exception as e:
            print(f"Error during CPU forward pass: {e}")
            sys.exit(1)

    if not torch.cuda.is_available():
        print("\nCUDA not available. Skipping CUDA execution and comparison.")
        return True

    print("\nMoving inputs to CUDA device...")
    try:
        Q_cuda = Q_cpu.to(device_cuda)
        R_cuda = R_cpu.to(device_cuda)
        S_cuda = S_cpu.to(device_cuda)
        Vq_1_cuda = Vq_1_cpu.to(device_cuda)
        Vq_2_cuda = Vq_2_cpu.to(device_cuda)
        Vr_1_cuda = Vr_1_cpu.to(device_cuda)
        Vr_2_cuda = Vr_2_cpu.to(device_cuda)
        Vs_1_cuda = Vs_1_cpu.to(device_cuda)
        Vs_2_cuda = Vs_2_cpu.to(device_cuda)
        dr_cuda = dr_cpu
        assert Q_cuda.device.type == 'cuda', "Input tensors not on CUDA!"
        print("Inputs moved to CUDA.")
    except Exception as e:
        print(f"Error moving inputs to CUDA: {e}")
        sys.exit(1)

    print("Running forward pass on CUDA...")
    try:
        # Manual CUDA (flash gather inside manual forward)
        Y_tuple_manual = manual_att3ntion.forward(Q_cuda, R_cuda, S_cuda, Vq_1_cuda, Vq_2_cuda, Vr_1_cuda, Vr_2_cuda, Vs_1_cuda, Vs_2_cuda, dr_cuda)
        Y_q_manual, Y_r_manual, Y_s_manual, _, _, _ = Y_tuple_manual
        print("Manual CUDA forward completed.")
    except Exception as e:
        print(f"Error during manual CUDA forward pass: {e}")
        sys.exit(1)

    try:
        # Torch C++ reference (same inputs)
        Y_tuple_ref = hyper_attn_cpp_reference.forward(Q_cuda.clone(), R_cuda.clone(), S_cuda.clone(),
                                                       Vq_1_cuda.clone(), Vq_2_cuda.clone(),
                                                       Vr_1_cuda.clone(), Vr_2_cuda.clone(),
                                                       Vs_1_cuda.clone(), Vs_2_cuda.clone(), dr_cuda)
        Y_q_ref, Y_r_ref, Y_s_ref, _, _, _ = Y_tuple_ref
        print("Torch reference forward completed.")
    except Exception as e:
        print(f"Error during torch reference forward pass: {e}")
        sys.exit(1)

    print("\nComparing CUDA outputs (manual vs torch)...")
    Y_q_m_cpu = Y_q_manual.cpu()
    Y_r_m_cpu = Y_r_manual.cpu()
    Y_s_m_cpu = Y_s_manual.cpu()
    Y_q_r_cpu = Y_q_ref.cpu()
    Y_r_r_cpu = Y_r_ref.cpu()
    Y_s_r_cpu = Y_s_ref.cpu()

    shape_match = True
    if Y_q_m_cpu.shape != Y_q_r_cpu.shape:
        print(f"ERROR: Y_q shape mismatch! Manual: {Y_q_m_cpu.shape}, Torch: {Y_q_r_cpu.shape}")
        shape_match = False
    if Y_r_m_cpu.shape != Y_r_r_cpu.shape:
        print(f"ERROR: Y_r shape mismatch! Manual: {Y_r_m_cpu.shape}, Torch: {Y_r_r_cpu.shape}")
        shape_match = False
    if Y_s_m_cpu.shape != Y_s_r_cpu.shape:
        print(f"ERROR: Y_s shape mismatch! Manual: {Y_s_m_cpu.shape}, Torch: {Y_s_r_cpu.shape}")
        shape_match = False

    if not shape_match:
        print("Exiting due to shape mismatches.")
        return False
    else:
        print("Shapes match.")

    # Check numerical equivalence for gather outputs
    yq_close = torch.allclose(Y_q_m_cpu, Y_q_r_cpu, rtol=rtol, atol=atol)
    yr_close = torch.allclose(Y_r_m_cpu, Y_r_r_cpu, rtol=rtol, atol=atol)
    ys_close = torch.allclose(Y_s_m_cpu, Y_s_r_cpu, rtol=rtol, atol=atol)

    forward_results = [
        ["Y_q", "PASS" if yq_close else "FAIL", (Y_q_m_cpu - Y_q_r_cpu).abs().max().item() if not yq_close else 0],
        ["Y_r", "PASS" if yr_close else "FAIL", (Y_r_m_cpu - Y_r_r_cpu).abs().max().item() if not yr_close else 0],
        ["Y_s", "PASS" if ys_close else "FAIL", (Y_s_m_cpu - Y_s_r_cpu).abs().max().item() if not ys_close else 0],
        ["Y_q_", "SKIPPED", "N/A"],
        ["Y_r_", "SKIPPED", "N/A"],
        ["Y_s_", "SKIPPED", "N/A"],
    ]

    all_passed = yq_close and yr_close and ys_close

    if not all_passed:
        print("\nForward Pass Results:")
        print_table(["Output", "Status", "Max Diff"], forward_results)
        print("\n*** Gather Equivalence Test Failed! ***")
        return False
    else:
        print("\nGather-only Equivalence Test Passed.")
        return True


configs = [  # dims: b, h, i, j, k, d
    (1, 1, 4, 4, 4, 64),
    (1, 1, 6, 6, 6, 64),
    (1, 1, 8, 8, 8, 64),
    (1, 1, 12, 12, 12, 64),
    (1, 1, 16, 16, 16, 64),
    (1, 1, 24, 24, 24, 64),
    (1, 1, 32, 32, 32, 64),
    (1, 1, 48, 48, 48, 64),
    (1, 1, 64, 64, 64, 64),
    (1, 1, 96, 96, 96, 64),
    (1, 1, 128, 128, 128, 64),
    (1, 1, 256, 256, 256, 64),
    (1, 1, 512, 512, 512, 64),
]


def benchmark():
    dropout_rate = 0.0

    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARKS (FORWARD PASS, YQ GATHER ONLY)")
    print("=" * 80)

    print("\n--- Custom CUDA & PyTorch C++ Reference Benchmarks ---")
    header_custom = (f"{'Seq Len':<10} | "
                     f"{'CUDA ms':<12} | {'Torch ms':<12} | "
                     f"{'CUDA TFLOP/s':<15} | {'Torch TFLOP/s':<15} | "
                     f"{'CUDA Peak MB':<12} | {'Torch Peak MB':<12}")
    print(header_custom)
    print("-" * len(header_custom))

    for B, H, I_dim, J_dim, K_dim, D_dim in configs:
        flops = (B * H) * (4 * I_dim * J_dim * K_dim * D_dim + 3 * J_dim * K_dim * D_dim)
        try:
            Q = torch.rand(B, H, I_dim, D_dim, device='cuda', dtype=dtype)
            R = torch.rand(B, H, J_dim, D_dim, device='cuda', dtype=dtype)
            S = torch.rand(B, H, K_dim, D_dim, device='cuda', dtype=dtype)
            Vq_1 = torch.rand(B, H, I_dim, D_dim, device='cuda', dtype=dtype)
            Vq_2 = torch.rand(B, H, I_dim, D_dim, device='cuda', dtype=dtype)
            Vr_1 = torch.rand(B, H, J_dim, D_dim, device='cuda', dtype=dtype)
            Vr_2 = torch.rand(B, H, J_dim, D_dim, device='cuda', dtype=dtype)
            Vs_1 = torch.rand(B, H, K_dim, D_dim, device='cuda', dtype=dtype)
            Vs_2 = torch.rand(B, H, K_dim, D_dim, device='cuda', dtype=dtype)
        except Exception as e:
            print(f"{I_dim:<10} | Error initializing tensors: {e}")
            continue

        peak_mem_manual_cuda_mb = 0.0
        manual_vram_used_mb = 0.0
        peak_mem_pytorch_ref_mb = 0.0
        pytorch_ref_vram_used_mb = 0.0

        # --- Manual CUDA Benchmark ---
        try:
            torch.cuda.reset_peak_memory_stats()
            handle = nvmlDeviceGetHandleByIndex(0)
            pre_used = nvmlDeviceGetMemoryInfo(handle).used
            
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            # Run and measure full forward; we focus on gather outputs in reporting
            Y_q_mc, Y_r_mc, Y_s_mc, _, _, _ = manual_att3ntion.forward(
                Q.clone(), R.clone(), S.clone(), Vq_1.clone(), Vq_2.clone(),
                Vr_1.clone(), Vr_2.clone(), Vs_1.clone(), Vs_2.clone(), dropout_rate)
            
            torch.cuda.synchronize()
            total_time_manual_cuda = time.perf_counter() - start_time
            post_used = nvmlDeviceGetMemoryInfo(handle).used
            manual_vram_used_mb = (post_used - pre_used) / (1024 * 1024)
            peak_mem_manual_cuda_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        except torch.cuda.OutOfMemoryError:
            total_time_manual_cuda = float('inf')
        except Exception as e:
            total_time_manual_cuda = float('nan')
            print(f"Error in manual attn: {e}")

        # --- PyTorch C++ Reference Benchmark ---
        try:
            Q_ref, R_ref, S_ref = Q.clone(), R.clone(), S.clone()
            Vq_1_ref, Vq_2_ref = Vq_1.clone(), Vq_2.clone()
            Vr_1_ref, Vr_2_ref = Vr_1.clone(), Vr_2.clone()
            Vs_1_ref, Vs_2_ref = Vs_1.clone(), Vs_2.clone()

            torch.cuda.reset_peak_memory_stats()
            handle = nvmlDeviceGetHandleByIndex(0)
            pre_used = nvmlDeviceGetMemoryInfo(handle).used

            torch.cuda.synchronize()
            start_time = time.perf_counter()

            Y_q_ref, Y_r_ref, Y_s_ref, _, _, _ = hyper_attn_cpp_reference.forward(
                Q_ref, R_ref, S_ref, Vq_1_ref, Vq_2_ref, Vr_1_ref, Vr_2_ref, Vs_1_ref, Vs_2_ref, dropout_rate)
            
            torch.cuda.synchronize()
            total_time_pytorch_ref = time.perf_counter() - start_time
            post_used = nvmlDeviceGetMemoryInfo(handle).used
            pytorch_ref_vram_used_mb = (post_used - pre_used) / (1024 * 1024)
            peak_mem_pytorch_ref_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        except torch.cuda.OutOfMemoryError:
            total_time_pytorch_ref = float('inf')
        except Exception as e:
            total_time_pytorch_ref = float('nan')
            print(f"Error in torch ref: {e}")

        cuda_tflops, torch_tflops = 0.0, 0.0
        if total_time_manual_cuda > 0 and not (total_time_manual_cuda == float('inf') or total_time_manual_cuda == float('nan')):
            cuda_tflops = (flops / total_time_manual_cuda) / 1e12
        if total_time_pytorch_ref > 0 and not (total_time_pytorch_ref == float('inf') or total_time_pytorch_ref == float('nan')):
            torch_tflops = (flops / total_time_pytorch_ref) / 1e12

        cuda_time_str = f"{total_time_manual_cuda * 1000:<12.4f}"
        cuda_mem_str = f"{peak_mem_manual_cuda_mb:<12.2f}"
        cuda_tflops_str = f"{cuda_tflops:<15.4f}"
        if total_time_manual_cuda == float('inf'):
            cuda_time_str = f"{'OOM':<12}"
            cuda_mem_str = f"{'N/A':<12}"
            cuda_tflops_str = f"{'N/A':<15}"
        elif total_time_manual_cuda == float('nan'):
            cuda_time_str = f"{'Error':<12}"
            cuda_mem_str = f"{'N/A':<12}"
            cuda_tflops_str = f"{'N/A':<15}"

        torch_time_str = f"{total_time_pytorch_ref * 1000:<12.4f}"
        torch_mem_str = f"{peak_mem_pytorch_ref_mb:<12.2f}"
        torch_tflops_str = f"{torch_tflops:<15.4f}"
        if total_time_pytorch_ref == float('inf'):
            torch_time_str = f"{'OOM':<12}"
            torch_mem_str = f"{'N/A':<12}"
            torch_tflops_str = f"{'N/A':<15}"
        elif total_time_pytorch_ref == float('nan'):
            torch_time_str = f"{'Error':<12}"
            torch_mem_str = f"{'N/A':<12}"
            torch_tflops_str = f"{'N/A':<15}"
        
        print(f"{I_dim:<10} | {cuda_time_str} | {torch_time_str} | {cuda_tflops_str} | {torch_tflops_str} | {cuda_mem_str} | {torch_mem_str}")

    print("-" * len(header_custom))


if __name__ == '__main__':
    nvmlInit()
    
    print("=" * 60)
    print("CUDA EQUIVALENCE TESTING (Yq_GATHER FORWARD ONLY)")
    print("=" * 60)
    
    forward_passed = run_test()
    
    if forward_passed:
        print("\n" + "=" * 60)
        print("ALL Yq_GATHER EQUIVALENCE TESTS PASSED! Proceeding with benchmarks...")
        print("=" * 60)
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            benchmark()
        else:
            print("CUDA not available. Skipping benchmarks.")
    else:
        print("\n" + "=" * 60)
        print("Yq_GATHER EQUIVALENCE TESTS FAILED - SKIPPING BENCHMARKS")
        print("Please fix the implementation issues before benchmarking.")
        print("=" * 60)
        sys.exit(1)
    
    nvmlShutdown()
