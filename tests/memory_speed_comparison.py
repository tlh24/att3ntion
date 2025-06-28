import torch
import time
import os
import sys
import hyper_attn_cpp_manual # Built from setup.py
import hyper_attn_cpp_reference # Built from setup.py
from self_attn_pytorch import SelfAttention
from pynvml import *
nvmlInit()

# --- Equivalence Test Configuration ---
B_test = 2
H_test = 2
I_test = 4
J_test = 4
K_test = 4
D_test = 8
dtype_test = torch.float32
device_cpu = torch.device("cpu")
device_cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rtol = 1e-4
atol = 1e-5

def generate_test_inputs(device):
    """Generates random input tensors on the specified device for equivalence testing."""
    Q = torch.randn(B_test, H_test, I_test, D_test, dtype=dtype_test, device=device)
    R = torch.randn(B_test, H_test, J_test, D_test, dtype=dtype_test, device=device)
    S = torch.randn(B_test, H_test, K_test, D_test, dtype=dtype_test, device=device)
    Vq_1 = torch.randn(B_test, H_test, I_test, D_test, dtype=dtype_test, device=device)
    Vq_2 = torch.randn(B_test, H_test, I_test, D_test, dtype=dtype_test, device=device)
    Vr_1 = torch.randn(B_test, H_test, J_test, D_test, dtype=dtype_test, device=device)
    Vr_2 = torch.randn(B_test, H_test, J_test, D_test, dtype=dtype_test, device=device)
    Vs_1 = torch.randn(B_test, H_test, K_test, D_test, dtype=dtype_test, device=device)
    Vs_2 = torch.randn(B_test, H_test, K_test, D_test, dtype=dtype_test, device=device)
    dropout_rate = 0.0
    return Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2, dropout_rate

def print_equivalence_table(headers, rows):
    """Helper function to print formatted tables for equivalence test results."""
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

def test_forward_equivalence():
    """Test forward pass equivalence between CPU and CUDA implementations."""
    print(f"\nTesting Forward Pass Equivalence (CPU vs CUDA)")
    print(f"Parameters: B={B_test}, H={H_test}, I={I_test}, J={J_test}, K={K_test}, D={D_test}")
    
    # Generate inputs on CPU
    print("Running forward pass on CPU...")
    Q_cpu, R_cpu, S_cpu, Vq_1_cpu, Vq_2_cpu, Vr_1_cpu, Vr_2_cpu, Vs_1_cpu, Vs_2_cpu, dr_cpu = generate_test_inputs(device_cpu)
    
    try:
        Y_tuple_cpu = hyper_attn_cpp_manual.forward(Q_cpu, R_cpu, S_cpu, Vq_1_cpu, Vq_2_cpu, Vr_1_cpu, Vr_2_cpu, Vs_1_cpu, Vs_2_cpu, dr_cpu)
        Y_q_cpu, Y_r_cpu, Y_s_cpu, Y_q__cpu, Y_r__cpu, Y_s__cpu = Y_tuple_cpu
        print("CPU forward pass completed.")
    except Exception as e:
        print(f"Error during CPU forward pass: {e}")
        return False

    if not torch.cuda.is_available():
        print("CUDA not available. Skipping CUDA execution and comparison.")
        return True  # Consider this a pass if CUDA is not available

    # Move inputs to CUDA
    print("Moving inputs to CUDA device...")
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
    except Exception as e:
        print(f"Error moving inputs to CUDA: {e}")
        return False

    # Run CUDA forward pass
    print("Running forward pass on CUDA...")
    try:
        Y_tuple_cuda = hyper_attn_cpp_manual.forward(Q_cuda, R_cuda, S_cuda, Vq_1_cuda, Vq_2_cuda, Vr_1_cuda, Vr_2_cuda, Vs_1_cuda, Vs_2_cuda, dr_cuda)
        Y_q_cuda, Y_r_cuda, Y_s_cuda, Y_q__cuda, Y_r__cuda, Y_s__cuda = Y_tuple_cuda
        print("CUDA forward pass completed.")
    except Exception as e:
        print(f"Error during CUDA forward pass: {e}")
        return False

    # Compare results
    print("Comparing CPU and CUDA outputs...")
    Y_q_cuda_cpu = Y_q_cuda.cpu()
    Y_r_cuda_cpu = Y_r_cuda.cpu()
    Y_s_cuda_cpu = Y_s_cuda.cpu()
    Y_q__cuda_cpu = Y_q__cuda.cpu()
    Y_r__cuda_cpu = Y_r__cuda.cpu()
    Y_s__cuda_cpu = Y_s__cuda.cpu()

    # Check shapes
    shapes_match = all([
        Y_q_cpu.shape == Y_q_cuda_cpu.shape,
        Y_r_cpu.shape == Y_r_cuda_cpu.shape,
        Y_s_cpu.shape == Y_s_cuda_cpu.shape,
        Y_q__cpu.shape == Y_q__cuda_cpu.shape,
        Y_r__cpu.shape == Y_r__cuda_cpu.shape,
        Y_s__cpu.shape == Y_s__cuda_cpu.shape
    ])
    
    if not shapes_match:
        print("ERROR: Shape mismatches detected!")
        return False

    # Check numerical equivalence
    yq_close = torch.allclose(Y_q_cpu, Y_q_cuda_cpu, rtol=rtol, atol=atol)
    yr_close = torch.allclose(Y_r_cpu, Y_r_cuda_cpu, rtol=rtol, atol=atol)
    ys_close = torch.allclose(Y_s_cpu, Y_s_cuda_cpu, rtol=rtol, atol=atol)
    yq__close = torch.allclose(Y_q__cpu, Y_q__cuda_cpu, rtol=rtol, atol=atol)
    yr__close = torch.allclose(Y_r__cpu, Y_r__cuda_cpu, rtol=rtol, atol=atol)
    ys__close = torch.allclose(Y_s__cpu, Y_s__cuda_cpu, rtol=rtol, atol=atol)

    forward_results = [
        ["Y_q", "PASS" if yq_close else "FAIL", (Y_q_cpu - Y_q_cuda_cpu).abs().max().item() if not yq_close else 0],
        ["Y_r", "PASS" if yr_close else "FAIL", (Y_r_cpu - Y_r_cuda_cpu).abs().max().item() if not yr_close else 0],
        ["Y_s", "PASS" if ys_close else "FAIL", (Y_s_cpu - Y_s_cuda_cpu).abs().max().item() if not ys_close else 0],
        ["Y_q_", "PASS" if yq__close else "FAIL", (Y_q__cpu - Y_q__cuda_cpu).abs().max().item() if not yq__close else 0],
        ["Y_r_", "PASS" if yr__close else "FAIL", (Y_r__cpu - Y_r__cuda_cpu).abs().max().item() if not yr__close else 0],
        ["Y_s_", "PASS" if ys__close else "FAIL", (Y_s__cpu - Y_s__cuda_cpu).abs().max().item() if not ys__close else 0]
    ]
    
    print("\nForward Pass Results:")
    print_equivalence_table(["Output", "Status", "Max Diff"], forward_results)

    all_passed = yq_close and yr_close and ys_close and yq__close and yr__close and ys__close
    
    if all_passed:
        print("*** Forward Pass Equivalence Test Passed! ***")
        return True
    else:
        print("*** Forward Pass Equivalence Test Failed! ***")
        return False

def test_backward_equivalence():
    """Test backward pass equivalence between CPU and CUDA implementations."""
    print("\nTesting Backward Pass Equivalence (CPU vs CUDA)")
    
    # Generate inputs on CPU
    Q_cpu, R_cpu, S_cpu, Vq_1_cpu, Vq_2_cpu, Vr_1_cpu, Vr_2_cpu, Vs_1_cpu, Vs_2_cpu, dr_cpu = generate_test_inputs(device_cpu)
    
    N_grad = max(I_test, J_test, K_test)
    grad_output_cpu = torch.randn(B_test, H_test, N_grad, D_test, dtype=dtype_test, device=device_cpu)

    print("Running backward pass on CPU...")
    try:
        grads_tuple_cpu = hyper_attn_cpp_manual.backward(
            grad_output_cpu,
            Q_cpu, R_cpu, S_cpu, 
            Vq_1_cpu, Vq_2_cpu, Vr_1_cpu, Vr_2_cpu, Vs_1_cpu, Vs_2_cpu, 
            dr_cpu
        )
        
        grad_names = ["grad_Q", "grad_R", "grad_S", "grad_Vq_1", "grad_Vq_2", 
                     "grad_Vr_1", "grad_Vr_2", "grad_Vs_1", "grad_Vs_2"]
        print("CPU backward pass completed.")
    except Exception as e:
        print(f"Error during CPU backward pass: {e}")
        return False

    if not torch.cuda.is_available():
        print("CUDA not available. Skipping CUDA backward execution and comparison.")
        return True

    # Move inputs to CUDA
    print("Moving inputs to CUDA device for backward pass...")
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
        grad_output_cuda = grad_output_cpu.to(device_cuda)
        dr_cuda = dr_cpu
    except Exception as e:
        print(f"Error moving inputs to CUDA for backward pass: {e}")
        return False

    print("Running backward pass on CUDA...")
    try:
        grads_tuple_cuda = hyper_attn_cpp_manual.backward(
            grad_output_cuda,
            Q_cuda, R_cuda, S_cuda, 
            Vq_1_cuda, Vq_2_cuda, Vr_1_cuda, Vr_2_cuda, Vs_1_cuda, Vs_2_cuda, 
            dr_cuda
        )
        print("CUDA backward pass completed.")
    except Exception as e:
        print(f"Error during CUDA backward pass: {e}")
        return False

    # Compare results
    print("Comparing CPU and CUDA gradient outputs...")
    
    results = []
    all_passed = True
    
    for i, name in enumerate(grad_names):
        try:
            grad_cuda_cpu = grads_tuple_cuda[i].cpu()
            grad_cpu = grads_tuple_cpu[i]
            
            shape_match = grad_cpu.shape == grad_cuda_cpu.shape
            if not shape_match:
                results.append([name, "FAIL", f"Shape mismatch: CPU {grad_cpu.shape}, CUDA {grad_cuda_cpu.shape}"])
                all_passed = False
                continue
                
            is_close = torch.allclose(grad_cpu, grad_cuda_cpu, rtol=rtol, atol=atol)
            max_diff = (grad_cpu - grad_cuda_cpu).abs().max().item() if not is_close else 0
            
            results.append([name, "PASS" if is_close else "FAIL", max_diff])
            if not is_close:
                all_passed = False
        except Exception as e:
            results.append([name, "FAIL", f"Error: {str(e)}"])
            all_passed = False
    
    print("\nBackward Pass Results:")
    print_equivalence_table(["Gradient", "Status", "Max Diff/Error"], results)
    
    if all_passed:
        print("*** Backward Pass Equivalence Test Passed! ***")
        return True
    else:
        print("*** Backward Pass Equivalence Test Failed! ***")
        return False

def run_equivalence_tests():
    """Run both forward and backward equivalence tests."""
    print("=" * 60)
    print("CUDA EQUIVALENCE TESTING")
    print("=" * 60)
    
    forward_passed = test_forward_equivalence()
    if not forward_passed:
        return False
        
    backward_passed = test_backward_equivalence()
    if not backward_passed:
        return False
        
    print("\n" + "=" * 60)
    print("ALL EQUIVALENCE TESTS PASSED! Proceeding with benchmarks...")
    print("=" * 60)
    return True

def get_grad_output_cuda(Y_q, Y_r, Y_s, Y_q_, Y_r_, Y_s_):
    B, H, I, D = Y_q.shape
    _, _, J, _ = Y_r.shape
    _, _, K, _ = Y_s.shape
    
    max_len = max(I, J, K)
    
    grad_output_combined = torch.zeros(B, H, max_len, D, device=Y_q.device, dtype=Y_q.dtype)
    
    # Gradients from gather outputs
    grad_output_combined[:, :, :I, :] += torch.ones_like(Y_q)
    grad_output_combined[:, :, :J, :] += torch.ones_like(Y_r)
    grad_output_combined[:, :, :K, :] += torch.ones_like(Y_s)
    
    # Gradients from scatter outputs (add to the same combined grad)
    grad_output_combined[:, :, :I, :] += torch.ones_like(Y_q_)
    grad_output_combined[:, :, :J, :] += torch.ones_like(Y_r_)
    grad_output_combined[:, :, :K, :] += torch.ones_like(Y_s_)
    
    return grad_output_combined

def benchmark():
    configs = [
        (1, 2, 4, 4, 4, 8),   
        (1, 2, 6, 6, 6, 8),  
        (1, 2, 8, 8, 8, 8),  
        (1, 2, 12, 12, 12, 8),  
        (1, 2, 16, 16, 16, 8),  
        (1, 2, 24, 24, 24, 8),  
        (1, 2, 32, 32, 32, 8),  
        (1, 2, 48, 48, 48, 8),
        (1, 2, 64, 64, 64, 8),
        (1, 2, 96, 96, 96, 8),
        (1, 2, 128, 128, 128, 8), #fails for larger inputs due to shared memory size / need to make this dynamic
        (1, 2, 256, 256, 256, 8),
        (1, 2, 512, 512, 512, 8),
        (1, 2, 1024, 1024, 1024, 8),
        # (1, 2, 4096, 4096, 4096, 8),
    ]

    dropout_rate = 0.0 # Keep dropout off for direct comparison of core ops

    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARKS")
    print("=" * 80)

    # --- Run Vanilla Attention Benchmark First ---
    print("\n--- Vanilla PyTorch Self-Attention Benchmark ---")
    header_vanilla = (f"{'Seq Len':<15} | "
                      f"{'Vanilla ms':<15} | {'Vanilla MB':<15} | {'Vanilla VRAM':<15}")
    print(header_vanilla)
    print("-" * len(header_vanilla))

    for B, H, I_dim, J_dim, K_dim, D_dim in configs:
        try:
            embedding_dim = H * D_dim
            seq_len_vanilla = I_dim
            
            x_vanilla = torch.rand(B, seq_len_vanilla, embedding_dim, device='cuda', dtype=torch.float32, requires_grad=True)

            vanilla_attention_model = SelfAttention(
                embedding_dim=embedding_dim,
                num_heads=H,
                dropout_rate=dropout_rate
            ).to('cuda')

            torch.cuda.reset_peak_memory_stats()
            
            handle = nvmlDeviceGetHandleByIndex(0)
            pre_used = nvmlDeviceGetMemoryInfo(handle).used
            
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            output_vanilla = vanilla_attention_model(x_vanilla)
            loss_vanilla = output_vanilla.sum()
            loss_vanilla.backward()
            
            torch.cuda.synchronize()
            total_time = time.perf_counter() - start_time
            
            post_used = nvmlDeviceGetMemoryInfo(handle).used
            vram_used_mb = (post_used - pre_used) / (1024 * 1024)
            peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            
            print(f"{I_dim:<15} | {total_time * 1000:<15.4f} | {peak_mem_mb:<15.2f} | {vram_used_mb:<15.2f}")

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{I_dim:<15} | {'OOM':<15} | {'N/A':<15} | {'N/A':<15}")
        except Exception as e:
            print(f"{I_dim:<15} | {'Error':<15} | {'N/A':<15} | {'N/A':<15}")

    # --- Run Custom CUDA & PyTorch Benchmarks Second ---
    print("\n--- Custom CUDA & PyTorch C++ Reference Benchmarks ---")
    header_custom = (f"{'Seq Len':<10} | "
                     f"{'CUDA ms':<12} | {'Torch ms':<12} | "
                     f"{'CUDA MB':<12} | {'Torch MB':<12} | "
                     f"{'CUDA VRAM':<12} | {'Torch VRAM':<12}")
    print(header_custom)
    print("-" * len(header_custom))

    for B, H, I_dim, J_dim, K_dim, D_dim in configs:
        # Initialize tensors
        try:
            Q = torch.rand(B, H, I_dim, D_dim, device='cuda', dtype=torch.float32)
            R = torch.rand(B, H, J_dim, D_dim, device='cuda', dtype=torch.float32)
            S = torch.rand(B, H, K_dim, D_dim, device='cuda', dtype=torch.float32)
            Vq_1 = torch.rand(B, H, I_dim, D_dim, device='cuda', dtype=torch.float32)
            Vq_2 = torch.rand(B, H, I_dim, D_dim, device='cuda', dtype=torch.float32)
            Vr_1 = torch.rand(B, H, J_dim, D_dim, device='cuda', dtype=torch.float32)
            Vr_2 = torch.rand(B, H, J_dim, D_dim, device='cuda', dtype=torch.float32)
            Vs_1 = torch.rand(B, H, K_dim, D_dim, device='cuda', dtype=torch.float32)
            Vs_2 = torch.rand(B, H, K_dim, D_dim, device='cuda', dtype=torch.float32)
        except Exception as e:
            print(f"{I_dim:<10} | Error initializing tensors: {e}")
            continue

        # Initialize variables before try blocks to avoid "referenced before assignment" errors
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

            Y_q_mc, Y_r_mc, Y_s_mc, Y_q__mc, Y_r__mc, Y_s__mc = hyper_attn_cpp_manual.forward(
                Q.clone(), R.clone(), S.clone(), Vq_1.clone(), Vq_2.clone(),
                Vr_1.clone(), Vr_2.clone(), Vs_1.clone(), Vs_2.clone(), dropout_rate)
            grad_output_cuda = get_grad_output_cuda(Y_q_mc, Y_r_mc, Y_s_mc, Y_q__mc, Y_r__mc, Y_s__mc)
            grads_manual = hyper_attn_cpp_manual.backward(
                grad_output_cuda, Q.clone(), R.clone(), S.clone(), Vq_1.clone(), Vq_2.clone(),
                Vr_1.clone(), Vr_2.clone(), Vs_1.clone(), Vs_2.clone(), dropout_rate)
            
            torch.cuda.synchronize()
            total_time_manual_cuda = time.perf_counter() - start_time
            post_used = nvmlDeviceGetMemoryInfo(handle).used
            manual_vram_used_mb = (post_used - pre_used) / (1024 * 1024)
            peak_mem_manual_cuda_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        except torch.cuda.OutOfMemoryError:
            total_time_manual_cuda = float('inf')
        except Exception:
            total_time_manual_cuda = float('nan') # Indicates non-OOM error

        # --- PyTorch C++ Reference Benchmark ---
        try:
            Q_ref, R_ref, S_ref = Q.clone().requires_grad_(True), R.clone().requires_grad_(True), S.clone().requires_grad_(True)
            Vq_1_ref, Vq_2_ref = Vq_1.clone().requires_grad_(True), Vq_2.clone().requires_grad_(True)
            Vr_1_ref, Vr_2_ref = Vr_1.clone().requires_grad_(True), Vr_2.clone().requires_grad_(True)
            Vs_1_ref, Vs_2_ref = Vs_1.clone().requires_grad_(True), Vs_2.clone().requires_grad_(True)

            torch.cuda.reset_peak_memory_stats()
            handle = nvmlDeviceGetHandleByIndex(0)
            pre_used = nvmlDeviceGetMemoryInfo(handle).used

            torch.cuda.synchronize()
            start_time = time.perf_counter()

            Y_q_ref, Y_r_ref, Y_s_ref, Y_q__ref, Y_r__ref, Y_s__ref = hyper_attn_cpp_reference.forward(
                Q_ref, R_ref, S_ref, Vq_1_ref, Vq_2_ref, Vr_1_ref, Vr_2_ref, Vs_1_ref, Vs_2_ref, dropout_rate)
            loss = (Y_q_ref.sum() + Y_r_ref.sum() + Y_s_ref.sum() + Y_q__ref.sum() + Y_r__ref.sum() + Y_s__ref.sum())
            loss.backward()

            torch.cuda.synchronize()
            total_time_pytorch_ref = time.perf_counter() - start_time
            post_used = nvmlDeviceGetMemoryInfo(handle).used
            pytorch_ref_vram_used_mb = (post_used - pre_used) / (1024 * 1024)
            peak_mem_pytorch_ref_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        except torch.cuda.OutOfMemoryError:
            total_time_pytorch_ref = float('inf')
        except Exception:
            total_time_pytorch_ref = float('nan')

        # --- Print Results for Custom Benchmarks ---
        cuda_time_str = f"{total_time_manual_cuda * 1000:<12.4f}"
        cuda_mem_str = f"{peak_mem_manual_cuda_mb:<12.2f}"
        cuda_vram_str = f"{manual_vram_used_mb:<12.2f}"
        if total_time_manual_cuda == float('inf'):
            cuda_time_str = f"{'OOM':<12}"
            cuda_mem_str = f"{'N/A':<12}"
            cuda_vram_str = f"{'N/A':<12}"
        elif total_time_manual_cuda == float('nan'):
            cuda_time_str = f"{'Error':<12}"
            cuda_mem_str = f"{'N/A':<12}"
            cuda_vram_str = f"{'N/A':<12}"

        torch_time_str = f"{total_time_pytorch_ref * 1000:<12.4f}"
        torch_mem_str = f"{peak_mem_pytorch_ref_mb:<12.2f}"
        torch_vram_str = f"{pytorch_ref_vram_used_mb:<12.2f}"
        if total_time_pytorch_ref == float('inf'):
            torch_time_str = f"{'OOM':<12}"
            torch_mem_str = f"{'N/A':<12}"
            torch_vram_str = f"{'N/A':<12}"
        elif total_time_pytorch_ref == float('nan'):
            torch_time_str = f"{'Error':<12}"
            torch_mem_str = f"{'N/A':<12}"
            torch_vram_str = f"{'N/A':<12}"
        
        print(f"{I_dim:<10} | {cuda_time_str} | {torch_time_str} | {cuda_mem_str} | {torch_mem_str} | {cuda_vram_str} | {torch_vram_str}")

    print("-" * len(header_custom))

if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("CUDA is not available. Aborting benchmark.")
    else:
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        try:
            import hyper_attn_cpp_manual
            import hyper_attn_cpp_reference
            print("Extensions imported successfully.")
            
            # First run equivalence tests
            if run_equivalence_tests():
                # Only run benchmarks if equivalence tests pass
                benchmark()
            else:
                print("\n" + "=" * 60)
                print("EQUIVALENCE TESTS FAILED - SKIPPING BENCHMARKS")
                print("Please fix the implementation issues before benchmarking.")
                print("=" * 60)
                sys.exit(1)
                
        except ImportError as e:
            print(f"ImportError: {e}")
            print("Please ensure the extensions are compiled and accessible.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    nvmlShutdown()
