import torch
import time
import hyper_attn_cpp_manual # Built from setup.py
import hyper_attn_cpp_reference # Built from setup.py
from self_attn_pytorch import SelfAttention
from pynvml import *
nvmlInit()

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
                     f"{'CUDA ms':<12} | {'CUDA MB':<12} | {'CUDA VRAM':<12} | "
                     f"{'Torch ms':<12} | {'Torch MB':<12} | {'Torch VRAM':<12}")
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
        print(f"{I_dim:<10} | ", end="")
        if total_time_manual_cuda == float('inf'):
            print(f"{'CUDA OOM':<12} | {'N/A':<12} | {'N/A':<12} | ", end="")
        elif total_time_manual_cuda == float('nan'):
            print(f"{'CUDA Error':<12} | {'N/A':<12} | {'N/A':<12} | ", end="")
        else:
            print(f"{total_time_manual_cuda * 1000:<12.4f} | {peak_mem_manual_cuda_mb:<12.2f} | {manual_vram_used_mb:<12.2f} | ", end="")

        if total_time_pytorch_ref == float('inf'):
            print(f"{'Torch OOM':<12} | {'N/A':<12} | {'N/A':<12}")
        elif total_time_pytorch_ref == float('nan'):
            print(f"{'Torch Error':<12} | {'N/A':<12} | {'N/A':<12}")
        else:
            print(f"{total_time_pytorch_ref * 1000:<12.4f} | {peak_mem_pytorch_ref_mb:<12.2f} | {pytorch_ref_vram_used_mb:<12.2f}")

    print("-" * len(header_custom))

if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("CUDA is not available. Aborting benchmark.")
    else:
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        try:
            import hyper_attn_cpp_manual
            import hyper_attn_cpp_reference
            print("Compiling extensions if not already done...")
            print("You might need to run 'python setup.py install' or 'python setup.py build_ext --inplace' first.")
            benchmark()
        except ImportError as e:
            print(f"ImportError: {e}")
            print("Please ensure the extensions are compiled and accessible.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    nvmlShutdown()
