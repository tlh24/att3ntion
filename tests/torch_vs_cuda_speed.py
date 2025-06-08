import torch
import time
import hyper_attn_cpp_manual # Built from setup.py
import hyper_attn_cpp_reference # Built from setup.py

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
    ]

    dropout_rate = 0.0 # Keep dropout off for direct comparison of core ops

    print(f"{'Config (B,H,I,J,K,D)':<25} | {'Manual CUDA (ms)':<20} | {'PyTorch C++ Ref (ms)':<25}")
    print("-" * 80)

    for B, H, I_dim, J_dim, K_dim, D_dim in configs:
        config_str = f"({B},{H},{I_dim},{J_dim},{K_dim},{D_dim})"
        print(f"{config_str:<25} | ", end="")

        # Initialize tensors on GPU
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
            print(f"Error initializing tensors for {config_str}: {e}")
            print("Skipping this configuration.")
            continue


        # --- Manual CUDA Benchmark ---
        total_time_manual_cuda = 0
        try:
            # Forward
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            Y_q_mc, Y_r_mc, Y_s_mc, Y_q__mc, Y_r__mc, Y_s__mc = hyper_attn_cpp_manual.forward(
                Q.clone(), R.clone(), S.clone(),
                Vq_1.clone(), Vq_2.clone(),
                Vr_1.clone(), Vr_2.clone(),
                Vs_1.clone(), Vs_2.clone(),
                dropout_rate
            )
            torch.cuda.synchronize()
            fwd_time_manual = time.perf_counter() - start_time
            total_time_manual_cuda += fwd_time_manual

            # Backward
            grad_output_cuda = get_grad_output_cuda(Y_q_mc, Y_r_mc, Y_s_mc, Y_q__mc, Y_r__mc, Y_s__mc)
            
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            grads_manual = hyper_attn_cpp_manual.backward(
                grad_output_cuda, 
                Q.clone(), R.clone(), S.clone(),
                Vq_1.clone(), Vq_2.clone(),
                Vr_1.clone(), Vr_2.clone(),
                Vs_1.clone(), Vs_2.clone(),
                dropout_rate
            )
            torch.cuda.synchronize()
            bwd_time_manual = time.perf_counter() - start_time
            total_time_manual_cuda += bwd_time_manual
            print(f"{total_time_manual_cuda * 1000:.4f} ms".ljust(20) + " | ", end="")

        except Exception as e:
            print(f"CUDA Error: {e}".ljust(20) + " | ", end="")
            total_time_manual_cuda = float('inf') # Indicate failure


        # --- PyTorch C++ Reference Benchmark (with autograd) ---
        Q_ref = Q.clone().requires_grad_(True)
        R_ref = R.clone().requires_grad_(True)
        S_ref = S.clone().requires_grad_(True)
        Vq_1_ref = Vq_1.clone().requires_grad_(True)
        Vq_2_ref = Vq_2.clone().requires_grad_(True)
        Vr_1_ref = Vr_1.clone().requires_grad_(True)
        Vr_2_ref = Vr_2.clone().requires_grad_(True)
        Vs_1_ref = Vs_1.clone().requires_grad_(True)
        Vs_2_ref = Vs_2.clone().requires_grad_(True)
        
        total_time_pytorch_ref = 0
        try:
            # Forward
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            Y_q_ref, Y_r_ref, Y_s_ref, Y_q__ref, Y_r__ref, Y_s__ref = hyper_attn_cpp_reference.forward(
                Q_ref, R_ref, S_ref,
                Vq_1_ref, Vq_2_ref,
                Vr_1_ref, Vr_2_ref,
                Vs_1_ref, Vs_2_ref,
                dropout_rate
            )
            torch.cuda.synchronize()
            fwd_time_ref = time.perf_counter() - start_time
            total_time_pytorch_ref += fwd_time_ref

            # Backward
            loss = (Y_q_ref.sum() + Y_r_ref.sum() + Y_s_ref.sum() + 
                    Y_q__ref.sum() + Y_r__ref.sum() + Y_s__ref.sum())
            
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize()
            bwd_time_ref = time.perf_counter() - start_time
            total_time_pytorch_ref += bwd_time_ref
            print(f"{total_time_pytorch_ref * 1000:.4f} ms".ljust(25))

        except Exception as e:
            print(f"PyTorch Ref Error: {e}".ljust(25))
            total_time_pytorch_ref = float('inf')

        # Clear gradients for next iteration if any tensors were reused with requires_grad=True
        # (cloning inputs each time, so not strictly necessary here but good practice)
        if Q_ref.grad is not None: Q_ref.grad.zero_()
        if R_ref.grad is not None: R_ref.grad.zero_()
        # ... and so on for all grad-requiring tensors

    print("-" * 80)

if __name__ == '__main__':
    # Before running, ensure the extensions are compiled:
    # python setup.py install (or build_ext --inplace)
    # Make sure the compiled .so files are in the python path or current directory.
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Aborting benchmark.")
    else:
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print("Compiling extensions if not already done...")
        print("You might need to run 'python setup.py install' or 'python setup.py build_ext --inplace' first.")
        try:
            # A simple check to see if modules can be imported
            # This doesn't guarantee they are correctly built for the current env.
            if 'hyper_attn_cpp_manual' not in globals() or 'hyper_attn_cpp_reference' not in globals():
                 raise ImportError("Custom extensions not found. Please build them first.")
            benchmark()
        except ImportError as e:
            print(f"ImportError: {e}")
            print("Please ensure the extensions are compiled and accessible.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
