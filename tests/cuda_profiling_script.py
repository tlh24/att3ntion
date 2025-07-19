import torch
import hyper_attn_cpp_manual

def get_grad_output_cuda(Y_q, Y_r, Y_s, Y_q_, Y_r_, Y_s_):
    """
    Creates a gradient tensor with the correct shape to drive the backward pass.
    For profiling, the values don't matter, only the shape and device.
    """
    B, H, I, D = Y_q.shape
    _, _, J, _ = Y_r.shape
    _, _, K, _ = Y_s.shape
    
    max_len = max(I, J, K)
    
    # The backward pass expects a single gradient tensor, so we create one
    # that is large enough and accumulate dummy gradients into it.
    grad_output_combined = torch.zeros(B, H, max_len, D, device=Y_q.device, dtype=Y_q.dtype)
    
    grad_output_combined[:, :, :I, :] += 1.0
    grad_output_combined[:, :, :J, :] += 1.0
    grad_output_combined[:, :, :K, :] += 1.0
    
    return grad_output_combined

def main():
    """
    Runs a single configuration of the manual CUDA attention mechanism
    for targeted profiling with tools like NVIDIA Nsight Compute (ncu).
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. Aborting.")
        return

    print(f"Using device: {torch.cuda.get_device_name(0)}")

    # --- Configuration ---
    # Hardcoded single configuration for a focused profiling run
    B, H, I_dim, J_dim, K_dim, D_dim = (1, 2, 128, 128, 128, 64)
    dropout_rate = 0.0
    print(f"Profiling config (B,H,I,J,K,D): ({B},{H},{I_dim},{J_dim},{K_dim},{D_dim})")

    # --- Tensor Initialization ---
    Q = torch.rand(B, H, I_dim, D_dim, device='cuda', dtype=torch.float32)
    R = torch.rand(B, H, J_dim, D_dim, device='cuda', dtype=torch.float32)
    S = torch.rand(B, H, K_dim, D_dim, device='cuda', dtype=torch.float32)
    Vq_1 = torch.rand(B, H, I_dim, D_dim, device='cuda', dtype=torch.float32)
    Vq_2 = torch.rand(B, H, I_dim, D_dim, device='cuda', dtype=torch.float32)
    Vr_1 = torch.rand(B, H, J_dim, D_dim, device='cuda', dtype=torch.float32)
    Vr_2 = torch.rand(B, H, J_dim, D_dim, device='cuda', dtype=torch.float32)
    Vs_1 = torch.rand(B, H, K_dim, D_dim, device='cuda', dtype=torch.float32)
    Vs_2 = torch.rand(B, H, K_dim, D_dim, device='cuda', dtype=torch.float32)

    # --- Execution ---
    # The profiler (ncu) hooks into the CUDA API, so we just need to execute the functions.
    
    # Forward Pass
    Y_q_mc, Y_r_mc, Y_s_mc, Y_q__mc, Y_r__mc, Y_s__mc = hyper_attn_cpp_manual.forward(
        Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2, dropout_rate
    )

    # Create dummy gradient for the backward pass
    grad_output_cuda = get_grad_output_cuda(Y_q_mc, Y_r_mc, Y_s_mc, Y_q__mc, Y_r__mc, Y_s__mc)
    
    # Backward Pass
    hyper_attn_cpp_manual.backward(
        grad_output_cuda, 
        Q, R, S,
        Vq_1, Vq_2,
        Vr_1, Vr_2,
        Vs_1, Vs_2,
        dropout_rate
    )

    # This final synchronize is important to ensure the GPU has finished all work
    # before the script exits, guaranteeing the profiler captures everything.
    torch.cuda.synchronize()

    print("\nCUDA profiling script finished successfully.")

if __name__ == '__main__':
    try:
        main()
    except ImportError:
        print("\nImportError: Could not import 'hyper_attn_cpp_manual'.")
        print("Please ensure the extension is compiled via 'python setup.py install' or 'develop'.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
