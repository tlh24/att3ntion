import torch
import hyper_attn_cpp_manual
import subprocess
import sys
import datetime
import os
import argparse

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

def run_kernel_pass(B, H, I_dim, J_dim, K_dim, D_dim):
    """
    Contains the core logic for running the CUDA kernels.
    This is what the profiler will measure.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. Aborting.")
        return

    print(f"Using device: {torch.cuda.get_device_name(0)}")
    print(f"Profiling config (B,H,I,J,K,D): ({B},{H},{I_dim},{J_dim},{K_dim},{D_dim})")

    dropout_rate = 0.0
    
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
    torch.cuda.synchronize()
    print("\nCUDA kernel run finished successfully.")


def launch_profiler(report_filename=None):
    """
    Constructs and launches the ncu profiling command.
    """
    # --- Configuration ---
    # You can now change these values directly in the script for a new run
    B, H, I_dim, J_dim, K_dim, D_dim = (1, 2, 128, 128, 128, 64)
    KERNEL_NAME = "Yq_gather_flash"
    
    # Dynamically generate the report filename
    if report_filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"profiling_reports/{KERNEL_NAME}_B{B}_H{H}_I{I_dim}_J{J_dim}_K{K_dim}_D{D_dim}_{timestamp}.ncu-rep"
    else:
        output_filename = report_filename

    dirpath = os.path.dirname(output_filename)
    if dirpath == "":
        dirpath = "profiling_reports"
        output_filename = os.path.join(dirpath, output_filename)

    os.makedirs(dirpath, exist_ok=True)

    # Construct the ncu command
    ncu_command = [
        "ncu",
        "--set", "full",
        "-k", KERNEL_NAME,
        "--section", "SchedulerStats",
        "--section", "InstructionStats",
        "--section", "MemoryWorkloadAnalysis",
        "-o", output_filename,
        sys.executable,  # Use the current python executable
        __file__,       # Pass the script itself as the target
        "--no-profile"  # Flag to tell the script not to launch ncu again
    ]

    print(f"Executing profiling command: {' '.join(ncu_command)}")
    # Use subprocess to run the command
    subprocess.run(ncu_command, check=True)
    print("\nCUDA profiling script finished successfully.")


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description="Run CUDA profiling script.")
        parser.add_argument("--no-profile", action="store_true",
                            help="Run kernels directly without launching ncu profiler.")
        parser.add_argument("--output-file", type=str, default=None,
                            help="Specify the output filename for the ncu report.")
        args = parser.parse_args()

        # Check if we should run the kernels directly or launch the profiler
        if args.no_profile:
            # This branch is executed when ncu calls the script
            B, H, I_dim, J_dim, K_dim, D_dim = (1, 2, 128, 128, 128, 64)
            run_kernel_pass(B, H, I_dim, J_dim, K_dim, D_dim)
        else:
            # This is the main branch that launches the profiler
            launch_profiler(report_filename=args.output_file)

    except ImportError:
        print("\nImportError: Could not import 'hyper_attn_cpp_manual'.")
        print("Please ensure the extension is compiled via 'python setup.py install' or 'develop'.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")