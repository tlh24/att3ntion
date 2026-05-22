import torch
import att3ntion._cuda_kernels as _cuda_kernels
import subprocess
import sys
import datetime
import os
import argparse

# ── Complete list of all CUDA kernels in the project ──
ALL_KERNELS = [
    # Forward gather
    "Yq_gather", "Yq_gather_tensor_core", "Yr_gather", "Ys_gather",
    # Forward scatter
    "Yq_scatter", "Yr_scatter", "Ys_scatter",
    # Backward V gradients (gather)
    "Vq_gather_grad", "Vr_gather_grad", "Vs_gather_grad",
    # Backward V gradients (scatter)
    "Vq_scatter_grad", "Vr_scatter_grad", "Vs_scatter_grad",
    # Backward Jacobian + query/key gradients
    "jacobian_corrections", "QS_grad_fused", "R_grad",
]

DEFAULT_DIMS = (1, 2, 128, 128, 128, 64)


def get_grad_outputs_cuda(Y_q, Y_r, Y_s):
    """
    Create backward gradients for Y_q, Y_r, and Y_s.
    For profiling, values don't matter; shapes and dtypes must match.
    """
    grad_Y_q = torch.ones_like(Y_q)
    grad_Y_r = torch.ones_like(Y_r)
    grad_Y_s = torch.ones_like(Y_s)
    return grad_Y_q, grad_Y_r, grad_Y_s


def run_kernel_pass(B, H, I_dim, J_dim, K_dim, D_dim,
                    forward_only=False, backward_only=False):
    """
    Core logic for running the CUDA kernels. This is what ncu measures.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. Aborting.")
        return

    print(f"Using device: {torch.cuda.get_device_name(0)}")
    print(f"Profiling config (B,H,I,J,K,D): ({B},{H},{I_dim},{J_dim},{K_dim},{D_dim})")

    dropout_rate = 0.0
    dtype = torch.float32

    # --- Tensor Initialization ---
    Q    = torch.rand(B, H, I_dim, D_dim, device='cuda', dtype=dtype)
    R    = torch.rand(B, H, J_dim, D_dim, device='cuda', dtype=dtype)
    S    = torch.rand(B, H, K_dim, D_dim, device='cuda', dtype=dtype)
    Vq_1 = torch.rand(B, H, I_dim, D_dim, device='cuda', dtype=dtype)
    Vq_2 = torch.rand(B, H, I_dim, D_dim, device='cuda', dtype=dtype)
    Vr_1 = torch.rand(B, H, J_dim, D_dim, device='cuda', dtype=dtype)
    Vr_2 = torch.rand(B, H, J_dim, D_dim, device='cuda', dtype=dtype)
    Vs_1 = torch.rand(B, H, K_dim, D_dim, device='cuda', dtype=dtype)
    Vs_2 = torch.rand(B, H, K_dim, D_dim, device='cuda', dtype=dtype)

    fwd_inputs = tuple(t.to(torch.bfloat16) for t in (Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2))
    bwd_inputs = fwd_inputs

    # --- Forward Pass ---
    # forward returns 12 values: Y_q, Y_r, Y_s, Y_q_, Y_r_, Y_s_,
    #                             m_i, l_i, m_j, l_j, m_k, l_k
    if not backward_only:
        print("Running forward pass...")
        (Y_q_mc, Y_r_mc, Y_s_mc, Y_q__mc, Y_r__mc, Y_s__mc,
         m_i, l_i, m_j, l_j, m_k, l_k) = \
            _cuda_kernels.forward(
                *fwd_inputs, dropout_rate
            )
    else:
        # Still need forward outputs to feed backward
        with torch.no_grad():
            (Y_q_mc, Y_r_mc, Y_s_mc, Y_q__mc, Y_r__mc, Y_s__mc,
             m_i, l_i, m_j, l_j, m_k, l_k) = \
                _cuda_kernels.forward(
                    *fwd_inputs, dropout_rate
                )

    # --- Backward Pass ---
    if not forward_only:
        print("Running backward pass...")
        grad_Y_q, grad_Y_r, grad_Y_s = get_grad_outputs_cuda(Y_q_mc, Y_r_mc, Y_s_mc)
        grad_Y_q_, grad_Y_r_, grad_Y_s_ = get_grad_outputs_cuda(Y_q__mc, Y_r__mc, Y_s__mc)
        _cuda_kernels.backward(
            grad_Y_q,
            grad_Y_r,
            grad_Y_s,
            grad_Y_q_,
            grad_Y_r_,
            grad_Y_s_,
            *bwd_inputs,
            m_i, l_i,
            m_j, l_j,
            m_k, l_k,
            dropout_rate
        )

    torch.cuda.synchronize()
    print("\nCUDA kernel run finished successfully.")


def launch_profiler(args):
    """
    Constructs and launches the ncu profiling command.
    """
    B, H, I_dim, J_dim, K_dim, D_dim = args.dims

    # ── Build output filename ──
    if args.output_file:
        output_filename = args.output_file
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        kernel_tag = args.kernel if args.kernel else "all_kernels"
        output_filename = (
            f"profiling_reports/"
            f"{kernel_tag}_B{B}_H{H}_I{I_dim}_J{J_dim}_K{K_dim}_D{D_dim}_{timestamp}"
        )

    # Ensure it lives under profiling_reports/ if no directory was given
    if not os.path.dirname(output_filename):
        output_filename = os.path.join("profiling_reports", output_filename)

    # Strip .ncu-rep if user added it (ncu adds it automatically with -o)
    if output_filename.endswith(".ncu-rep"):
        output_filename = output_filename[:-len(".ncu-rep")]

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    # ── Build ncu command ──
    ncu_command = [
        "ncu",
        "--set", "full",
        "--section", "SchedulerStats",
        "--section", "InstructionStats",
        "--section", "MemoryWorkloadAnalysis",
        "-o", output_filename,
    ]

    # Only filter to a specific kernel if one was requested
    if args.kernel:
        ncu_command += ["-k", args.kernel]

    # Target python script + passthrough args
    ncu_command += [
        sys.executable,
        __file__,
        "--no-profile",
        "--dims", ",".join(str(x) for x in args.dims),
    ]
    if args.forward_only:
        ncu_command.append("--forward-only")
    if args.backward_only:
        ncu_command.append("--backward-only")

    print(f"\n{'='*70}")
    print(f"  NCU Profiling Command")
    print(f"{'='*70}")
    print(f"  {' '.join(ncu_command)}")
    print(f"  Report will be saved to: {output_filename}.ncu-rep")
    print(f"{'='*70}\n")

    subprocess.run(ncu_command, check=True)
    print(f"\n✅ Profiling complete! Report: {output_filename}.ncu-rep")
    print(f"   Open on your laptop with:  ncu-ui {output_filename}.ncu-rep")


def parse_dims(s):
    """Parse 'B,H,I,J,K,D' string into a tuple of 6 ints."""
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError("--dims must be 6 comma-separated ints: B,H,I,J,K,D")
    return tuple(parts)


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(
            description="Profile CUDA hypergraph attention kernels with NVIDIA Nsight Compute.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Profile ALL kernels (forward + backward), auto-named report:
  python %(prog)s

  # Profile a single kernel:
  python %(prog)s --kernel QS_grad_fused

  # Custom report name:
  python %(prog)s --output-file my_experiment_v2

  # Custom dimensions:
  python %(prog)s --dims 2,4,256,256,256,64

  # Forward-only profiling:
  python %(prog)s --forward-only

  # Combine options:
  python %(prog)s --kernel Yq_gather --dims 1,8,512,512,512,64 --output-file big_gather_test

Available kernels:
  Forward:  Yq_gather, Yq_gather_tensor_core, Yr_gather, Ys_gather,
            Yq_scatter, Yr_scatter, Ys_scatter
  Backward: Vq_gather_grad, Vr_gather_grad, Vs_gather_grad,
            Vq_scatter_grad, Vr_scatter_grad, Vs_scatter_grad,
            jacobian_corrections, QS_grad_fused, R_grad
""")

        parser.add_argument("--no-profile", action="store_true",
                            help="(Internal) Run kernels directly without launching ncu.")
        parser.add_argument("--kernel", "-k", type=str, default=None,
                            help="Profile only this kernel. Omit to profile ALL kernels.")
        parser.add_argument("--output-file", "-o", type=str, default=None,
                            help="Custom report filename (without .ncu-rep extension).")
        parser.add_argument("--dims", type=parse_dims,
                            default=DEFAULT_DIMS,
                            help="Tensor dimensions as B,H,I,J,K,D (default: 1,2,128,128,128,64)")
        parser.add_argument("--forward-only", action="store_true",
                            help="Only run forward pass kernels.")
        parser.add_argument("--backward-only", action="store_true",
                            help="Only run backward pass kernels.")

        args = parser.parse_args()

        if args.forward_only and args.backward_only:
            parser.error("Cannot use --forward-only and --backward-only together.")

        if args.no_profile:
            # This branch is executed when ncu calls the script
            B, H, I_dim, J_dim, K_dim, D_dim = args.dims
            run_kernel_pass(B, H, I_dim, J_dim, K_dim, D_dim,
                            forward_only=args.forward_only,
                            backward_only=args.backward_only)
        else:
            launch_profiler(args)

    except ImportError:
        print("\nImportError: Could not import 'att3ntion._cuda_kernels'.")
        print("Please ensure the extension is compiled via 'python setup.py install' or 'develop'.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        raise
