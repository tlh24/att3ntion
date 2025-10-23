import os
import sys
import math
import argparse
import torch


# Make project root importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def compute_reference_grads(Q, R, S,
                            Vq_1, Vq_2,
                            Vr_1, Vr_2,
                            Vs_1, Vs_2,
                            grad_output):
    """Compute ground-truth grads using PyTorch autograd.

    This mirrors the math used by the CUDA extension:
      - A = einsum('bhid,bhjd,bhkd->bhijk', Q, R, S) / sqrt(D)
      - Aq = softmax over (j,k)
      - Ar = softmax over (i,k)
      - As = softmax over (i,j)
      - Gather Y: Y_q, Y_r, Y_s
      - Scatter Y: Y_q_, Y_r_, Y_s_ using elementwise products of A* A*

    The scalar loss is the dot of each Y_* with grad_output sliced to the
    appropriate sequence length for that output.
    """
    B, H, I, D = Q.shape
    J = R.size(2)
    K = S.size(2)

    # Clone and enable grad on reference tensors to avoid polluting inputs
    Q_ref   = Q.detach().clone().requires_grad_(True)
    R_ref   = R.detach().clone().requires_grad_(True)
    S_ref   = S.detach().clone().requires_grad_(True)
    Vq1_ref = Vq_1.detach().clone().requires_grad_(True)
    Vq2_ref = Vq_2.detach().clone().requires_grad_(True)
    Vr1_ref = Vr_1.detach().clone().requires_grad_(True)
    Vr2_ref = Vr_2.detach().clone().requires_grad_(True)
    Vs1_ref = Vs_1.detach().clone().requires_grad_(True)
    Vs2_ref = Vs_2.detach().clone().requires_grad_(True)

    scale = 1.0 / math.sqrt(float(D))
    A = torch.einsum('bhid,bhjd,bhkd->bhijk', Q_ref, R_ref, S_ref) * scale

    # Aq: softmax over (j,k)
    Aq = torch.softmax(A.flatten(3, 4), dim=-1).reshape_as(A)

    # Ar: softmax over (i,k)
    A_r = A.permute(0, 1, 3, 2, 4)                          # [B,H,J,I,K]
    Ar = torch.softmax(A_r.flatten(3, 4), dim=-1).reshape_as(A_r)
    Ar = Ar.permute(0, 1, 3, 2, 4)                           # [B,H,I,J,K]

    # As: softmax over (i,j)
    A_s = A.permute(0, 1, 4, 2, 3)                           # [B,H,K,I,J]
    As = torch.softmax(A_s.flatten(3, 4), dim=-1).reshape_as(A_s)
    As = As.permute(0, 1, 3, 4, 2)                           # [B,H,I,J,K]

    # Gather outputs
    Y_q  = torch.einsum('bhijk,bhjd,bhkd->bhid', Aq, Vr1_ref, Vs1_ref)
    Y_r  = torch.einsum('bhijk,bhid,bhkd->bhjd', Ar, Vq1_ref, Vs1_ref)
    Y_s  = torch.einsum('bhijk,bhid,bhjd->bhkd', As, Vq1_ref, Vr1_ref)

    # Scatter outputs (elementwise products of attention maps)
    Y_q_ = torch.einsum('bhijk,bhjd,bhkd->bhid', Ar * As, Vr2_ref, Vs2_ref)
    Y_r_ = torch.einsum('bhijk,bhid,bhkd->bhjd', Aq * As, Vq2_ref, Vs2_ref)
    Y_s_ = torch.einsum('bhijk,bhid,bhjd->bhkd', Aq * Ar, Vq2_ref, Vr2_ref)

    # Slice grad_output to each output's length
    T_i = grad_output[:, :, :I, :]  # for Y_q, Y_q_
    T_j = grad_output[:, :, :J, :]  # for Y_r, Y_r_
    T_k = grad_output[:, :, :K, :]  # for Y_s, Y_s_

    loss = (
        (Y_q  * T_i).sum() +
        (Y_r  * T_j).sum() +
        (Y_s  * T_k).sum() +
        (Y_q_ * T_i).sum() +
        (Y_r_ * T_j).sum() +
        (Y_s_ * T_k).sum()
    )

    loss.backward()

    grads = (
        Q_ref.grad, R_ref.grad, S_ref.grad,
        Vq1_ref.grad, Vq2_ref.grad,
        Vr1_ref.grad, Vr2_ref.grad,
        Vs1_ref.grad, Vs2_ref.grad,
    )
    return grads


def compute_manual_grads(ext_mod,
                         Q, R, S,
                         Vq_1, Vq_2,
                         Vr_1, Vr_2,
                         Vs_1, Vs_2,
                         grad_output,
                         dropout_rate=0.0):
    return ext_mod.backward(
        grad_output, Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2, dropout_rate
    )


def compare_tensors(name, a, b, rtol, atol):
    if a is None or b is None:
        return False, float('nan')
    same = torch.allclose(a, b, rtol=rtol, atol=atol)
    max_abs = (a - b).abs().max().item()
    return same, max_abs


def main():
    parser = argparse.ArgumentParser("CUDA backward equivalence vs PyTorch reference")
    parser.add_argument('--B', type=int, default=1)
    parser.add_argument('--H', type=int, default=2)
    parser.add_argument('--I', type=int, default=8)
    parser.add_argument('--J', type=int, default=8)
    parser.add_argument('--K', type=int, default=8)
    parser.add_argument('--D', type=int, default=32)
    parser.add_argument('--rtol', type=float, default=1e-4)
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    try:
        import hyper_attn_cpp_manual as manual_ext
    except ImportError as e:
        print(f"Error: Could not import 'hyper_attn_cpp_manual': {e}")
        print("Please compile the extension first: python setup.py develop")
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    B, H, I, J, K, D = args.B, args.H, args.I, args.J, args.K, args.D

    # Inputs
    Q    = torch.randn(B, H, I, D, device=device, dtype=torch.float32)
    R    = torch.randn(B, H, J, D, device=device, dtype=torch.float32)
    S    = torch.randn(B, H, K, D, device=device, dtype=torch.float32)
    Vq_1 = torch.randn(B, H, I, D, device=device, dtype=torch.float32)
    Vq_2 = torch.randn(B, H, I, D, device=device, dtype=torch.float32)
    Vr_1 = torch.randn(B, H, J, D, device=device, dtype=torch.float32)
    Vr_2 = torch.randn(B, H, J, D, device=device, dtype=torch.float32)
    Vs_1 = torch.randn(B, H, K, D, device=device, dtype=torch.float32)
    Vs_2 = torch.randn(B, H, K, D, device=device, dtype=torch.float32)

    max_len = max(I, J, K)
    grad_output = torch.randn(B, H, max_len, D, device=device, dtype=torch.float32)

    # Compute manual (CUDA) grads
    torch.cuda.synchronize() if device.type == 'cuda' else None
    grads_manual = compute_manual_grads(
        manual_ext, Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2, grad_output
    )

    # Compute reference (autograd) grads
    grads_ref = compute_reference_grads(
        Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2, grad_output
    )

    names = [
        'grad_Q', 'grad_R', 'grad_S',
        'grad_Vq_1', 'grad_Vq_2',
        'grad_Vr_1', 'grad_Vr_2',
        'grad_Vs_1', 'grad_Vs_2'
    ]

    print("\nBackward Equivalence Check (CUDA vs Reference)")
    print(f"Config: B={B}, H={H}, I={I}, J={J}, K={K}, D={D}")
    print(f"Tolerances: rtol={args.rtol}, atol={args.atol}")
    print("-" * 80)

    all_ok = True
    for name, gm, gr in zip(names, grads_manual, grads_ref):
        ok, max_abs = compare_tensors(name, gm, gr, args.rtol, args.atol)
        status = 'PASS' if ok else 'FAIL'
        print(f"{name:<10} | {status:<4} | max_abs_diff: {max_abs:.3e}")
        all_ok = all_ok and ok

    print("-" * 80)
    if all_ok:
        print("All gradients match within tolerance.")
        sys.exit(0)
    else:
        print("Gradient mismatches detected.")
        sys.exit(2)


if __name__ == '__main__':
    main()


