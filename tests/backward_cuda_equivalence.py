import os
import sys
import math
import argparse
from typing import Tuple
import torch


# Make project root importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ==============================================================================
# Reference Implementation for grad_Q Intermediate Sums
# (Based on user-provided Python snippet)
# ==============================================================================

# def compute_softmax_stats(Q, R, S, scale):
#     """Computes m and l statistics for online softmax.
    
#     Expects 2D tensors [N, D] for single batch/head.
#     """
#     I, D = Q.shape
#     J = R.size(0)
#     K = S.size(0)

#     A = torch.einsum('id,jd,kd->ijk', Q, R, S) * scale

#     # Stats for Aq (i-centric, softmax over j,k)
#     A_for_q = A.flatten(1, 2)  # [I, J*K]
#     m_i = A_for_q.max(dim=-1).values  # [I]
#     l_i = torch.exp(A_for_q - m_i.unsqueeze(-1)).sum(dim=-1)  # [I]

#     # Stats for Ar (j-centric, softmax over i,k)
#     A_for_r = A.permute(1, 0, 2).flatten(1, 2)  # [J, I*K]
#     m_j = A_for_r.max(dim=-1).values  # [J]
#     l_j = torch.exp(A_for_r - m_j.unsqueeze(-1)).sum(dim=-1)  # [J]

#     # Stats for As (k-centric, softmax over i,j)
#     A_for_s = A.permute(2, 0, 1).flatten(1, 2)  # [K, I*J]
#     m_k = A_for_s.max(dim=-1).values  # [K]
#     l_k = torch.exp(A_for_s - m_k.unsqueeze(-1)).sum(dim=-1)  # [K]

#     return (m_i, l_i, m_j, l_j, m_k, l_k)

# def _numerators(logits, mi, mj, mk, li, lj, lk):
#     """
#     logits : [Bq, Br, Bk]
#     Broadcasting shapes:
#         mi,li : [Bq, 1 , 1]
#         mj,lj : [1 , Br, 1]
#         mk,lk : [1 , 1 , Bk]
#     returns: Aq_tile , Ar_tile , As_tile   (all same shape as logits)
#     """
#     Aq = torch.exp(logits - mi) / li
#     Ar = torch.exp(logits - mj) / lj
#     As = torch.exp(logits - mk) / lk
#     return Aq, Ar, As

# def _pass1(Q,R,S, Vq1,Vq2, Vr1,Vr2, Vs1,Vs2,
#            grad_out, m_i,l_i, m_j,l_j, m_k,l_k,
#            scale, Bq,Br,Bk):
#     N, D = Q.shape
#     # 1-D accumulators for the ∑ grad_A* * A* terms
#     sum_q = torch.zeros(N, device=Q.device, dtype=Q.dtype)
#     sum_r = torch.zeros_like(sum_q)
#     sum_s = torch.zeros_like(sum_q)

#     # convenience views for broadcasting
#     m_i = m_i.view(N,1,1); l_i = l_i.view(N,1,1)
#     m_j = m_j.view(1,N,1); l_j = l_j.view(1,N,1)
#     m_k = m_k.view(1,1,N); l_k = l_k.view(1,1,N)

#     for k0 in range(0, N, Bk):
#         k1 = min(k0+Bk, N)
#         Sk   = S   [k0:k1]
#         Vs1k = Vs1 [k0:k1]
#         Vs2k = Vs2 [k0:k1]
#         dYk  = grad_out[k0:k1]
#         mk, lk = m_k[...,k0:k1], l_k[...,k0:k1]

#         for j0 in range(0, N, Br):
#             j1 = min(j0+Br, N)
#             Rj   = R   [j0:j1]
#             Vr1j = Vr1 [j0:j1]
#             Vr2j = Vr2 [j0:j1]
#             dYj  = grad_out[j0:j1]
#             mj, lj = m_j[:, j0:j1, :], l_j[:, j0:j1, :]

#             for i0 in range(0, N, Bq):
#                 i1 = min(i0+Bq, N)
#                 Qi   = Q   [i0:i1]
#                 Vq1i = Vq1 [i0:i1]
#                 Vq2i = Vq2 [i0:i1]
#                 dYi  = grad_out[i0:i1]
#                 mi, li = m_i[i0:i1], l_i[i0:i1]

#                 logits = torch.einsum('id,jd,kd->ijk', Qi,Rj,Sk) * scale
#                 Aq, Ar, As = _numerators(logits, mi,mj,mk, li,lj,lk)

#                 gAq  = torch.einsum('id,jd,kd->ijk', dYi, Vr1j, Vs1k)
#                 gAq += torch.einsum('jd,id,kd->ijk', dYj, Vq2i, Vs2k) * As
#                 gAq += torch.einsum('kd,id,jd->ijk', dYk, Vq2i, Vr2j) * Ar

#                 gAr  = torch.einsum('jd,id,kd->ijk', dYj, Vq1i, Vs1k)
#                 gAr += torch.einsum('id,jd,kd->ijk', dYi, Vr2j, Vs2k) * As
#                 gAr += torch.einsum('kd,id,jd->ijk', dYk, Vq2i, Vr2j) * Aq

#                 gAs  = torch.einsum('kd,id,jd->ijk', dYk, Vq1i, Vr1j)
#                 gAs += torch.einsum('id,jd,kd->ijk', dYi, Vr2j, Vs2k) * Ar
#                 gAs += torch.einsum('jd,id,kd->ijk', dYj, Vq2i, Vs2k) * Aq

#                 sum_q[i0:i1] += (gAq * Aq).sum(dim=(1,2))
#                 sum_r[j0:j1] += (gAr * Ar).sum(dim=(0,2))
#                 sum_s[k0:k1] += (gAs * As).sum(dim=(0,1))

#     return sum_q, sum_r, sum_s

# ==============================================================================
# End of Reference Implementation
# ==============================================================================


try:
    import hyper_attn_cpp_reference as baseline_ext
except ImportError:
    baseline_ext = None


def compute_reference_grads(Q, R, S,
                            Vq_1, Vq_2,
                            Vr_1, Vr_2,
                            Vs_1, Vs_2,
                            grad_output):
    """Compute reference gradients via the Torch-only C++ extension.

    We call `hyper_attn_cpp_reference.forward`, then back-propagate a synthetic
    loss identical to the one used for the CUDA kernel so the two gradient sets
    are directly comparable.
    """

    if baseline_ext is None:
        raise RuntimeError("hyper_attn_cpp_reference is not available; build the extension first (python setup.py develop)")

    B, H, I, D = Q.shape
    J = R.size(2)
    K = S.size(2)

    # Clone tensors and enable grad
    tensors = [Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2]
    tensors = [t.detach().clone().requires_grad_(True) for t in tensors]
    (Q_ref, R_ref, S_ref,
     Vq1_ref, Vq2_ref,
     Vr1_ref, Vr2_ref,
     Vs1_ref, Vs2_ref) = tensors

    # Forward pass through reference extension; returns 6 tensors
    outputs = baseline_ext.forward(
        Q_ref, R_ref, S_ref,
        Vq1_ref, Vq2_ref,
        Vr1_ref, Vr2_ref,
        Vs1_ref, Vs2_ref
    )

    if not (isinstance(outputs, tuple) and len(outputs) == 6):
        raise ValueError("Expected 6-tensor output from reference extension")

    Y_q, Y_r, Y_s, Y_q_, Y_r_, Y_s_ = outputs

    # Synthetic loss identical to CUDA path
    T_i = grad_output[:, :, :I, :]
    T_j = grad_output[:, :, :J, :]
    T_k = grad_output[:, :, :K, :]

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
                         dropout_rate=0.0) -> Tuple[torch.Tensor, ...]:
    """Calls the backward pass and returns all 9 output tensors."""
    return ext_mod.backward(
        grad_output, Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2, dropout_rate
    )


def verify_softmax_normalization(Q, R, S, m_i, l_i, m_j, l_j, m_k, l_k, scale, atol=1e-4):
    """
    Verify that the softmax probabilities computed from m/l stats sum to 1.0.
    
    For 3D attention:
    - Aq: sum_{j,k} exp(A[i,j,k] - m_i[i]) / l_i[i] should = 1.0 for all i
    - Ar: sum_{i,k} exp(A[i,j,k] - m_j[j]) / l_j[j] should = 1.0 for all j
    - As: sum_{i,j} exp(A[i,j,k] - m_k[k]) / l_k[k] should = 1.0 for all k
    
    Returns a dict with verification results.
    """
    B, H, I, D = Q.shape
    J = R.size(2)
    K = S.size(2)
    
    all_ok = True
    results = {}
    
    for b in range(B):
        for h in range(H):
            # Compute the full 3D attention logits A[i,j,k]
            Qbh = Q[b, h]  # [I, D]
            Rbh = R[b, h]  # [J, D]
            Sbh = S[b, h]  # [K, D]
            
            # A[i,j,k] = scale * sum_d(Q[i,d] * R[j,d] * S[k,d])
            A = torch.einsum('id,jd,kd->ijk', Qbh, Rbh, Sbh) * scale
            
            # --- Check Aq normalization (softmax over j,k for each i) ---
            mi = m_i[b, h]  # [I]
            li = l_i[b, h]  # [I]
            
            # Aq[i,j,k] = exp(A[i,j,k] - m_i[i]) / l_i[i]
            # Sum should be 1.0 for each i
            Aq_sum = torch.zeros(I, device=Q.device, dtype=Q.dtype)
            for i in range(I):
                Aq_sum[i] = torch.exp(A[i, :, :] - mi[i]).sum() / li[i]
            
            aq_ok = torch.allclose(Aq_sum, torch.ones_like(Aq_sum), atol=atol)
            aq_max_dev = (Aq_sum - 1.0).abs().max().item()
            
            # --- Check Ar normalization (softmax over i,k for each j) ---
            mj = m_j[b, h]  # [J]
            lj = l_j[b, h]  # [J]
            
            Ar_sum = torch.zeros(J, device=Q.device, dtype=Q.dtype)
            for j in range(J):
                Ar_sum[j] = torch.exp(A[:, j, :] - mj[j]).sum() / lj[j]
            
            ar_ok = torch.allclose(Ar_sum, torch.ones_like(Ar_sum), atol=atol)
            ar_max_dev = (Ar_sum - 1.0).abs().max().item()
            
            # --- Check As normalization (softmax over i,j for each k) ---
            mk = m_k[b, h]  # [K]
            lk = l_k[b, h]  # [K]
            
            As_sum = torch.zeros(K, device=Q.device, dtype=Q.dtype)
            for k in range(K):
                As_sum[k] = torch.exp(A[:, :, k] - mk[k]).sum() / lk[k]
            
            as_ok = torch.allclose(As_sum, torch.ones_like(As_sum), atol=atol)
            as_max_dev = (As_sum - 1.0).abs().max().item()
            
            results[(b, h)] = {
                'Aq_ok': aq_ok, 'Aq_max_dev': aq_max_dev,
                'Ar_ok': ar_ok, 'Ar_max_dev': ar_max_dev,
                'As_ok': as_ok, 'As_max_dev': as_max_dev,
            }
            
            if not (aq_ok and ar_ok and as_ok):
                all_ok = False
    
    return all_ok, results


def compare_softmax_stats_with_reference(Q, R, S, m_i, l_i, m_j, l_j, m_k, l_k, scale, rtol=1e-4, atol=1e-5):
    """
    Compare CUDA-computed softmax stats (m, l) against PyTorch reference.
    
    This catches bugs in the online softmax tiling/accumulation.
    """
    B, H, I, D = Q.shape
    J = R.size(2)
    K = S.size(2)
    
    all_ok = True
    results = {}
    
    for b in range(B):
        for h in range(H):
            # Compute full logits
            A = torch.einsum('id,jd,kd->ijk', Q[b,h], R[b,h], S[b,h]) * scale
            
            # --- Aq stats (softmax over j,k for each i) ---
            A_for_q = A.flatten(1, 2)  # [I, J*K]
            m_i_ref = A_for_q.max(dim=-1).values
            l_i_ref = torch.exp(A_for_q - m_i_ref.unsqueeze(-1)).sum(dim=-1)
            
            mi_ok = torch.allclose(m_i[b,h], m_i_ref, rtol=rtol, atol=atol)
            li_ok = torch.allclose(l_i[b,h], l_i_ref, rtol=rtol, atol=atol)
            mi_max_diff = (m_i[b,h] - m_i_ref).abs().max().item()
            li_max_diff = (l_i[b,h] - l_i_ref).abs().max().item()
            
            # --- Ar stats (softmax over i,k for each j) ---
            A_for_r = A.permute(1, 0, 2).flatten(1, 2)  # [J, I*K]
            m_j_ref = A_for_r.max(dim=-1).values
            l_j_ref = torch.exp(A_for_r - m_j_ref.unsqueeze(-1)).sum(dim=-1)
            
            mj_ok = torch.allclose(m_j[b,h], m_j_ref, rtol=rtol, atol=atol)
            lj_ok = torch.allclose(l_j[b,h], l_j_ref, rtol=rtol, atol=atol)
            mj_max_diff = (m_j[b,h] - m_j_ref).abs().max().item()
            lj_max_diff = (l_j[b,h] - l_j_ref).abs().max().item()
            
            # --- As stats (softmax over i,j for each k) ---
            A_for_s = A.permute(2, 0, 1).flatten(1, 2)  # [K, I*J]
            m_k_ref = A_for_s.max(dim=-1).values
            l_k_ref = torch.exp(A_for_s - m_k_ref.unsqueeze(-1)).sum(dim=-1)
            
            mk_ok = torch.allclose(m_k[b,h], m_k_ref, rtol=rtol, atol=atol)
            lk_ok = torch.allclose(l_k[b,h], l_k_ref, rtol=rtol, atol=atol)
            mk_max_diff = (m_k[b,h] - m_k_ref).abs().max().item()
            lk_max_diff = (l_k[b,h] - l_k_ref).abs().max().item()
            
            results[(b, h)] = {
                'm_i_ok': mi_ok, 'm_i_diff': mi_max_diff,
                'l_i_ok': li_ok, 'l_i_diff': li_max_diff,
                'm_j_ok': mj_ok, 'm_j_diff': mj_max_diff,
                'l_j_ok': lj_ok, 'l_j_diff': lj_max_diff,
                'm_k_ok': mk_ok, 'm_k_diff': mk_max_diff,
                'l_k_ok': lk_ok, 'l_k_diff': lk_max_diff,
            }
            
            if not all([mi_ok, li_ok, mj_ok, lj_ok, mk_ok, lk_ok]):
                all_ok = False
    
    return all_ok, results


def compare_tensors(name, a, b, rtol, atol):
    if a is None or b is None:
        return False, float('nan')
    same = torch.allclose(a, b, rtol=rtol, atol=atol)
    max_abs = (a - b).abs().max().item()
    return same, max_abs


def run_single_test(manual_ext, config, device, rtol, atol, verbose=True):
    """
    Run a single backward equivalence test with the given configuration.
    
    Returns (all_ok, results_dict)
    """
    B = config['B']
    H = config['H']
    I = config['I']
    J = config['J']
    K = config['K']
    D = config['D']
    input_scale = config.get('input_scale', 1.0)  # For stress testing large values
    
    # Create inputs with optional scaling for stress tests
    Q    = torch.randn(B, H, I, D, device=device, dtype=torch.float32) * input_scale
    R    = torch.randn(B, H, J, D, device=device, dtype=torch.float32) * input_scale
    S    = torch.randn(B, H, K, D, device=device, dtype=torch.float32) * input_scale
    Vq_1 = torch.randn(B, H, I, D, device=device, dtype=torch.float32) * input_scale
    Vq_2 = torch.randn(B, H, I, D, device=device, dtype=torch.float32) * input_scale
    Vr_1 = torch.randn(B, H, J, D, device=device, dtype=torch.float32) * input_scale
    Vr_2 = torch.randn(B, H, J, D, device=device, dtype=torch.float32) * input_scale
    Vs_1 = torch.randn(B, H, K, D, device=device, dtype=torch.float32) * input_scale
    Vs_2 = torch.randn(B, H, K, D, device=device, dtype=torch.float32) * input_scale

    max_len = max(I, J, K)
    grad_output = torch.randn(B, H, max_len, D, device=device, dtype=torch.float32)

    results = {'config': config, 'gradients': {}, 'nan_check': True}
    
    # --- Compute manual (CUDA) grads ---
    torch.cuda.synchronize() if device.type == 'cuda' else None
    try:
        grads_manual = compute_manual_grads(
            manual_ext, Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2, grad_output
        )
    except Exception as e:
        if verbose:
            print(f"  ERROR: CUDA backward raised exception: {e}")
        results['error'] = str(e)
        return False, results
    
    # --- Check for NaN/Inf in CUDA outputs (critical for stability tests) ---
    names = [
        'grad_Q', 'grad_R', 'grad_S',
        'grad_Vq_1', 'grad_Vq_2',
        'grad_Vr_1', 'grad_Vr_2',
        'grad_Vs_1', 'grad_Vs_2'
    ]
    
    for name, gm in zip(names, grads_manual):
        if gm is not None and (torch.isnan(gm).any() or torch.isinf(gm).any()):
            results['nan_check'] = False
            if verbose:
                nan_count = torch.isnan(gm).sum().item()
                inf_count = torch.isinf(gm).sum().item()
                print(f"  WARNING: {name} has {nan_count} NaNs, {inf_count} Infs")
    
    # --- Compute reference (autograd) grads ---
    try:
        grads_ref = compute_reference_grads(
            Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2, grad_output
        )
    except Exception as e:
        if verbose:
            print(f"  ERROR: Reference backward raised exception: {e}")
        results['error'] = str(e)
        return False, results

    # --- Compare gradients ---
    all_ok = results['nan_check']
    for name, gm, gr in zip(names, grads_manual, grads_ref):
        ok, max_abs = compare_tensors(name, gm, gr, rtol, atol)
        results['gradients'][name] = {'ok': ok, 'max_abs_diff': max_abs}
        if verbose:
            status = 'PASS' if ok else 'FAIL'
            print(f"  {name:<10} | {status:<4} | max_abs_diff: {max_abs:.3e}")
        all_ok = all_ok and ok
    
    return all_ok, results


def main():
    parser = argparse.ArgumentParser("CUDA backward equivalence vs PyTorch reference")
    parser.add_argument('--B', type=int, default=1)
    parser.add_argument('--H', type=int, default=2)
    parser.add_argument('--N', type=int, default=8, help='Sequence length I=J=K=N (must be multiple of 8)')
    parser.add_argument('--D', type=int, default=32, help='Dimension (must be <= 64)')
    parser.add_argument('--rtol', type=float, default=1e-4)
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--sweep', action='store_true', 
                        help='Run full test sweep across multiple configurations')
    parser.add_argument('--stress', action='store_true',
                        help='Include numerical stability stress tests (large values)')
    parser.add_argument('--quick', action='store_true',
                        help='Run only small/quick test cases')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed per-test output')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    try:
        import hyper_attn_cpp_manual as manual_ext
    except ImportError as e:
        print(f"Error: Could not import 'hyper_attn_cpp_manual': {e}")
        print("Please compile the extension first: python setup.py develop")
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # =========================================================================
    # Test Case Definitions
    # =========================================================================
    # Constraints:
    #   - I = J = K (square only)
    #   - N (I=J=K) must be a multiple of 8
    #   - D <= 64
    
    # Quick tests (small, fast)
    quick_cases = [
        {'name': 'tiny_N8',          'B': 1, 'H': 1, 'I': 8,  'J': 8,  'K': 8,  'D': 16},
        {'name': 'small_N16',        'B': 1, 'H': 2, 'I': 16, 'J': 16, 'K': 16, 'D': 32},
        {'name': 'small_N24',        'B': 1, 'H': 1, 'I': 24, 'J': 24, 'K': 24, 'D': 32},
    ]
    
    # Standard tests (moderate size, covers common cases)
    standard_cases = [
        # Square cases with increasing N
        {'name': 'medium_N32',       'B': 1, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 32},
        {'name': 'medium_N48',       'B': 1, 'H': 2, 'I': 48, 'J': 48, 'K': 48, 'D': 32},
        {'name': 'larger_N64',       'B': 2, 'H': 4, 'I': 64, 'J': 64, 'K': 64, 'D': 32},
        {'name': 'large_N128',       'B': 1, 'H': 2, 'I': 128,'J': 128,'K': 128,'D': 32},
        
        # Multi-batch configurations
        {'name': 'multi_batch_B4',   'B': 4, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 32},
        {'name': 'multi_batch_B8',   'B': 8, 'H': 1, 'I': 32, 'J': 32, 'K': 32, 'D': 32},
        
        # Multi-head configurations
        {'name': 'multi_head_H4',    'B': 1, 'H': 4, 'I': 32, 'J': 32, 'K': 32, 'D': 32},
        {'name': 'multi_head_H8',    'B': 1, 'H': 8, 'I': 32, 'J': 32, 'K': 32, 'D': 32},
        {'name': 'multi_head_H12',   'B': 1, 'H': 12,'I': 32, 'J': 32, 'K': 32, 'D': 32},
        
        # Combined batch + head
        {'name': 'multi_B2_H4',      'B': 2, 'H': 4, 'I': 32, 'J': 32, 'K': 32, 'D': 32},
        {'name': 'multi_B4_H8',      'B': 4, 'H': 8, 'I': 32, 'J': 32, 'K': 32, 'D': 32},
        
        # Different D values (D <= 64)
        {'name': 'small_D8',         'B': 1, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 8},
        {'name': 'small_D16',        'B': 1, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 16},
        {'name': 'medium_D48',       'B': 1, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 48},
        {'name': 'max_D64',          'B': 1, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 64},
    ]
    
    # Stress tests (numerical stability - tests EXP_CLIP and DENOM_EPS)
    stress_cases = [
        # Large input values -> large logits -> tests overflow protection
        {'name': 'stress_scale3',        'B': 1, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 32, 'input_scale': 3.0},
        {'name': 'stress_scale5',        'B': 1, 'H': 1, 'I': 16, 'J': 16, 'K': 16, 'D': 32, 'input_scale': 5.0},
        
        # Larger sequences with stress
        {'name': 'stress_N64_scale2',    'B': 1, 'H': 2, 'I': 64, 'J': 64, 'K': 64, 'D': 32, 'input_scale': 2.0},
        {'name': 'stress_N128_scale2',   'B': 1, 'H': 2, 'I': 128,'J': 128,'K': 128,'D': 32, 'input_scale': 2.0},
        
        # Stress with max D
        {'name': 'stress_D64_scale3',    'B': 1, 'H': 2, 'I': 32, 'J': 32, 'K': 32, 'D': 64, 'input_scale': 3.0},
    ]
    
    # =========================================================================
    # Select test cases based on flags
    # =========================================================================
    
    if args.sweep or args.stress or args.quick:
        # Running multiple test cases
        if args.quick:
            test_cases = quick_cases
        elif args.stress:
            test_cases = quick_cases + standard_cases + stress_cases
        else:  # --sweep
            test_cases = quick_cases + standard_cases
        
        passed = 0
        failed = 0
        failed_cases = []
        
        # Track per-output statistics
        output_names = ['grad_Q', 'grad_R', 'grad_S', 'grad_Vq_1', 'grad_Vq_2', 
                        'grad_Vr_1', 'grad_Vr_2', 'grad_Vs_1', 'grad_Vs_2']
        output_stats = {name: {'pass': 0, 'fail': 0} for name in output_names}
        
        if not args.verbose:
            print(f"Running {len(test_cases)} tests", end="", flush=True)
        
        for i, config in enumerate(test_cases):
            name = config.get('name', f'test_{i}')
            
            if args.verbose:
                scale_info = f" (input_scale={config.get('input_scale', 1.0)})" if config.get('input_scale', 1.0) != 1.0 else ""
                print(f"\n[{i+1}/{len(test_cases)}] {name}: B={config['B']}, H={config['H']}, "
                      f"I=J=K={config['I']}, D={config['D']}{scale_info}")
                print("-" * 60)
            
            torch.manual_seed(args.seed + i)  # Different seed per test for variety
            
            ok, results = run_single_test(manual_ext, config, device, args.rtol, args.atol, verbose=args.verbose)
            
            # Track per-output stats
            for out_name, out_result in results.get('gradients', {}).items():
                if out_result.get('ok', False):
                    output_stats[out_name]['pass'] += 1
                else:
                    output_stats[out_name]['fail'] += 1
            
            if ok:
                passed += 1
                if args.verbose:
                    print(f"  => PASSED")
                else:
                    print(".", end="", flush=True)
            else:
                failed += 1
                failed_cases.append(name)
                if args.verbose:
                    print(f"  => FAILED")
                else:
                    print("x", end="", flush=True)
        
        if not args.verbose:
            print(" done.\n")
        
        # Summary
        print("\n" + "="*80)
        print("BACKWARD PASS SUMMARY")
        print("="*80)
        
        # Overall results
        total = len(test_cases)
        pass_pct = 100 * passed / total if total > 0 else 0
        print(f"\n{'OVERALL RESULTS':^80}")
        print("-" * 80)
        print(f"  Total Tests: {total}")
        print(f"  Passed:      {passed} ({pass_pct:.0f}%)")
        print(f"  Failed:      {failed} ({100-pass_pct:.0f}%)")
        
        # Per-output breakdown table
        print(f"\n{'PER-OUTPUT BREAKDOWN':^80}")
        print("-" * 80)
        print(f"  {'Output':<12} | {'Pass':<6} | {'Fail':<6} | {'Rate':<8} | {'Status'}")
        print(f"  {'-'*12}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*20}")
        
        qrs_pass = 0
        qrs_fail = 0
        v_pass = 0
        v_fail = 0
        
        for out_name in output_names:
            stats = output_stats[out_name]
            total_out = stats['pass'] + stats['fail']
            pct = 100 * stats['pass'] / total_out if total_out > 0 else 0
            if stats['fail'] == 0:
                status = "✓ OK"
            elif stats['pass'] == 0:
                status = "✗ ALL FAIL"
            else:
                status = f"⚠ PARTIAL"
            
            # Track Q/R/S vs V gradients
            if out_name in ['grad_Q', 'grad_R', 'grad_S']:
                qrs_pass += stats['pass']
                qrs_fail += stats['fail']
            else:
                v_pass += stats['pass']
                v_fail += stats['fail']
            
            print(f"  {out_name:<12} | {stats['pass']:<6} | {stats['fail']:<6} | {pct:>5.0f}%   | {status}")
        
        print(f"  {'-'*12}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*20}")
        
        # Q/R/S vs V summary
        qrs_total = qrs_pass + qrs_fail
        v_total = v_pass + v_fail
        qrs_pct = 100 * qrs_pass / qrs_total if qrs_total > 0 else 0
        v_pct = 100 * v_pass / v_total if v_total > 0 else 0
        
        print(f"  {'grad_Q/R/S':<12} | {qrs_pass:<6} | {qrs_fail:<6} | {qrs_pct:>5.0f}%   | {'✓ OK' if qrs_fail == 0 else '⚠ ISSUES'}")
        print(f"  {'grad_V*':<12} | {v_pass:<6} | {v_fail:<6} | {v_pct:>5.0f}%   | {'✓ OK' if v_fail == 0 else '⚠ ISSUES'}")
        
        # Quick diagnostic table
        print(f"\n{'DIAGNOSTIC BREAKDOWN':^80}")
        print("-" * 80)
        print(f"  {'Category':<20} | {'Status':<10} | {'Notes'}")
        print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*40}")
        
        # Analyze patterns in failures
        small_n_fails = [c for c in failed_cases if 'N8' in c or 'N16' in c or 'N24' in c]
        medium_n_fails = [c for c in failed_cases if 'N32' in c or 'N48' in c]
        large_n_fails = [c for c in failed_cases if 'N64' in c or 'N128' in c]
        batch_fails = [c for c in failed_cases if 'batch' in c.lower() or c.startswith('multi_B')]
        head_fails = [c for c in failed_cases if 'head' in c.lower() or '_H' in c]
        stress_fails = [c for c in failed_cases if 'stress' in c.lower()]
        
        def status_icon(count, total_cat):
            if count == 0: return "✓ OK"
            elif count == total_cat: return "✗ ALL FAIL"
            else: return f"⚠ {count} fail"
        
        # Count totals per category from test_cases
        small_n_total = len([c for c in test_cases if 'N8' in c.get('name','') or 'N16' in c.get('name','') or 'N24' in c.get('name','')])
        medium_n_total = len([c for c in test_cases if 'N32' in c.get('name','') or 'N48' in c.get('name','')])
        large_n_total = len([c for c in test_cases if 'N64' in c.get('name','') or 'N128' in c.get('name','')])
        
        print(f"  {'Small N (8-24)':<20} | {status_icon(len(small_n_fails), small_n_total):<10} | {', '.join(small_n_fails) if small_n_fails else 'All passing'}")
        print(f"  {'Medium N (32-48)':<20} | {status_icon(len(medium_n_fails), medium_n_total):<10} | {', '.join(medium_n_fails) if medium_n_fails else 'All passing'}")
        print(f"  {'Large N (64-128)':<20} | {status_icon(len(large_n_fails), large_n_total):<10} | {', '.join(large_n_fails) if large_n_fails else 'All passing'}")
        print(f"  {'Multi-Batch':<20} | {status_icon(len(batch_fails), 4):<10} | {', '.join(batch_fails) if batch_fails else 'All passing'}")
        print(f"  {'Multi-Head':<20} | {status_icon(len(head_fails), 5):<10} | {', '.join(head_fails) if head_fails else 'All passing'}")
        print(f"  {'Stress Tests':<20} | {status_icon(len(stress_fails), 5):<10} | {', '.join(stress_fails) if stress_fails else 'All passing'}")
        
        # Failed cases list
        if failed_cases:
            print(f"\n{'FAILED CASES':^80}")
            print("-" * 80)
            for i, case in enumerate(failed_cases, 1):
                print(f"  {i:2}. {case}")
        
        # Root cause analysis
        print(f"\n{'ROOT CAUSE ANALYSIS':^80}")
        print("-" * 80)
        
        if qrs_fail > 0 and v_fail == 0:
            print("  🔍 grad_Q/R/S failing while grad_V* passes")
            print("     → Focus on: Pass 3 kernels (grad_Q_kernel, grad_R_kernel, grad_S_kernel)")
            print("     → Check: sum_q/sum_r/sum_s accumulation in Pass 1")
        elif v_fail > 0 and qrs_fail == 0:
            print("  🔍 grad_V* failing while grad_Q/R/S passes")
            print("     → Focus on: Pass 2 kernels (grad_gather_V*, grad_scatter_V*)")
        elif qrs_fail > 0 and v_fail > 0:
            print("  🔍 Both grad_Q/R/S and grad_V* failing")
            print("     → Likely: shared issue (softmax stats m/l from forward pass)")
        
        if len(small_n_fails) > 0:
            print("  🔍 Small N tests failing - fundamental logic bug")
            print("     → Debug with N=8 first (single block, no tiling)")
        elif len(medium_n_fails) > 0 and len(small_n_fails) == 0:
            print("  🔍 Medium N fails but Small N passes")
            print("     → Likely: TILING or BLOCK boundary issue")
        
        if len(batch_fails) > 0:
            print("  🔍 Multi-batch tests failing")
            print("     → Check: batch stride calculation (bh indexing)")
        if len(head_fails) > 0:
            print("  🔍 Multi-head tests failing")
            print("     → Check: head indexing in kernel launches")
        if len(stress_fails) > 0 and len([c for c in failed_cases if 'stress' not in c.lower()]) == 0:
            print("  🔍 Only stress tests failing")
            print("     → Numerical stability issue (check EXP_CLIP, DENOM_EPS)")
        
        if passed == total:
            print("  ✓ All tests passing!")
        
        # Failed cases list (compact)
        if failed_cases:
            print(f"\n{'FAILED CASES (' + str(len(failed_cases)) + ' total)':^80}")
            print("-" * 80)
            # Print in columns
            cols = 3
            for i in range(0, len(failed_cases), cols):
                row = failed_cases[i:i+cols]
                print("  " + "  ".join(f"{c:<25}" for c in row))
        
        print("="*80)
        
        sys.exit(0 if failed == 0 else 2)
    
    else:
        # Single test mode (original behavior)
        # Validate constraints: N must be multiple of 8, D <= 64
        if args.N % 8 != 0:
            print(f"Error: N={args.N} must be a multiple of 8")
            sys.exit(1)
        if args.D > 64:
            print(f"Error: D={args.D} must be <= 64")
            sys.exit(1)
        
        I = J = K = args.N
        config = {
            'name': 'single_test',
            'B': args.B, 'H': args.H, 
            'I': I, 'J': J, 'K': K, 
            'D': args.D
        }
        
        print("\nBackward Equivalence Check (CUDA vs Reference)")
        print(f"Config: B={args.B}, H={args.H}, I=J=K={I}, D={args.D}")
        print(f"Tolerances: rtol={args.rtol}, atol={args.atol}")
        print("-" * 80)
        
        ok, results = run_single_test(manual_ext, config, device, args.rtol, args.atol, verbose=True)
        
        print("-" * 80)
        if ok:
            print("All gradients match within tolerance.")
            sys.exit(0)
        else:
            print("Mismatches detected.")
            sys.exit(2)


if __name__ == '__main__':
    main()


