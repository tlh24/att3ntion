import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time

# using this to test manual in a language i understand :3

## FORWARD PASS

def compute_dot_product(Q, R, S, b, h, i, j, k, D):
    """Helper function to compute dot product between three vectors at specific indices"""
    dot = 0.0
    for d in range(D):
        dot += Q[b][h][i][d] * R[b][h][j][d] * S[b][h][k][d]
    return dot

def compute_softmax_3d(Q, R, S, b, h, fixed_dim, fixed_idx, dim1_size, dim2_size, 
                      dim1_idx_fn, dim2_idx_fn, fixed_dim_idx_fn, scale, attn_out):
    """Helper function to compute 3D softmax with configurable dimensions"""
    max_val = float('-inf')
    for idx1 in range(dim1_size):
        for idx2 in range(dim2_size):
            i = fixed_idx if fixed_dim == 0 else (idx1 if dim1_idx_fn == 0 else (idx2 if dim2_idx_fn == 0 else 0))
            j = fixed_idx if fixed_dim == 1 else (idx1 if dim1_idx_fn == 1 else (idx2 if dim2_idx_fn == 1 else 0))
            k = fixed_idx if fixed_dim == 2 else (idx1 if dim1_idx_fn == 2 else (idx2 if dim2_idx_fn == 2 else 0))
            
            dot = compute_dot_product(Q, R, S, b, h, i, j, k, Q.shape[3])
            dot *= scale
            if dot > max_val:
                max_val = dot
    
    sum_exp = 0.0
    for idx1 in range(dim1_size):
        for idx2 in range(dim2_size):
            i = fixed_idx if fixed_dim == 0 else (idx1 if dim1_idx_fn == 0 else (idx2 if dim2_idx_fn == 0 else 0))
            j = fixed_idx if fixed_dim == 1 else (idx1 if dim1_idx_fn == 1 else (idx2 if dim2_idx_fn == 1 else 0))
            k = fixed_idx if fixed_dim == 2 else (idx1 if dim1_idx_fn == 2 else (idx2 if dim2_idx_fn == 2 else 0))
            
            dot = compute_dot_product(Q, R, S, b, h, i, j, k, Q.shape[3])
            dot *= scale
            sum_exp += math.exp(dot - max_val)
    
    idx = 0
    for idx1 in range(dim1_size):
        for idx2 in range(dim2_size):
            i = fixed_idx if fixed_dim == 0 else (idx1 if dim1_idx_fn == 0 else (idx2 if dim2_idx_fn == 0 else 0))
            j = fixed_idx if fixed_dim == 1 else (idx1 if dim1_idx_fn == 1 else (idx2 if dim2_idx_fn == 1 else 0))
            k = fixed_idx if fixed_dim == 2 else (idx1 if dim1_idx_fn == 2 else (idx2 if dim2_idx_fn == 2 else 0))
            
            dot = compute_dot_product(Q, R, S, b, h, i, j, k, Q.shape[3])
            dot *= scale
            attn_out[idx] = math.exp(dot - max_val) / sum_exp
            idx += 1

def compute_Y_gather(Y_out, Q, R, S, V1, V2, fixed_dim):
    """Compute the gathering of attention weights and values for a fixed dimension"""
    B, H, I = Q.shape[0], Q.shape[1], Q.shape[2]
    J, K, D = R.shape[2], S.shape[2], Q.shape[3]
    scale = 1.0 / math.sqrt(D)
    sizes = [I, J, K]
    
    for b in range(B):
        for h in range(H):
            for fixed_idx in range(sizes[fixed_dim]):
                if fixed_dim == 0:
                    dim1, dim2 = 1, 2
                    dim1_size, dim2_size = J, K
                elif fixed_dim == 1:
                    dim1, dim2 = 0, 2
                    dim1_size, dim2_size = I, K
                else:
                    dim1, dim2 = 0, 1
                    dim1_size, dim2_size = I, J
                
                for d in range(D):
                    Y_out[b][h][fixed_idx][d] = 0.0
                    max_val = float('-inf')
                    
                    for idx1 in range(dim1_size):
                        for idx2 in range(dim2_size):
                            i = fixed_idx if fixed_dim == 0 else (idx1 if dim1 == 0 else idx2)
                            j = fixed_idx if fixed_dim == 1 else (idx1 if dim1 == 1 else idx2)
                            k = fixed_idx if fixed_dim == 2 else (idx1 if dim1 == 2 else idx2)
                            
                            dot_prod = 0.0
                            for d_dot in range(D):
                                dot_prod += Q[b][h][i][d_dot] * R[b][h][j][d_dot] * S[b][h][k][d_dot]
                            dot_prod *= scale
                            max_val = max(max_val, dot_prod)
                    
                    sum_exp = 0.0
                    for idx1 in range(dim1_size):
                        for idx2 in range(dim2_size):
                            i = fixed_idx if fixed_dim == 0 else (idx1 if dim1 == 0 else idx2)
                            j = fixed_idx if fixed_dim == 1 else (idx1 if dim1 == 1 else idx2)
                            k = fixed_idx if fixed_dim == 2 else (idx1 if dim1 == 2 else idx2)
                            
                            dot_prod = 0.0
                            for d_dot in range(D):
                                dot_prod += Q[b][h][i][d_dot] * R[b][h][j][d_dot] * S[b][h][k][d_dot]
                            dot_prod *= scale
                            sum_exp += math.exp(dot_prod - max_val)
                    
                    for idx1 in range(dim1_size):
                        for idx2 in range(dim2_size):
                            i = fixed_idx if fixed_dim == 0 else (idx1 if dim1 == 0 else idx2)
                            j = fixed_idx if fixed_dim == 1 else (idx1 if dim1 == 1 else idx2)
                            k = fixed_idx if fixed_dim == 2 else (idx1 if dim1 == 2 else idx2)
                            
                            v1_idx = i if dim1 == 0 else (j if dim1 == 1 else k)
                            v2_idx = i if dim2 == 0 else (j if dim2 == 1 else k)
                            
                            dot_prod = 0.0
                            for d_dot in range(D):
                                dot_prod += Q[b][h][i][d_dot] * R[b][h][j][d_dot] * S[b][h][k][d_dot]
                            dot_prod *= scale
                            
                            attn = math.exp(dot_prod - max_val) / sum_exp
                            Y_out[b][h][fixed_idx][d] += attn * V1[b][h][v1_idx][d] * V2[b][h][v2_idx][d]

def compute_Y_scatter_q(Y_q, Q, R, S, Vr_2, Vs_2):
    """Compute the scatter attention weights and values for Q dimension"""
    B, H, I = Q.shape[0], Q.shape[1], Q.shape[2]
    J, K, D = R.shape[2], S.shape[2], Q.shape[3]
    scale = 1.0 / math.sqrt(D)
    
    for b in range(B):
        for h in range(H):
            for i in range(I):
                Ar_values = [[0.0 for _ in range(K)] for _ in range(J)]
                for j in range(J):
                    softmax_results = [0.0] * (I * K)
                    compute_softmax_3d(
                        Q, R, S,
                        b, h,
                        1, j,
                        I, K,
                        0, 2, 1,
                        scale, softmax_results
                    )
                    
                    for k in range(K):
                        idx = i * K + k
                        Ar_values[j][k] = softmax_results[idx]
                
                As_values = [[0.0 for _ in range(J)] for _ in range(K)]
                for k in range(K):
                    softmax_results = [0.0] * (I * J)
                    compute_softmax_3d(
                        Q, R, S,
                        b, h,
                        2, k,
                        I, J,
                        0, 1, 2,
                        scale, softmax_results
                    )
                    
                    for j in range(J):
                        idx = i * J + j
                        As_values[k][j] = softmax_results[idx]
                
                for d in range(D):
                    sum_val = 0.0
                    for j in range(J):
                        for k in range(K):
                            attn = Ar_values[j][k] * As_values[k][j]
                            sum_val += attn * Vr_2[b][h][j][d] * Vs_2[b][h][k][d]
                    Y_q[b][h][i][d] += sum_val

def compute_Y_scatter_r(Y_r_, Q, R, S, Vq_2, Vs_2):
    """Compute the scatter attention weights and values for R dimension"""
    B, H, I = Q.shape[0], Q.shape[1], Q.shape[2]
    J, K, D = R.shape[2], S.shape[2], Q.shape[3]
    scale = 1.0 / math.sqrt(D)
    
    for b in range(B):
        for h in range(H):
            for j in range(J):
                Aq_values = [[0.0 for _ in range(K)] for _ in range(I)]
                for i in range(I):
                    softmax_results = [0.0] * (J * K)
                    compute_softmax_3d(
                        Q, R, S,
                        b, h,
                        0, i,
                        J, K,
                        1, 2, 0,
                        scale, softmax_results
                    )
                    
                    for k in range(K):
                        idx = j * K + k
                        Aq_values[i][k] = softmax_results[idx]
                
                As_values = [[0.0 for _ in range(I)] for _ in range(K)]
                for k in range(K):
                    softmax_results = [0.0] * (I * J)
                    compute_softmax_3d(
                        Q, R, S,
                        b, h,
                        2, k,
                        I, J,
                        0, 1, 2,
                        scale, softmax_results
                    )
                    
                    for i in range(I):
                        idx = i * J + j
                        As_values[k][i] = softmax_results[idx]
                
                for d in range(D):
                    sum_val = 0.0
                    for i in range(I):
                        for k in range(K):
                            attn = Aq_values[i][k] * As_values[k][i]
                            sum_val += attn * Vq_2[b][h][i][d] * Vs_2[b][h][k][d]
                    Y_r_[b][h][j][d] += sum_val

def compute_Y_scatter_s(Y_s_, Q, R, S, Vq_2, Vr_2):
    """Compute the scatter attention weights and values for S dimension"""
    B, H, I = Q.shape[0], Q.shape[1], Q.shape[2]
    J, K, D = R.shape[2], S.shape[2], Q.shape[3]
    scale = 1.0 / math.sqrt(D)
    
    for b in range(B):
        for h in range(H):
            for k in range(K):
                Aq_values = [[0.0 for _ in range(J)] for _ in range(I)]
                for i in range(I):
                    softmax_results = [0.0] * (J * K)
                    compute_softmax_3d(
                        Q, R, S,
                        b, h,
                        0, i,
                        J, K,
                        1, 2, 0,
                        scale, softmax_results
                    )
                    
                    for j in range(J):
                        idx = j * K + k
                        Aq_values[i][j] = softmax_results[idx]
                
                Ar_values = [[0.0 for _ in range(I)] for _ in range(J)]
                for j in range(J):
                    softmax_results = [0.0] * (I * K)
                    compute_softmax_3d(
                        Q, R, S,
                        b, h,
                        1, j,
                        I, K,
                        0, 2, 1,
                        scale, softmax_results
                    )
                    
                    for i in range(I):
                        idx = i * K + k
                        Ar_values[j][i] = softmax_results[idx]
                
                for d in range(D):
                    sum_val = 0.0
                    for i in range(I):
                        for j in range(J):
                            attn = Aq_values[i][j] * Ar_values[j][i]
                            sum_val += attn * Vq_2[b][h][i][d] * Vr_2[b][h][j][d]
                    Y_s_[b][h][k][d] += sum_val

def forward_pass(Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2, dropout_rate=0.0):
    """Forward pass of the hypergraph attention mechanism"""
    B, H, I, D = Q.shape
    J, K = R.shape[2], S.shape[2]
    
    # Initialize output tensors
    Y_q = torch.zeros((B, H, I, D), device=Q.device, dtype=Q.dtype)
    Y_r = torch.zeros((B, H, J, D), device=Q.device, dtype=Q.dtype)
    Y_s = torch.zeros((B, H, K, D), device=Q.device, dtype=Q.dtype)
    Y_q_ = torch.zeros((B, H, I, D), device=Q.device, dtype=Q.dtype)
    Y_r_ = torch.zeros((B, H, J, D), device=Q.device, dtype=Q.dtype)
    Y_s_ = torch.zeros((B, H, K, D), device=Q.device, dtype=Q.dtype)
    
    # Convert tensors to Python-readable format
    Q_np = Q.detach().cpu().numpy()
    R_np = R.detach().cpu().numpy()
    S_np = S.detach().cpu().numpy()
    Vq_1_np = Vq_1.detach().cpu().numpy()
    Vq_2_np = Vq_2.detach().cpu().numpy()
    Vr_1_np = Vr_1.detach().cpu().numpy()
    Vr_2_np = Vr_2.detach().cpu().numpy()
    Vs_1_np = Vs_1.detach().cpu().numpy()
    Vs_2_np = Vs_2.detach().cpu().numpy()
    Y_q_np = Y_q.detach().cpu().numpy()
    Y_r_np = Y_r.detach().cpu().numpy()
    Y_s_np = Y_s.detach().cpu().numpy()
    Y_q__np = Y_q_.detach().cpu().numpy()
    Y_r__np = Y_r_.detach().cpu().numpy() 
    Y_s__np = Y_s_.detach().cpu().numpy()
    
    # Compute the gather and scatter operations
    compute_Y_gather(Y_q_np, Q_np, R_np, S_np, Vr_1_np, Vs_1_np, 0)
    compute_Y_gather(Y_r_np, Q_np, R_np, S_np, Vq_1_np, Vs_1_np, 1)
    compute_Y_gather(Y_s_np, Q_np, R_np, S_np, Vq_1_np, Vr_1_np, 2)
    
    compute_Y_scatter_q(Y_q__np, Q_np, R_np, S_np, Vr_2_np, Vs_2_np)
    compute_Y_scatter_r(Y_r__np, Q_np, R_np, S_np, Vq_2_np, Vs_2_np)
    compute_Y_scatter_s(Y_s__np, Q_np, R_np, S_np, Vq_2_np, Vr_2_np)
    
    # Convert results back to PyTorch tensors
    Y_q = torch.tensor(Y_q_np, device=Q.device, dtype=Q.dtype)
    Y_r = torch.tensor(Y_r_np, device=Q.device, dtype=Q.dtype)
    Y_s = torch.tensor(Y_s_np, device=Q.device, dtype=Q.dtype)
    Y_q_ = torch.tensor(Y_q__np, device=Q.device, dtype=Q.dtype)
    Y_r_ = torch.tensor(Y_r__np, device=Q.device, dtype=Q.dtype)
    Y_s_ = torch.tensor(Y_s__np, device=Q.device, dtype=Q.dtype)
    
    # Return the sum of all outputs

    max_seq_len = max(I, J, K)
    Y = torch.zeros((B, H, max_seq_len, D), device=Q.device, dtype=Q.dtype)
    Y[:, :, :I, :] += Y_q
    Y[:, :, :J, :] += Y_r
    Y[:, :, :K, :] += Y_s
    Y[:, :, :I, :] += Y_q_
    Y[:, :, :J, :] += Y_r_
    Y[:, :, :K, :] += Y_s_
    return Y


## BACKWARD PASS

def compute_grad_Vq_1(grad_Vq_1, grad_output, Q, R, S, Vr_1, Vs_1, dropout_rate=0.0):
    """Compute gradients for Vq_1"""
    B, H, I = Q.shape[0], Q.shape[1], Q.shape[2]
    J, K, D = R.shape[2], S.shape[2], Q.shape[3]
    scale = 1.0 / math.sqrt(D)
    
    grad_output_np = grad_output.detach().cpu().numpy()
    Q_np = Q.detach().cpu().numpy()
    R_np = R.detach().cpu().numpy()
    S_np = S.detach().cpu().numpy()
    Vr_1_np = Vr_1.detach().cpu().numpy()
    Vs_1_np = Vs_1.detach().cpu().numpy()
    grad_Vq_1_np = grad_Vq_1.detach().cpu().numpy()
    
    for b in range(B):
        for h in range(H):
            for j in range(J):
                Ar_j_values = [0.0] * (I * K)
                
                compute_softmax_3d(
                    Q_np, R_np, S_np,
                    b, h,
                    1, j,
                    I, K,
                    0, 2, 1,
                    scale, Ar_j_values
                )
                
                for d in range(D):
                    dy_r = grad_output_np[b][h][j][d]
                    
                    for i in range(I):
                        for k in range(K):
                            attn_idx = i * K + k
                            contribution = dy_r * Ar_j_values[attn_idx] * Vs_1_np[b][h][k][d]
                            grad_Vq_1_np[b][h][i][d] += contribution
            
            for k in range(K):
                As_k_values = [0.0] * (I * J)
                
                compute_softmax_3d(
                    Q_np, R_np, S_np,
                    b, h,
                    2, k,
                    I, J,
                    0, 1, 2,
                    scale, As_k_values
                )
                
                for d in range(D):
                    dy_s = grad_output_np[b][h][k][d]
                    
                    for i in range(I):
                        for j in range(J):
                            attn_idx = i * J + j
                            contribution = dy_s * As_k_values[attn_idx] * Vr_1_np[b][h][j][d]
                            grad_Vq_1_np[b][h][i][d] += contribution
    
    return torch.tensor(grad_Vq_1_np, device=grad_Vq_1.device, dtype=grad_Vq_1.dtype)

def compute_grad_Vr_1(grad_Vr_1, grad_output, Q, R, S, Vq_1, Vs_1, dropout_rate=0.0):
    """Compute gradients for Vr_1"""
    B, H, I = Q.shape[0], Q.shape[1], Q.shape[2]
    J, K, D = R.shape[2], S.shape[2], Q.shape[3]
    scale = 1.0 / math.sqrt(D)
    
    grad_output_np = grad_output.detach().cpu().numpy()
    Q_np = Q.detach().cpu().numpy()
    R_np = R.detach().cpu().numpy()
    S_np = S.detach().cpu().numpy()
    Vq_1_np = Vq_1.detach().cpu().numpy()
    Vs_1_np = Vs_1.detach().cpu().numpy()
    grad_Vr_1_np = grad_Vr_1.detach().cpu().numpy()
    
    for b in range(B):
        for h in range(H):
            for i in range(I):
                Aq_i_values = [0.0] * (J * K)
                
                compute_softmax_3d(
                    Q_np, R_np, S_np,
                    b, h,
                    0, i,
                    J, K,
                    1, 2, 0,
                    scale, Aq_i_values
                )
                
                for d in range(D):
                    dy_q = grad_output_np[b][h][i][d]
                    
                    for j in range(J):
                        for k in range(K):
                            attn_idx = j * K + k
                            contribution = dy_q * Aq_i_values[attn_idx] * Vs_1_np[b][h][k][d]
                            grad_Vr_1_np[b][h][j][d] += contribution
            
            for k in range(K):
                As_k_values = [0.0] * (I * J)
                
                compute_softmax_3d(
                    Q_np, R_np, S_np,
                    b, h,
                    2, k,
                    I, J,
                    0, 1, 2,
                    scale, As_k_values
                )
                
                for d in range(D):
                    dy_s = grad_output_np[b][h][k][d]
                    
                    for i in range(I):
                        for j in range(J):
                            attn_idx = i * J + j
                            contribution = dy_s * As_k_values[attn_idx] * Vq_1_np[b][h][i][d]
                            grad_Vr_1_np[b][h][j][d] += contribution
    
    return torch.tensor(grad_Vr_1_np, device=grad_Vr_1.device, dtype=grad_Vr_1.dtype)

def compute_grad_Vs_1(grad_Vs_1, grad_output, Q, R, S, Vq_1, Vr_1, dropout_rate=0.0):
    """Compute gradients for Vs_1"""
    B, H, I = Q.shape[0], Q.shape[1], Q.shape[2]
    J, K, D = R.shape[2], S.shape[2], Q.shape[3]
    scale = 1.0 / math.sqrt(D)
    
    grad_output_np = grad_output.detach().cpu().numpy()
    Q_np = Q.detach().cpu().numpy()
    R_np = R.detach().cpu().numpy()
    S_np = S.detach().cpu().numpy()
    Vq_1_np = Vq_1.detach().cpu().numpy()
    Vr_1_np = Vr_1.detach().cpu().numpy()
    grad_Vs_1_np = grad_Vs_1.detach().cpu().numpy()
    
    for b in range(B):
        for h in range(H):
            for i in range(I):
                Aq_i_values = [0.0] * (J * K)
                
                compute_softmax_3d(
                    Q_np, R_np, S_np,
                    b, h,
                    0, i,
                    J, K,
                    1, 2, 0,
                    scale, Aq_i_values
                )
                
                for d in range(D):
                    dy_q = grad_output_np[b][h][i][d]
                    
                    for j in range(J):
                        for k in range(K):
                            attn_idx = j * K + k
                            attn = Aq_i_values[attn_idx]
                            grad_Vs_1_np[b][h][k][d] += dy_q * attn * Vr_1_np[b][h][j][d]
            
            for j in range(J):
                Ar_j_values = [0.0] * (I * K)
                
                compute_softmax_3d(
                    Q_np, R_np, S_np,
                    b, h,
                    1, j,
                    I, K,
                    0, 2, 1,
                    scale, Ar_j_values
                )
                
                for d in range(D):
                    dy_r = grad_output_np[b][h][j][d]
                    
                    for i in range(I):
                        for k in range(K):
                            attn_idx = i * K + k
                            attn = Ar_j_values[attn_idx]
                            grad_Vs_1_np[b][h][k][d] += dy_r * attn * Vq_1_np[b][h][i][d]
    
    return torch.tensor(grad_Vs_1_np, device=grad_Vs_1.device, dtype=grad_Vs_1.dtype)

def backward_pass(grad_output, Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2, dropout_rate=0.0):
    """Backward pass for the hypergraph attention mechanism"""
    
    B, H, I, D = Q.shape
    J, K = R.shape[2], S.shape[2]
    scale = 1.0 / math.sqrt(D)

    Q_fwd = Q.detach()
    R_fwd = R.detach()
    S_fwd = S.detach()

    grad_Q = torch.zeros_like(Q)
    grad_R = torch.zeros_like(R)
    grad_S = torch.zeros_like(S)
    grad_Vq_1 = torch.zeros_like(Vq_1)
    grad_Vq_2 = torch.zeros_like(Vq_2)
    grad_Vr_1 = torch.zeros_like(Vr_1)
    grad_Vr_2 = torch.zeros_like(Vr_2)
    grad_Vs_1 = torch.zeros_like(Vs_1)
    grad_Vs_2 = torch.zeros_like(Vs_2)
    
    grad_Vq_1 = compute_grad_Vq_1(grad_Vq_1, grad_output, Q, R, S, Vr_1, Vs_1, dropout_rate)
    grad_Vr_1 = compute_grad_Vr_1(grad_Vr_1, grad_output, Q, R, S, Vq_1, Vs_1, dropout_rate)
    grad_Vs_1 = compute_grad_Vs_1(grad_Vs_1, grad_output, Q, R, S, Vq_1, Vr_1, dropout_rate)
    
    return (grad_Q, grad_R, grad_S, 
            grad_Vq_1, grad_Vq_2,
            grad_Vr_1, grad_Vr_2,
            grad_Vs_1, grad_Vs_2)


def test_manual_backward_vs_autograd():
    """
    Tests against autograd. 
    """
    print("=== Testing Manual Backward Pass against PyTorch Autograd ===")
    
    configs = [
        # (batch_size, n_heads, seq_len, hidden_dim)
        (2, 1, 3, 4),  
    ]
    
    all_passed = True
    results = []
    
    for config in configs:
        batch_size, n_heads, seq_len, hidden_dim = config
        print(f"\nTesting config: batch_size={batch_size}, n_heads={n_heads}, seq_len={seq_len}, hidden_dim={hidden_dim}")
        
        Q = torch.randn(batch_size, n_heads, seq_len, hidden_dim, requires_grad=True, dtype=torch.double)
        R = torch.randn(batch_size, n_heads, seq_len, hidden_dim, requires_grad=True, dtype=torch.double)
        S = torch.randn(batch_size, n_heads, seq_len, hidden_dim, requires_grad=True, dtype=torch.double)
        Vq_1 = torch.randn(batch_size, n_heads, seq_len, hidden_dim, requires_grad=True, dtype=torch.double)
        Vq_2 = torch.randn(batch_size, n_heads, seq_len, hidden_dim, requires_grad=True, dtype=torch.double)
        Vr_1 = torch.randn(batch_size, n_heads, seq_len, hidden_dim, requires_grad=True, dtype=torch.double)
        Vr_2 = torch.randn(batch_size, n_heads, seq_len, hidden_dim, requires_grad=True, dtype=torch.double)
        Vs_1 = torch.randn(batch_size, n_heads, seq_len, hidden_dim, requires_grad=True, dtype=torch.double)
        Vs_2 = torch.randn(batch_size, n_heads, seq_len, hidden_dim, requires_grad=True, dtype=torch.double)
        
        Q_auto = Q.clone().detach().requires_grad_(True)
        R_auto = R.clone().detach().requires_grad_(True)
        S_auto = S.clone().detach().requires_grad_(True)
        Vq_1_auto = Vq_1.clone().detach().requires_grad_(True)
        Vq_2_auto = Vq_2.clone().detach().requires_grad_(True)
        Vr_1_auto = Vr_1.clone().detach().requires_grad_(True)
        Vr_2_auto = Vr_2.clone().detach().requires_grad_(True)
        Vs_1_auto = Vs_1.clone().detach().requires_grad_(True)
        Vs_2_auto = Vs_2.clone().detach().requires_grad_(True)
        
        max_seq_len = max(seq_len, seq_len, seq_len)
        grad_output = torch.randn(batch_size, n_heads, max_seq_len, hidden_dim, dtype=torch.double)
        
        # autograd path
        def forward_pass_pytorch_reference(Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2):
            B, H, I, D = Q.shape
            J, K = R.shape[2], S.shape[2] # Assuming R, S have shape [B, H, seq_len, D]
            scale = 1.0 / math.sqrt(D)

            dot_product = torch.einsum('bhid,bhjd,bhkd->bhijk', Q, R, S) * scale

            # --- Compute Attention Distributions ---
            attention_flat_q = dot_product.reshape(B, H, I, J*K)
            Aq = F.softmax(attention_flat_q, dim=-1).reshape(B, H, I, J, K)

            attention_perm_r = dot_product.permute(0, 1, 3, 2, 4) # [b, h, j, i, k]
            Ar_flat = attention_perm_r.reshape(B, H, J, I*K)
            Ar = F.softmax(Ar_flat, dim=-1).reshape(B, H, J, I, K)
            Ar = Ar.permute(0, 1, 3, 2, 4) # Permute back to [b, h, i, j, k]

            attention_perm_s = dot_product.permute(0, 1, 4, 2, 3) # [b, h, k, i, j]
            As_flat = attention_perm_s.reshape(B, H, K, I*J)
            As = F.softmax(As_flat, dim=-1).reshape(B, H, K, I, J)
            As = As.permute(0, 1, 3, 4, 2) # Permute back to [b, h, i, j, k]

            # --- Gather Operations ---
            Y_q = torch.einsum('bhijk,bhjd,bhkd->bhid', Aq, Vr_1, Vs_1)
            Y_r = torch.einsum('bhijk,bhid,bhkd->bhjd', Ar, Vq_1, Vs_1)
            Y_s = torch.einsum('bhijk,bhid,bhjd->bhkd', As, Vq_1, Vr_1)

            # --- Scatter Operations ( Corrected) ---
            ArAs = Ar * As
            Y_q_ = torch.einsum("bhijk,bhjd,bhkd->bhid", ArAs, Vr_2, Vs_2)

            AqAs = Aq * As
            Y_r_ = torch.einsum("bhijk,bhid,bhkd->bhjd", AqAs, Vq_2, Vs_2)

            AqAr = Aq * Ar
            Y_s_ = torch.einsum("bhijk,bhid,bhjd->bhkd", AqAr, Vq_2, Vr_2)

            max_seq_len = max(I, J, K)
            Y = torch.zeros((B, H, max_seq_len, D), device=Q.device, dtype=Q.dtype)
            Y[:, :, :I, :] += Y_q
            Y[:, :, :J, :] += Y_r
            Y[:, :, :K, :] += Y_s
            Y[:, :, :I, :] += Y_q_
            Y[:, :, :J, :] += Y_r_
            Y[:, :, :K, :] += Y_s_

            return Y
        
        start_time = time.time()
        
        # Run autograd path
        output_auto = forward_pass_pytorch_reference(
            Q_auto, R_auto, S_auto, 
            Vq_1_auto, Vq_2_auto, 
            Vr_1_auto, Vr_2_auto, 
            Vs_1_auto, Vs_2_auto
        )
        
        output_auto.backward(grad_output) #backprop
        
        # Collect autograd gradients
        dQ_auto = Q_auto.grad.clone()
        dR_auto = R_auto.grad.clone()
        dS_auto = S_auto.grad.clone()
        dVq_1_auto = Vq_1_auto.grad.clone()
        dVq_2_auto = Vq_2_auto.grad.clone()
        dVr_1_auto = Vr_1_auto.grad.clone()
        dVr_2_auto = Vr_2_auto.grad.clone()
        dVs_1_auto = Vs_1_auto.grad.clone()
        dVs_2_auto = Vs_2_auto.grad.clone()
        
        # ========= MANUAL PATH =========
        output_manual = forward_pass(
            Q, R, S, 
            Vq_1, Vq_2, 
            Vr_1, Vr_2, 
            Vs_1, Vs_2
        )
        
        dQ_manual, dR_manual, dS_manual, dVq_1_manual, dVq_2_manual, dVr_1_manual, dVr_2_manual, dVs_1_manual, dVs_2_manual = backward_pass(
            grad_output, Q, R, S, 
            Vq_1, Vq_2, 
            Vr_1, Vr_2, 
            Vs_1, Vs_2
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        rtol = 1e-5
        atol = 1e-5
        
        def compare_gradients(manual_grad, auto_grad, name):
            """Compare the manually computed gradients with autograd gradients."""
            if torch.allclose(manual_grad, auto_grad, rtol=rtol, atol=atol):
                print(f"✓ {name} gradients match")
                return True
            else:
                max_diff = torch.max(torch.abs(manual_grad - auto_grad)).item()
                mean_diff = torch.mean(torch.abs(manual_grad - auto_grad)).item()
                print(f"✗ {name} gradients DO NOT match! Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
                return False
        
        grad_matches = {}
        grad_matches['Q'] = compare_gradients(dQ_manual, dQ_auto, "Q")
        grad_matches['R'] = compare_gradients(dR_manual, dR_auto, "R")
        grad_matches['S'] = compare_gradients(dS_manual, dS_auto, "S")
        grad_matches['Vq_1'] = compare_gradients(dVq_1_manual, dVq_1_auto, "Vq_1")
        grad_matches['Vq_2'] = compare_gradients(dVq_2_manual, dVq_2_auto, "Vq_2")
        grad_matches['Vr_1'] = compare_gradients(dVr_1_manual, dVr_1_auto, "Vr_1")
        grad_matches['Vr_2'] = compare_gradients(dVr_2_manual, dVr_2_auto, "Vr_2")
        grad_matches['Vs_1'] = compare_gradients(dVs_1_manual, dVs_1_auto, "Vs_1")
        grad_matches['Vs_2'] = compare_gradients(dVs_2_manual, dVs_2_auto, "Vs_2")
        
        config_passed = all(grad_matches.values())
        all_passed = all_passed and config_passed
        
        results.append({
            'config': config,
            'passed': config_passed,
            'time': elapsed_time,
            'grad_matches': grad_matches
        })
        
        print(f"Time taken: {elapsed_time:.3f} seconds")
        print(f"Configuration {'PASSED' if config_passed else 'FAILED'}\n")
    
    print("\n===== OVERALL TEST RESULTS =====")
    print(f"Overall test status: {'PASSED' if all_passed else 'FAILED'}")
    for i, result in enumerate(results):
        config = result['config']
        passed = result['passed']
        time_taken = result['time']
        print(f"Config {i+1}: {config} - {'PASSED' if passed else 'FAILED'} ({time_taken:.3f}s)")
        if not passed:
            print("  Failed gradients:")
            for name, match in result['grad_matches'].items():
                if not match:
                    print(f"  - {name}")
    
    return all_passed


if __name__ == "__main__":
    print("Starting gradient validation tests...")
    all_passed = test_manual_backward_vs_autograd()
    
    print("\nGradient validation tests completed.")