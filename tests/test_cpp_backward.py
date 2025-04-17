import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

def forward_pass(Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2):
    """
    Reference forward pass using PyTorch for autograd comparison.
    Matches the scatter logic from hyper_attn_cpp_reference.cpp.
    """
    batch_size, n_heads = Q.shape[0], Q.shape[1]
    seq_len, d_model = Q.shape[2], Q.shape[3]
    I, J, K = seq_len, seq_len, seq_len 

    scale = 1.0 / math.sqrt(d_model)
    dot_product = torch.einsum("bhid,bhjd,bhkd->bhijk", Q, R, S) * scale

    attention_flat_q = dot_product.reshape(batch_size, n_heads, I, J*K)
    Aq = F.softmax(attention_flat_q, dim=-1).reshape(batch_size, n_heads, I, J, K)

    attention_perm_r = dot_product.permute(0, 1, 3, 2, 4) 
    Ar_flat = attention_perm_r.reshape(batch_size, n_heads, J, I*K)
    Ar = F.softmax(Ar_flat, dim=-1).reshape(batch_size, n_heads, J, I, K)
    Ar = Ar.permute(0, 1, 3, 2, 4) 

    attention_perm_s = dot_product.permute(0, 1, 4, 2, 3) # [b, h, k, i, j]
    As_flat = attention_perm_s.reshape(batch_size, n_heads, K, I*J)
    As = F.softmax(As_flat, dim=-1).reshape(batch_size, n_heads, K, I, J)
    As = As.permute(0, 1, 3, 4, 2) # Permute back to [b, h, i, j, k]

    # gather operations
    Y_q = torch.einsum("bhijk,bhjd,bhkd->bhid", Aq, Vr_1, Vs_1)
    Y_r = torch.einsum("bhijk,bhid,bhkd->bhjd", Ar, Vq_1, Vs_1)
    Y_s = torch.einsum("bhijk,bhid,bhjd->bhkd", As, Vq_1, Vr_1)

    # scatter operations  (fixed) 
    ArAs = Ar * As
    Y_q_ = torch.einsum("bhijk,bhjd,bhkd->bhid", ArAs, Vr_2, Vs_2)
    AqAs = Aq * As
    Y_r_ = torch.einsum("bhijk,bhid,bhkd->bhjd", AqAs, Vq_2, Vs_2)
    AqAr = Aq * Ar
    Y_s_ = torch.einsum("bhijk,bhid,bhjd->bhkd", AqAr, Vq_2, Vr_2)

    # --- Sum all components ---
    # Pad outputs if sequence lengths I, J, K differ, assuming max_seq_len = seq_len here
    Y = torch.zeros((batch_size, n_heads, seq_len, d_model), device=Q.device, dtype=Q.dtype)
    Y[:, :, :I, :] += Y_q
    Y[:, :, :J, :] += Y_r
    Y[:, :, :K, :] += Y_s
    Y[:, :, :I, :] += Y_q_
    Y[:, :, :J, :] += Y_r_
    Y[:, :, :K, :] += Y_s_

    return Y

def test_grad_tensors(name, grad_cpp, grad_auto, print_samples=True):
    """Utility function to compare gradients and report results."""
    test_passed = torch.allclose(grad_auto, grad_cpp, rtol=1e-4, atol=1e-4)
    print(f"{name} gradient matches PyTorch autograd: {'✓' if test_passed else '✗'}")
    
    if not test_passed:
        max_diff = torch.max(torch.abs(grad_auto - grad_cpp)).item()
        print(f"  Max difference: {max_diff:.8f}")
        
        if torch.all(grad_cpp == 0):
            print("  WARNING: C++ implementation is returning all zeros!")
        
        if torch.norm(grad_cpp) > 1e-10 and torch.norm(grad_auto) > 1e-10:
            cosine_sim = torch.nn.functional.cosine_similarity(
                grad_cpp.reshape(-1), grad_auto.reshape(-1), dim=0
            ).item()
            print(f"  Cosine similarity: {cosine_sim:.6f}")
        else:
            print("  Cosine similarity: N/A (one or both tensors are effectively zero)")
    
    if print_samples:
        print(f"\nSample values from C++ {name}:")
        sample_idx = [(0, 0, 0, 0), (1, 1, 2, 3), (0, 1, 3, 5)]
        for idx in sample_idx:
            b, h, i, d = idx
            print(f"  {idx}: {grad_cpp[b, h, i, d].item():.6f}")
        
        print(f"\nSample values from PyTorch autograd {name}:")
        for idx in sample_idx:
            b, h, i, d = idx
            print(f"  {idx}: {grad_auto[b, h, i, d].item():.6f}")
    
    return test_passed

def test_all_gradients():
    """
    Test all gradient computations in the backward pass.
    Compare the cpp implementation against PyTorch autograd.
    """
    print("\n=== Testing All Gradient Computations ===")
    torch.manual_seed(42)
    
    batch_size = 2
    seq_len = 5
    d_model = 16
    n_heads = 2
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dtype = torch.float #higher precision
    
    Q = torch.randn(batch_size, n_heads, seq_len, d_model, requires_grad=True, device=device, dtype=dtype)
    R = torch.randn(batch_size, n_heads, seq_len, d_model, requires_grad=True, device=device, dtype=dtype)
    S = torch.randn(batch_size, n_heads, seq_len, d_model, requires_grad=True, device=device, dtype=dtype)
    
    Vq_1 = torch.randn(batch_size, n_heads, seq_len, d_model, requires_grad=True, device=device, dtype=dtype)
    Vq_2 = torch.randn(batch_size, n_heads, seq_len, d_model, requires_grad=True, device=device, dtype=dtype)
    Vr_1 = torch.randn(batch_size, n_heads, seq_len, d_model, requires_grad=True, device=device, dtype=dtype)
    Vr_2 = torch.randn(batch_size, n_heads, seq_len, d_model, requires_grad=True, device=device, dtype=dtype)
    Vs_1 = torch.randn(batch_size, n_heads, seq_len, d_model, requires_grad=True, device=device, dtype=dtype)
    Vs_2 = torch.randn(batch_size, n_heads, seq_len, d_model, requires_grad=True, device=device, dtype=dtype)
    
    grad_output = torch.randn(batch_size, n_heads, seq_len, d_model, device=device, dtype=dtype)
    
    try:
        import hyper_attn_cpp_manual
        print("Successfully imported hyper_attn_cpp_manual module")
    except ImportError:
        print("Failed to import hyper_attn_cpp_manual. Make sure it's compiled correctly.")
        return {}, {}
    
    Q_auto = Q.detach().clone().requires_grad_(True)
    R_auto = R.detach().clone().requires_grad_(True)
    S_auto = S.detach().clone().requires_grad_(True)
    Vq_1_auto = Vq_1.detach().clone().requires_grad_(True)
    Vq_2_auto = Vq_2.detach().clone().requires_grad_(True)
    Vr_1_auto = Vr_1.detach().clone().requires_grad_(True)
    Vr_2_auto = Vr_2.detach().clone().requires_grad_(True)
    Vs_1_auto = Vs_1.detach().clone().requires_grad_(True)
    Vs_2_auto = Vs_2.detach().clone().requires_grad_(True)
    
    # === AUTOGRAD PATH ===
    output_auto = forward_pass(
        Q_auto, R_auto, S_auto,
        Vq_1_auto, Vq_2_auto,
        Vr_1_auto, Vr_2_auto,
        Vs_1_auto, Vs_2_auto
    )
    
    output_auto.backward(grad_output)
    
    autograd_grads = {
        'grad_Q': Q_auto.grad,
        'grad_R': R_auto.grad,
        'grad_S': S_auto.grad,
        'grad_Vq_1': Vq_1_auto.grad,
        'grad_Vq_2': Vq_2_auto.grad,
        'grad_Vr_1': Vr_1_auto.grad,
        'grad_Vr_2': Vr_2_auto.grad,
        'grad_Vs_1': Vs_1_auto.grad,
        'grad_Vs_2': Vs_2_auto.grad
    }
    
    # === MANUAL CPP PATH ===
    try:
        cpp_grads_tuple = hyper_attn_cpp_manual.backward(
            grad_output, Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2, 0.0
        )
        print("Successfully called hyper_attn_cpp_manual.backward function")
    except Exception as e:
        print(f"Error calling hyper_attn_cpp_manual.backward function: {e}")
        print("Input shapes:")
        print(f"  grad_output: {grad_output.shape}")
        print(f"  Q: {Q.shape}, R: {R.shape}, S: {S.shape}")
        print(f"  Vq_1: {Vq_1.shape}, Vq_2: {Vq_2.shape}")
        print(f"  Vr_1: {Vr_1.shape}, Vr_2: {Vr_2.shape}")
        print(f"  Vs_1: {Vs_1.shape}, Vs_2: {Vs_2.shape}")
        return {}, autograd_grads
    
    cpp_grads = {
        'grad_Q': cpp_grads_tuple[0],
        'grad_R': cpp_grads_tuple[1],
        'grad_S': cpp_grads_tuple[2],
        'grad_Vq_1': cpp_grads_tuple[3],
        'grad_Vq_2': cpp_grads_tuple[4],
        'grad_Vr_1': cpp_grads_tuple[5],
        'grad_Vr_2': cpp_grads_tuple[6],
        'grad_Vs_1': cpp_grads_tuple[7],
        'grad_Vs_2': cpp_grads_tuple[8]
    }
    
    print("\nChecking shapes of returned C++ gradients:")
    all_shapes_match = True
    for name in cpp_grads:
        if cpp_grads[name].shape != autograd_grads[name].shape:
            print(f"  Shape mismatch for {name}: C++ {cpp_grads[name].shape}, Autograd {autograd_grads[name].shape}")
            all_shapes_match = False
    if all_shapes_match:
        print("  All C++ gradient shapes match autograd.")
    else:
         return {}, autograd_grads # Stop if shapes mismatch
    
    return cpp_grads, autograd_grads

def run_tests():
    """Run all gradient tests."""
    print("=== Testing Gradient Computation in hyper_attn_backward.cpp ===\n")
    
    cpp_grads, autograd_grads = test_all_gradients()
    
    if not cpp_grads:
        print("Failed to obtain C++ gradients. Cannot proceed with tests.")
        return
    
    test_results = {}
    
    print("\n=== Detailed Test Results ===")
    
    value_tensors_1 = ['grad_Vq_1', 'grad_Vr_1', 'grad_Vs_1']
    for name in value_tensors_1:
        print(f"\n----- Testing {name} -----")
        test_results[name] = test_grad_tensors(name, cpp_grads[name], autograd_grads[name])
    
    value_tensors_2 = ['grad_Vq_2', 'grad_Vr_2', 'grad_Vs_2']
    for name in value_tensors_2:
        print(f"\n----- Testing {name} -----")
        test_results[name] = test_grad_tensors(name, cpp_grads[name], autograd_grads[name])
    
    query_key_tensors = ['grad_Q', 'grad_R', 'grad_S']
    for name in query_key_tensors:
        print(f"\n----- Testing {name} -----")
        test_results[name] = test_grad_tensors(name, cpp_grads[name], autograd_grads[name])
    
    print("\n=== Summary ===")
    all_passed = True
    for name, passed in test_results.items():
        print(f"{name} test: {'✓' if passed else '✗'}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 SUCCESS: All gradient implementations in hyper_attn_backward.cpp are correct!")
    else:
        print("\n⚠️ FAILURE: There are issues with one or more gradient implementations:")
        failed_tests = [name for name, passed in test_results.items() if not passed]
        for name in failed_tests:
            print(f"  - {name} implementation needs fixing")
        
if __name__ == "__main__":
    run_tests()