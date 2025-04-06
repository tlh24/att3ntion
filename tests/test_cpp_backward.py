import torch
import torch.nn as nn
import math
import numpy as np

def forward_pass(Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2):
    """
    Forward pass that matches the C++ implementation.
    """
    batch_size, n_heads = Q.shape[0], Q.shape[1]
    seq_len, d_model = Q.shape[2], Q.shape[3]
    
    # Compute dot product for attention weights
    scale = 1.0 / math.sqrt(d_model)
    attention = torch.einsum("bhid,bhjd,bhkd->bhijk", (Q, R, S)) * scale
    
    # Compute Aq (softmax over j,k for each i)
    attention_flat_q = attention.reshape(batch_size*n_heads*seq_len, seq_len*seq_len)
    Aq = torch.softmax(attention_flat_q, dim=-1).reshape(batch_size, n_heads, seq_len, seq_len, seq_len)
    
    # Compute Ar (softmax over i,k for each j)
    attention_perm_r = attention.permute(0, 1, 3, 2, 4)
    Ar_flat = attention_perm_r.reshape(batch_size*n_heads*seq_len, seq_len*seq_len)
    Ar = torch.softmax(Ar_flat, dim=-1).reshape(batch_size, n_heads, seq_len, seq_len, seq_len)
    Ar = Ar.permute(0, 1, 3, 2, 4)
    
    # Compute As (softmax over i,j for each k)
    attention_perm_s = attention.permute(0, 1, 4, 2, 3)
    As_flat = attention_perm_s.reshape(batch_size*n_heads*seq_len, seq_len*seq_len)
    As = torch.softmax(As_flat, dim=-1).reshape(batch_size, n_heads, seq_len, seq_len, seq_len)
    As = As.permute(0, 1, 3, 4, 2)
    
    # Y_q (gather to i)
    Y_q = torch.einsum("bhijk,bhjd,bhkd->bhid", (Aq, Vr_1, Vs_1))
    
    # Y_r (gather to j)
    Y_r = torch.einsum("bhijk,bhid,bhkd->bhjd", (Ar, Vq_1, Vs_1))
    
    # Y_s (gather to k)
    Y_s = torch.einsum("bhijk,bhid,bhjd->bhkd", (As, Vq_1, Vr_1))
    
    # Scatter computations (Y_q_, Y_r_, Y_s_)
    Y_q_ = torch.einsum("bhijk,bhjd,bhkd->bhid", (Ar, Vr_2, Vs_2))
    Y_r_ = torch.einsum("bhijk,bhid,bhkd->bhjd", (As, Vq_2, Vs_2))
    Y_s_ = torch.einsum("bhijk,bhid,bhjd->bhkd", (Aq, Vq_2, Vr_2))
    
    # Sum all components
    return Y_q + Y_r + Y_s + Y_q_ + Y_r_ + Y_s_

def test_grad_tensors(name, grad_cpp, grad_auto, print_samples=True):
    """Utility function to compare gradients and report results."""
    test_passed = torch.allclose(grad_auto, grad_cpp, rtol=1e-4, atol=1e-4)
    print(f"{name} gradient matches PyTorch autograd: {'✓' if test_passed else '✗'}")
    
    if not test_passed:
        max_diff = torch.max(torch.abs(grad_auto - grad_cpp)).item()
        print(f"  Max difference: {max_diff:.8f}")
        
        # Check if we're getting all zeros (likely implementation issue)
        if torch.all(grad_cpp == 0):
            print("  WARNING: C++ implementation is returning all zeros!")
        
        # Compute cosine similarity to check if directions are at least correct
        if torch.norm(grad_cpp) > 1e-10 and torch.norm(grad_auto) > 1e-10:
            cosine_sim = torch.nn.functional.cosine_similarity(
                grad_cpp.reshape(-1), grad_auto.reshape(-1), dim=0
            ).item()
            print(f"  Cosine similarity: {cosine_sim:.6f}")
        else:
            print("  Cosine similarity: N/A (one or both tensors are effectively zero)")
    
    # Print some sample values if requested
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
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Test with a small configuration
    batch_size = 2
    seq_len = 5
    d_model = 16
    n_heads = 2
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create test tensors
    Q = torch.randn(batch_size, n_heads, seq_len, d_model, requires_grad=True, device=device)
    R = torch.randn(batch_size, n_heads, seq_len, d_model, requires_grad=True, device=device)
    S = torch.randn(batch_size, n_heads, seq_len, d_model, requires_grad=True, device=device)
    
    Vq_1 = torch.randn(batch_size, n_heads, seq_len, d_model, requires_grad=True, device=device)
    Vq_2 = torch.randn(batch_size, n_heads, seq_len, d_model, requires_grad=True, device=device)
    Vr_1 = torch.randn(batch_size, n_heads, seq_len, d_model, requires_grad=True, device=device)
    Vr_2 = torch.randn(batch_size, n_heads, seq_len, d_model, requires_grad=True, device=device)
    Vs_1 = torch.randn(batch_size, n_heads, seq_len, d_model, requires_grad=True, device=device)
    Vs_2 = torch.randn(batch_size, n_heads, seq_len, d_model, requires_grad=True, device=device)
    
    # Generate a random gradient tensor
    grad_output = torch.randn(batch_size, n_heads, seq_len, d_model, device=device)
    
    # Import the specific backward module
    try:
        import hyper_attn_cpp_backward
        print("Successfully imported hyper_attn_cpp_backward module")
    except ImportError:
        print("Failed to import hyper_attn_cpp_backward. Make sure it's compiled correctly.")
        return {}, {}
    
    # Compute with PyTorch autograd
    # Make copies of inputs to ensure we're using fresh tensors for autograd
    Q_auto = Q.detach().clone().requires_grad_(True)
    R_auto = R.detach().clone().requires_grad_(True)
    S_auto = S.detach().clone().requires_grad_(True)
    Vq_1_auto = Vq_1.detach().clone().requires_grad_(True)
    Vq_2_auto = Vq_2.detach().clone().requires_grad_(True)
    Vr_1_auto = Vr_1.detach().clone().requires_grad_(True)
    Vr_2_auto = Vr_2.detach().clone().requires_grad_(True)
    Vs_1_auto = Vs_1.detach().clone().requires_grad_(True)
    Vs_2_auto = Vs_2.detach().clone().requires_grad_(True)
    
    # Forward pass
    output = forward_pass(
        Q_auto, R_auto, S_auto, 
        Vq_1_auto, Vq_2_auto, 
        Vr_1_auto, Vr_2_auto, 
        Vs_1_auto, Vs_2_auto
    )
    
    # Backward pass through PyTorch autograd
    output.backward(grad_output)
    
    # Get gradients from PyTorch autograd
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
    
    # Call the C++ backward function directly to get gradients
    try:
        cpp_grads_tuple = hyper_attn_cpp_backward.backward(
            grad_output, Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2, 0.0
        )
        print("Successfully called backward function")
    except Exception as e:
        print(f"Error calling backward function: {e}")
        return {}, autograd_grads
    
    # Extract gradients from the C++ output tuple
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
    
    return cpp_grads, autograd_grads

def run_tests():
    """Run all gradient tests."""
    print("=== Testing Gradient Computation in hyper_attn_backward.cpp ===\n")
    
    # Test all gradients
    cpp_grads, autograd_grads = test_all_gradients()
    
    if not cpp_grads:
        print("Failed to obtain C++ gradients. Cannot proceed with tests.")
        return
    
    # Dictionary to store test results
    test_results = {}
    
    # Test each gradient
    print("\n=== Detailed Test Results ===")
    
    # First test the main value tensor gradients (Vq_1, Vr_1, Vs_1)
    # These are the most important for the core functionality
    main_tensors = ['grad_Vq_1', 'grad_Vr_1', 'grad_Vs_1']
    for name in main_tensors:
        print(f"\n----- Testing {name} -----")
        test_results[name] = test_grad_tensors(name, cpp_grads[name], autograd_grads[name])
    
    # Then test secondary value tensor gradients
    secondary_tensors = ['grad_Vq_2', 'grad_Vr_2', 'grad_Vs_2']
    for name in secondary_tensors:
        print(f"\n----- Testing {name} -----")
        test_results[name] = test_grad_tensors(name, cpp_grads[name], autograd_grads[name])
    
    # Finally test the query/key/value tensor gradients
    # These tend to have the most complex gradient calculations
    query_key_tensors = ['grad_Q', 'grad_R', 'grad_S']
    for name in query_key_tensors:
        print(f"\n----- Testing {name} -----")
        test_results[name] = test_grad_tensors(name, cpp_grads[name], autograd_grads[name])
    
    # Print summary
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
        
        # Additional helpful information for debugging
        print("\nCommon issues to check:")
        print("1. Missing index permutations in attention tensors")
        print("2. Incorrect einsum dimension ordering")
        print("3. Missing function calls in hyper_attn_backward() function")
        print("4. Tensor dimension mismatches")
        print("5. For grad_Vs_1, check if compute_grad_Vs_1() is being called with correct parameters")

if __name__ == "__main__":
    run_tests()