import torch
import numpy as np
from hyper_attn_pytorch import HypergraphAttention
import hyper_attn_cpp
from sklearn.metrics.pairwise import cosine_similarity
import time

def test_equivalence(verbose=True, check_grads=False):
    """Test the equivalence of PyTorch and C++ implementation.
       Handles: 
       - Shape matching
       - Output value matching
       - NaN value matching
       - Performance comparison
    """
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Test configurations
    configs = [
        # (batch_size, seq_len, d_model, n_heads)
        (2, 10, 64, 4),       # Base case
        (1, 5, 32, 2),        # Small model, small sequence
        (4, 20, 128, 8),      # Larger model
        (8, 8, 64, 4),        # Square inputs
        (3, 15, 48, 3),       # Different dimensions
    ]
    
    all_passed = True
    all_times_pytorch = []
    all_times_cpp = []
    
    for config in configs:
        batch_size, seq_len, d_model, n_heads = config
        
        if verbose:
            print(f"\n=== Testing configuration: batch={batch_size}, seq={seq_len}, dim={d_model}, heads={n_heads} ===")
        
        # Initialize models
        pytorch_attn = HypergraphAttention(d_model, n_heads)
        pytorch_attn.eval()  # Set to eval mode
        
        # Create input (use both normal and edge case values)
        inputs = [
            torch.randn(batch_size, seq_len, d_model),                     # Random normal
            torch.randn(batch_size, seq_len, d_model) * 0.01,              # Small values
            torch.randn(batch_size, seq_len, d_model) * 10,                # Large values
            torch.ones(batch_size, seq_len, d_model) * 0.5,                # Constant values
            torch.cat([torch.ones(batch_size, seq_len//2, d_model),        # Step function
                       torch.zeros(batch_size, seq_len - seq_len//2, d_model)], dim=1)
        ]
        
        input_names = ["Random", "Small", "Large", "Constant", "Step"]
        
        for i, x in enumerate(inputs):
            try:
                # Time PyTorch implementation
                start = time.time()
                with torch.no_grad():
                    pytorch_out = pytorch_attn(x)
                pytorch_time = time.time() - start
                
                # Time C++ implementation
                start = time.time()
                with torch.no_grad():
                    cpp_out = hyper_attn_cpp.forward(x, d_model, n_heads, 0.0)
                cpp_time = time.time() - start
                
                all_times_pytorch.append(pytorch_time)
                all_times_cpp.append(cpp_time)
                
                # Basic shape check
                shape_match = pytorch_out.shape == cpp_out.shape
                
                # Check if outputs are close
                max_diff = torch.max(torch.abs(pytorch_out - cpp_out)).item()
                outputs_match = torch.allclose(pytorch_out, cpp_out, rtol=1e-4, atol=1e-4)
                
                # Check with cosine similarity as well
                cos_sim = compare_attention_patterns(pytorch_out, cpp_out)
                
                # Skip gradient check for this version
                grad_match = True
                
                # Check for NaN values
                has_nan_pytorch = torch.isnan(pytorch_out).any().item()
                has_nan_cpp = torch.isnan(cpp_out).any().item()
                
                test_passed = shape_match and outputs_match and cos_sim > 0.999 and not has_nan_pytorch and not has_nan_cpp
                all_passed = all_passed and test_passed
                
                if verbose:
                    print(f"{input_names[i]}: {'✓' if test_passed else '✗'} " + 
                          f"(Max diff={max_diff:.8f}, Cosine sim={cos_sim:.6f}, " +
                          f"PyTorch: {pytorch_time*1000:.2f}ms, C++: {cpp_time*1000:.2f}ms)")
                    
                    if not test_passed:
                        print(f"  Shape match: {shape_match}")
                        print(f"  Values match: {outputs_match}")
                        print(f"  Has NaN (PyTorch): {has_nan_pytorch}")
                        print(f"  Has NaN (C++): {has_nan_cpp}")
                
            except Exception as e:
                all_passed = False
                if verbose:
                    print(f"{input_names[i]}: ✗ Exception: {str(e)}")
    
    # Performance comparison
    avg_pytorch = sum(all_times_pytorch) / len(all_times_pytorch) * 1000
    avg_cpp = sum(all_times_cpp) / len(all_times_cpp) * 1000
    speedup = avg_pytorch / avg_cpp if avg_cpp > 0 else float('inf')
    
    print("\n=== Performance Summary ===")
    print(f"Average time (PyTorch): {avg_pytorch:.2f}ms")
    print(f"Average time (C++): {avg_cpp:.2f}ms")
    print(f"Speedup: {speedup:.2f}x")
    
    print("\n=== Final Result ===")
    if all_passed:
        print("✓ All tests passed! PyTorch and C++ implementations are equivalent")
    else:
        print("✗ Some tests failed. Implementations are not fully equivalent")
    
    return all_passed

def compare_attention_patterns(output1, output2, threshold=0.999):
    """Compare outputs using cosine similarity."""
    out1_flat = output1.detach().cpu().numpy().reshape(output1.shape[0], -1)
    out2_flat = output2.detach().cpu().numpy().reshape(output2.shape[0], -1)
    
    sim_matrix = cosine_similarity(out1_flat, out2_flat)
    
    avg_sim = np.mean(np.diag(sim_matrix))
    
    return avg_sim

def test_edge_cases():
    """Test some edge cases that might be problematic."""
    torch.manual_seed(42)
    
    # Model parameters
    d_model = 64
    n_heads = 4
    
    # Initialize PyTorch model
    pytorch_attn = HypergraphAttention(d_model, n_heads)
    pytorch_attn.eval()
    
    test_cases = [
        # Zero values
        torch.zeros(2, 10, d_model),
        
        # Very large batch size, small sequence
        torch.randn(128, 2, d_model),
        
        # Small batch size, large sequence
        torch.randn(1, 100, d_model),
        
        # Large values with non-uniform distribution
        torch.cat([torch.randn(1, 5, d_model) * 100, 
                   torch.randn(1, 5, d_model) * 0.01], dim=1),
    ]
    
    all_passed = True
    
    print("\n=== Testing Edge Cases ===")
    for i, x in enumerate(test_cases):
        try:
            with torch.no_grad():
                pytorch_out = pytorch_attn(x)
                cpp_out = hyper_attn_cpp.forward(x, d_model, n_heads, 0.0)
            
            # Check if outputs are close
            outputs_match = torch.allclose(pytorch_out, cpp_out, rtol=1e-4, atol=1e-4)
            all_passed = all_passed and outputs_match
            
            max_diff = torch.max(torch.abs(pytorch_out - cpp_out)).item()
            cos_sim = compare_attention_patterns(pytorch_out, cpp_out)
            
            print(f"Edge case {i+1}: {'✓' if outputs_match else '✗'} " +
                  f"(Max diff={max_diff:.8f}, Cosine sim={cos_sim:.6f})")
            
        except Exception as e:
            all_passed = False
            print(f"Edge case {i+1}: ✗ Exception: {str(e)}")
    
    print("\n=== Edge Cases Summary ===")
    if all_passed:
        print("✓ All edge cases passed")
    else:
        print("✗ Some edge cases failed")
    
    return all_passed

def test_with_different_dropouts():
    """Test behavior with different dropout rates."""
    torch.manual_seed(42)
    
    # Model parameters
    d_model = 64
    n_heads = 4
    dropout_rates = [0.0, 0.1, 0.3, 0.5]
    
    print("\n=== Testing Different Dropout Rates (Eval Mode) ===")
    all_passed = True
    
    for rate in dropout_rates:
        # Initialize PyTorch model with dropout
        pytorch_attn = HypergraphAttention(d_model, n_heads, dropout_rate=rate)
        pytorch_attn.eval()  # Important: set to eval mode
        
        x = torch.randn(3, 10, d_model)
        
        with torch.no_grad():
            pytorch_out = pytorch_attn(x)
            cpp_out = hyper_attn_cpp.forward(x, d_model, n_heads, rate)
        
        # Outputs should match in eval mode regardless of dropout rate
        outputs_match = torch.allclose(pytorch_out, cpp_out, rtol=1e-4, atol=1e-4)
        all_passed = all_passed and outputs_match
        
        print(f"Dropout rate {rate}: {'✓' if outputs_match else '✗'}")
    
    print("\n=== Dropout Test Summary ===")
    if all_passed:
        print("✓ All dropout tests passed in eval mode")
    else:
        print("✗ Some dropout tests failed")
    
    return all_passed

def test_output_stability():
    """Test that outputs are stable across multiple runs with same input."""
    torch.manual_seed(42)
    
    # Model parameters
    d_model = 64
    n_heads = 4
    
    # Initialize PyTorch model
    pytorch_attn = HypergraphAttention(d_model, n_heads)
    pytorch_attn.eval()
    
    x = torch.randn(2, 10, d_model)
    
    print("\n=== Testing Output Stability ===")
    
    # First run
    with torch.no_grad():
        pytorch_out1 = pytorch_attn(x)
        cpp_out1 = hyper_attn_cpp.forward(x, d_model, n_heads, 0.0)
    
    # Second run
    with torch.no_grad():
        pytorch_out2 = pytorch_attn(x)
        cpp_out2 = hyper_attn_cpp.forward(x, d_model, n_heads, 0.0)
    
    # Check stability
    pytorch_stable = torch.allclose(pytorch_out1, pytorch_out2, rtol=1e-5, atol=1e-5)
    cpp_stable = torch.allclose(cpp_out1, cpp_out2, rtol=1e-5, atol=1e-5)
    
    print(f"PyTorch output stable: {'✓' if pytorch_stable else '✗'}")
    print(f"C++ output stable: {'✓' if cpp_stable else '✗'}")
    
    all_stable = pytorch_stable and cpp_stable
    
    print("\n=== Stability Test Summary ===")
    if all_stable:
        print("✓ Both implementations produce stable outputs")
    else:
        print("✗ Some stability issues detected")
    
    return all_stable

if __name__ == "__main__":
    main_test_passed = test_equivalence(verbose=True, check_grads=False)
    edge_test_passed = test_edge_cases()
    dropout_test_passed = test_with_different_dropouts()
    stability_test_passed = test_output_stability()
    
    all_tests_passed = (main_test_passed and edge_test_passed and 
                         dropout_test_passed and stability_test_passed)
    
    print("\n=== OVERALL TESTING RESULTS ===")
    print(f"Main equivalence tests: {'✓' if main_test_passed else '✗'}")
    print(f"Edge case tests: {'✓' if edge_test_passed else '✗'}")
    print(f"Dropout tests: {'✓' if dropout_test_passed else '✗'}")
    print(f"Stability tests: {'✓' if stability_test_passed else '✗'}")
    
    if all_tests_passed:
        print("\n🎉 OVERALL RESULT: PASSED - Implementations are fully equivalent!")
    else:
        print("\n⚠️ OVERALL RESULT: PARTIAL - Some tests passed, but complete equivalence not confirmed")