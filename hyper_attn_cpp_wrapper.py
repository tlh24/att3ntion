import torch
import torch.nn as nn
import math
import hyper_attn_cpp_manual

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class HypergraphAttentionCPP(nn.Module):
    """
    PyTorch wrapper for cpp/hyper_attn_cpp_manual.cpp.
    """
    def __init__(self, d_model, n_heads, dropout_rate=0):
        super(HypergraphAttentionCPP, self).__init__()

        torch.manual_seed(42)
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model
        
        self.Wq = nn.Linear(d_model, d_model*n_heads, bias=False)
        self.Wr = nn.Linear(d_model, d_model*n_heads, bias=False)
        self.Ws = nn.Linear(d_model, d_model*n_heads, bias=False)
        
        self.Wv_q = nn.Linear(d_model, d_model*n_heads*2, bias=True)
        self.Wv_r = nn.Linear(d_model, d_model*n_heads*2, bias=True)
        self.Wv_s = nn.Linear(d_model, d_model*n_heads*2, bias=True)
        
        self.Wo = nn.Linear(d_model, d_model, bias=True)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.gelu = QuickGELU()
        
    def forward(self, x):
        batch_size, ntok, d_model = x.shape
        
        Q = self.Wq(x)
        R = self.Wr(x)
        S = self.Ws(x)
        
        Vq = self.Wv_q(x)
        Vr = self.Wv_r(x)
        Vs = self.Wv_s(x)
        
        Q = Q.reshape(batch_size, ntok, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        R = R.reshape(batch_size, ntok, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        S = S.reshape(batch_size, ntok, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        Vq_1, Vq_2 = Vq.reshape(batch_size, ntok, self.n_heads, self.head_dim*2).permute(0, 2, 1, 3).split(self.head_dim, dim=-1)
        Vr_1, Vr_2 = Vr.reshape(batch_size, ntok, self.n_heads, self.head_dim*2).permute(0, 2, 1, 3).split(self.head_dim, dim=-1)
        Vs_1, Vs_2 = Vs.reshape(batch_size, ntok, self.n_heads, self.head_dim*2).permute(0, 2, 1, 3).split(self.head_dim, dim=-1)
        
        # Call the manual C++ extension for the core 3-way attention computation
        y_q, y_r, y_s, y_q_scatter, y_r_scatter, y_s_scatter = hyper_attn_cpp_manual.forward(
            Q, R, S,
            Vq_1, Vq_2, 
            Vr_1, Vr_2,
            Vs_1, Vs_2
        )
        
        # Combine the outputs (assuming we want to use all of them)
        y = y_q + y_r + y_s + y_q_scatter + y_r_scatter + y_s_scatter
        
        y = y.permute(0, 2, 1, 3).sum(dim=2).squeeze()
        y = self.gelu(y)
        y = self.Wo(y)
        
        return y

# Example usage
if __name__ == "__main__":
    import numpy as np
    import time
    from sklearn.metrics.pairwise import cosine_similarity
    from hyper_attn_pytorch import HypergraphAttention
    
    print("Testing HypergraphAttentionCPPmanual")
    def compare_attention_patterns(output1, output2):
        """Compare outputs using cosine similarity."""
        out1_flat = output1.detach().cpu().numpy().reshape(output1.shape[0], -1)
        out2_flat = output2.detach().cpu().numpy().reshape(output2.shape[0], -1)
        sim_matrix = cosine_similarity(out1_flat, out2_flat)
        return np.mean(np.diag(sim_matrix))
    
    def test_equivalence():
        """Test the equivalence of PyTorch and C++ implementation."""
        print("\n=== Testing Hypergraph Attention Equivalence ===")
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Test configurations - minimal set
        configs = [
            # (batch_size, seq_len, d_model, n_heads)
            (2, 10, 64, 4),      # Base case
            (8, 8, 64, 4),       # Square inputs
        ]
        
        all_passed = True
        all_times_pytorch = []
        all_times_cpp = []
        
        for config in configs:
            batch_size, seq_len, d_model, n_heads = config
            
            print(f"\n=== Configuration: batch={batch_size}, seq={seq_len}, dim={d_model}, heads={n_heads} ===")
            
            # Initialize models
            pytorch_attn = HypergraphAttention(d_model, n_heads)
            cpp_attn = HypergraphAttentionCPP(d_model, n_heads)
            
            # Set both to eval mode
            pytorch_attn.eval()
            cpp_attn.eval()
            
            # Make sure weights match for fair comparison
            cpp_attn.Wq.weight.data = pytorch_attn.Wq.weight.data.clone()
            cpp_attn.Wr.weight.data = pytorch_attn.Wr.weight.data.clone()
            cpp_attn.Ws.weight.data = pytorch_attn.Ws.weight.data.clone()
            cpp_attn.Wv_q.weight.data = pytorch_attn.Wv_q.weight.data.clone()
            cpp_attn.Wv_r.weight.data = pytorch_attn.Wv_r.weight.data.clone()
            cpp_attn.Wv_s.weight.data = pytorch_attn.Wv_s.weight.data.clone()
            cpp_attn.Wo.weight.data = pytorch_attn.Wo.weight.data.clone()
            cpp_attn.Wo.bias.data = pytorch_attn.Wo.bias.data.clone()
            
            # Create test inputs (minimal set of edge cases)
            inputs = [
                torch.randn(batch_size, seq_len, d_model),                    # Random normal
                torch.randn(batch_size, seq_len, d_model) * 0.01,             # Small values
                torch.cat([torch.ones(batch_size, seq_len//2, d_model),       # Step function
                          torch.zeros(batch_size, seq_len - seq_len//2, d_model)], dim=1)
            ]
            input_names = ["Random", "Small values", "Step function"]
            
            for i, x in enumerate(inputs):
                # Time PyTorch implementation
                start = time.time()
                with torch.no_grad():
                    pytorch_out = pytorch_attn(x)
                pytorch_time = time.time() - start
                
                # Time C++ implementation
                start = time.time()
                with torch.no_grad():
                    cpp_out = cpp_attn(x)
                cpp_time = time.time() - start
                
                all_times_pytorch.append(pytorch_time)
                all_times_cpp.append(cpp_time)
                
                # Check results
                shape_match = pytorch_out.shape == cpp_out.shape
                max_diff = torch.max(torch.abs(pytorch_out - cpp_out)).item()
                outputs_match = torch.allclose(pytorch_out, cpp_out, rtol=1e-4, atol=1e-4)
                cos_sim = compare_attention_patterns(pytorch_out, cpp_out)
                
                # Check for NaNs
                has_nan_pytorch = torch.isnan(pytorch_out).any().item()
                has_nan_cpp = torch.isnan(cpp_out).any().item()
                
                test_passed = shape_match and outputs_match and cos_sim > 0.999 and not has_nan_pytorch and not has_nan_cpp
                all_passed = all_passed and test_passed
                
                print(f"{input_names[i]}: {'✓' if test_passed else '✗'} " + 
                      f"(Max diff={max_diff:.8f}, Cosine sim={cos_sim:.6f}, " +
                      f"PyTorch: {pytorch_time*1000:.2f}ms, C++: {cpp_time*1000:.2f}ms)")
                
                if not test_passed:
                    print(f"  Shape match: {shape_match}")
                    print(f"  Values match: {outputs_match}")
                    print(f"  Has NaN (PyTorch): {has_nan_pytorch}")
                    print(f"  Has NaN (C++): {has_nan_cpp}")
        
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
    
    def test_stability():
        """Test that outputs are stable across multiple runs with the same input."""
        print("\n=== Testing Output Stability ===")
        torch.manual_seed(42)
        
        d_model = 64
        n_heads = 4
        
        # Initialize models
        pytorch_attn = HypergraphAttention(d_model, n_heads)
        cpp_attn = HypergraphAttentionCPP(d_model, n_heads)
        
        # Set both to eval mode
        pytorch_attn.eval()
        cpp_attn.eval()
        
        # Make sure weights match
        cpp_attn.Wq.weight.data = pytorch_attn.Wq.weight.data.clone()
        cpp_attn.Wr.weight.data = pytorch_attn.Wr.weight.data.clone()
        cpp_attn.Ws.weight.data = pytorch_attn.Ws.weight.data.clone()
        cpp_attn.Wv_q.weight.data = pytorch_attn.Wv_q.weight.data.clone()
        cpp_attn.Wv_r.weight.data = pytorch_attn.Wv_r.weight.data.clone()
        cpp_attn.Wv_s.weight.data = pytorch_attn.Wv_s.weight.data.clone()
        cpp_attn.Wo.weight.data = pytorch_attn.Wo.weight.data.clone()
        cpp_attn.Wo.bias.data = pytorch_attn.Wo.bias.data.clone()
        
        x = torch.randn(2, 10, d_model)
        
        with torch.no_grad():
            pytorch_out1 = pytorch_attn(x)
            cpp_out1 = cpp_attn(x)
        
        with torch.no_grad():
            pytorch_out2 = pytorch_attn(x)
            cpp_out2 = cpp_attn(x)
        
        pytorch_stable = torch.allclose(pytorch_out1, pytorch_out2, rtol=1e-5, atol=1e-5)
        cpp_stable = torch.allclose(cpp_out1, cpp_out2, rtol=1e-5, atol=1e-5)
        
        print(f"PyTorch output stable: {'✓' if pytorch_stable else '✗'}")
        print(f"C++ output stable: {'✓' if cpp_stable else '✗'}")
        
        return pytorch_stable and cpp_stable
    
    # Run basic example to show the module works
    batch_size = 4
    seq_len = 32
    d_model = 128
    n_heads = 8
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    model = HypergraphAttentionCPP(d_model, n_heads, dropout_rate=0.1).to(device)
    output = model(x)
    
    print(f"Basic functionality test:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    equivalence_result = test_equivalence()
    stability_result = test_stability()
    
    print("\n=== OVERALL TESTING RESULTS ===")
    print(f"Equivalence tests: {'✓' if equivalence_result else '✗'}")
    print(f"Stability tests: {'✓' if stability_result else '✗'}")
    
    if equivalence_result and stability_result:
        print("\n🎉 OVERALL RESULT: PASSED - Implementations are fully equivalent!")
    else:
        print("\n⚠️ OVERALL RESULT: FAILED - Implementations are not fully equivalent") 