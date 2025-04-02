import torch
import hyper_attn_cpp_manual        
import hyper_attn_cpp_reference  

def make_inputs(B=1, H=1, L=3, D=4, seed=42):
    torch.manual_seed(seed)
    Q = torch.randn(B, H, L, D)
    R = torch.randn(B, H, L, D)
    S = torch.randn(B, H, L, D)
    Vq_1 = torch.randn(B, H, L, D)
    Vq_2 = torch.randn(B, H, L, D)
    Vr_1 = torch.randn(B, H, L, D)
    Vr_2 = torch.randn(B, H, L, D)
    Vs_1 = torch.randn(B, H, L, D)
    Vs_2 = torch.randn(B, H, L, D)
    return Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2

def compare_forward():
    Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2 = make_inputs()

    Yq_ref, Yr_ref, Ys_ref, Yq_scatter_ref, Yr_scatter_ref, Ys_scatter_ref = hyper_attn_cpp_reference.forward(Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2)
    Yq_opt, Yr_opt, Ys_opt, Yq_scatter_opt, Yr_scatter_opt, Ys_scatter_opt = hyper_attn_cpp_manual.forward(Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2)

    for name, ref, opt in [("Y_q_gather", Yq_ref, Yq_opt), 
                          ("Y_r_gather", Yr_ref, Yr_opt), 
                          ("Y_s_gather", Ys_ref, Ys_opt),
                          ("Y_q_scatter", Yq_scatter_ref, Yq_scatter_opt),
                          ("Y_r_scatter", Yr_scatter_ref, Yr_scatter_opt),
                          ("Y_s_scatter", Ys_scatter_ref, Ys_scatter_opt)]:
        close = torch.allclose(ref, opt, atol=1e-4, rtol=1e-4)
        diff = (ref - opt).abs().max().item()
        print(f"\n{name} {'✅ passed' if close else '❌ FAILED'} | Max diff: {diff:.6f}")
        if not close:
            print(f"\n{name} Reference:\n{ref}\n\n{name} Optimized:\n{opt}")

def compare_backward():
    """Compare the backward pass implementation with PyTorch autograd"""
    print("\n===== Comparing Backward Pass =====")
    
    # Enable debug mode for more detailed tracing
    debug = True
    
    # Create input tensors with gradients for autograd
    torch.manual_seed(42)
    Q = torch.randn(1, 1, 3, 4, requires_grad=True)
    R = torch.randn(1, 1, 3, 4, requires_grad=True)
    S = torch.randn(1, 1, 3, 4, requires_grad=True)
    Vq_1 = torch.randn(1, 1, 3, 4, requires_grad=True)
    Vq_2 = torch.randn(1, 1, 3, 4, requires_grad=True)
    Vr_1 = torch.randn(1, 1, 3, 4, requires_grad=True)
    Vr_2 = torch.randn(1, 1, 3, 4, requires_grad=True)
    Vs_1 = torch.randn(1, 1, 3, 4, requires_grad=True)
    Vs_2 = torch.randn(1, 1, 3, 4, requires_grad=True)
    
    # Create identical copies for the C++ implementation
    Q_cpp = Q.detach().clone()
    R_cpp = R.detach().clone()
    S_cpp = S.detach().clone()
    Vq_1_cpp = Vq_1.detach().clone()
    Vq_2_cpp = Vq_2.detach().clone()
    Vr_1_cpp = Vr_1.detach().clone()
    Vr_2_cpp = Vr_2.detach().clone()
    Vs_1_cpp = Vs_1.detach().clone()
    Vs_2_cpp = Vs_2.detach().clone()
    
    if debug:
        print("\n==== Input Tensor Stats ====")
        for name, tensor in [("Q", Q), ("R", R), ("S", S), 
                            ("Vq_1", Vq_1), ("Vq_2", Vq_2),
                            ("Vr_1", Vr_1), ("Vr_2", Vr_2),
                            ("Vs_1", Vs_1), ("Vs_2", Vs_2)]:
            print(f"{name}: mean={tensor.mean().item():.4f}, max={tensor.max().item():.4f}, min={tensor.min().item():.4f}")
    
    # Define a hook to capture intermediate values
    intermediate_values = {}
    def save_gradients(name):
        def hook(grad):
            intermediate_values[f'grad_{name}'] = grad.detach().clone()
            return grad
        return hook
    
    # Register hooks on key tensors
    Q.register_hook(save_gradients('Q'))
    R.register_hook(save_gradients('R'))
    S.register_hook(save_gradients('S'))
    Vq_1.register_hook(save_gradients('Vq_1'))
    Vq_2.register_hook(save_gradients('Vq_2'))
    Vr_1.register_hook(save_gradients('Vr_1'))
    Vr_2.register_hook(save_gradients('Vr_2'))
    Vs_1.register_hook(save_gradients('Vs_1'))
    Vs_2.register_hook(save_gradients('Vs_2'))
    
    # Forward pass with autograd using the reference implementation
    with torch.autograd.detect_anomaly():
        Y_q, Y_r, Y_s, Y_q_, Y_r_, Y_s_ = hyper_attn_cpp_reference.forward(
            Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2)
    
    # Register hooks on output tensors
    Y_q.register_hook(save_gradients('Y_q'))
    Y_r.register_hook(save_gradients('Y_r'))
    Y_s.register_hook(save_gradients('Y_s'))
    Y_q_.register_hook(save_gradients('Y_q_'))
    Y_r_.register_hook(save_gradients('Y_r_'))
    Y_s_.register_hook(save_gradients('Y_s_'))
    
    # Forward pass with C++ implementation
    Y_q_cpp, Y_r_cpp, Y_s_cpp, Y_q_cpp_, Y_r_cpp_, Y_s_cpp_ = hyper_attn_cpp_manual.forward(
        Q_cpp, R_cpp, S_cpp, Vq_1_cpp, Vq_2_cpp, Vr_1_cpp, Vr_2_cpp, Vs_1_cpp, Vs_2_cpp)
    
    # Compare forward outputs
    if debug:
        print("\n==== Forward Pass Comparison ====")
        for name, torch_out, cpp_out in [
            ("Y_q", Y_q, Y_q_cpp),
            ("Y_r", Y_r, Y_r_cpp),
            ("Y_s", Y_s, Y_s_cpp),
            ("Y_q_", Y_q_, Y_q_cpp_),
            ("Y_r_", Y_r_, Y_r_cpp_),
            ("Y_s_", Y_s_, Y_s_cpp_)
        ]:
            diff = (torch_out - cpp_out).abs().max().item()
            print(f"{name}: max diff = {diff:.6f}")
    
    # Create random gradients for output
    torch.manual_seed(43)  # Different seed for gradients
    grad_Y_q = torch.randn_like(Y_q)
    grad_Y_r = torch.randn_like(Y_r)
    grad_Y_s = torch.randn_like(Y_s)
    grad_Y_q_ = torch.randn_like(Y_q_)
    grad_Y_r_ = torch.randn_like(Y_r_)
    grad_Y_s_ = torch.randn_like(Y_s_)
    
    if debug:
        print("\n==== Gradient Tensor Stats ====")
        for name, tensor in [("grad_Y_q", grad_Y_q), ("grad_Y_r", grad_Y_r), ("grad_Y_s", grad_Y_s),
                            ("grad_Y_q_", grad_Y_q_), ("grad_Y_r_", grad_Y_r_), ("grad_Y_s_", grad_Y_s_)]:
            print(f"{name}: mean={tensor.mean().item():.4f}, max={tensor.max().item():.4f}, min={tensor.min().item():.4f}")
    
    # Backward pass with autograd
    Y_sum = Y_q + Y_r + Y_s + Y_q_ + Y_r_ + Y_s_
    grad_sum = grad_Y_q + grad_Y_r + grad_Y_s + grad_Y_q_ + grad_Y_r_ + grad_Y_s_
    Y_sum.backward(grad_sum)
    
    # Backward pass with C++ implementation
    dQ_cpp, dR_cpp, dS_cpp, dVq_1_cpp, dVq_2_cpp, dVr_1_cpp, dVr_2_cpp, dVs_1_cpp, dVs_2_cpp = hyper_attn_cpp_manual.backward(
        grad_Y_q, grad_Y_r, grad_Y_s, grad_Y_q_, grad_Y_r_, grad_Y_s_,
        Q_cpp, R_cpp, S_cpp, Vq_1_cpp, Vq_2_cpp, Vr_1_cpp, Vr_2_cpp, Vs_1_cpp, Vs_2_cpp)
    
    # Debug intermediate values from PyTorch's backward pass
    if debug:
        print("\n==== PyTorch Intermediate Values ====")
        for key, value in intermediate_values.items():
            if key.startswith('grad_'):
                print(f"{key}: mean={value.mean().item():.4f}, max={value.max().item():.4f}, min={value.min().item():.4f}")
                
        # Print the first few values of key gradient tensors
        for name in ['grad_Q', 'grad_R', 'grad_S']:
            if name in intermediate_values:
                grad = intermediate_values[name]
                print(f"\n{name} first elements: {grad.flatten()[:10].tolist()}")
    
    # Debug information to identify scaling issues
    if debug:
        print("\n==== Gradient Magnitude Analysis ====")
        for name, autograd, cpp in [
            ("dQ", Q.grad, dQ_cpp),
            ("dR", R.grad, dR_cpp),
            ("dS", S.grad, dS_cpp)
        ]:
            autograd_norm = torch.norm(autograd).item()
            cpp_norm = torch.norm(cpp).item()
            ratio = autograd_norm / (cpp_norm + 1e-10)
            print(f"{name}: PyTorch norm = {autograd_norm:.6f}, C++ norm = {cpp_norm:.6f}, ratio = {ratio:.6f}")
    
    # Compare gradients
    for name, autograd, cpp in [
        ("dQ", Q.grad, dQ_cpp),
        ("dR", R.grad, dR_cpp),
        ("dS", S.grad, dS_cpp),
        ("dVq_1", Vq_1.grad, dVq_1_cpp),
        ("dVq_2", Vq_2.grad, dVq_2_cpp),
        ("dVr_1", Vr_1.grad, dVr_1_cpp),
        ("dVr_2", Vr_2.grad, dVr_2_cpp),
        ("dVs_1", Vs_1.grad, dVs_1_cpp),
        ("dVs_2", Vs_2.grad, dVs_2_cpp)
    ]:
        close = torch.allclose(autograd, cpp, atol=1e-4, rtol=1e-4)
        diff = (autograd - cpp).abs().max().item()
        print(f"\n{name} {'✅ passed' if close else '❌ FAILED'} | Max diff: {diff:.6f}")
        if not close and debug:
            # Print detailed stats of both gradients
            print(f"{name} PyTorch: mean={autograd.mean().item():.6f}, max={autograd.max().item():.6f}, min={autograd.min().item():.6f}")
            print(f"{name} C++:     mean={cpp.mean().item():.6f}, max={cpp.max().item():.6f}, min={cpp.min().item():.6f}")
            
            # Print the first few elements of both tensors
            print("\nFirst few elements comparison:")
            print(f"PyTorch: {autograd.flatten()[:5].tolist()}")
            print(f"C++:     {cpp.flatten()[:5].tolist()}")
            
            # Check if there's a consistent scale factor
            ratios = autograd.flatten() / (cpp.flatten() + 1e-10)
            avg_ratio = ratios[~torch.isnan(ratios) & ~torch.isinf(ratios)].mean().item()
            print(f"Average ratio: {avg_ratio:.6f}")


if __name__ == "__main__":
    compare_forward()
    compare_backward()
