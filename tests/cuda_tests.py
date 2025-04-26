import torch
import os
import sys

# --- Configuration ---
B = 2
H = 2
I = 4
J = 4
K = 4
D = 8
dtype = torch.float32
device_cpu = torch.device("cpu")
device_cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rtol = 1e-4
atol = 1e-5

print(f"Testing Gather Equivalence (CPU vs CUDA)")
print(f"Parameters: B={B}, H={H}, I={I}, J={J}, K={K}, D={D}")
print(f"CPU Device: {device_cpu}")
print(f"CUDA Device: {device_cuda}")

# --- Import Extension ---
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

try:
    # Use the correct module name as defined in setup.py
    import hyper_attn_cpp_manual as manual_att3ntion
    print("Successfully imported C++/CUDA extension.")
except ImportError:
    print("\nError: Failed to import the C++/CUDA extension 'hyper_attn_cpp_manual'.")
    print("Please ensure the extension has been compiled successfully and the name matches setup.py.")
    sys.exit(1) # Exit if the extension cannot be imported

# --- Helper Function ---
def generate_inputs(device):
    """Generates random input tensors on the specified device."""
    Q = torch.randn(B, H, I, D, dtype=dtype, device=device)
    R = torch.randn(B, H, J, D, dtype=dtype, device=device)
    S = torch.randn(B, H, K, D, dtype=dtype, device=device)
    Vq_1 = torch.randn(B, H, I, D, dtype=dtype, device=device)
    Vq_2 = torch.randn(B, H, I, D, dtype=dtype, device=device)
    Vr_1 = torch.randn(B, H, J, D, dtype=dtype, device=device)
    Vr_2 = torch.randn(B, H, J, D, dtype=dtype, device=device)
    Vs_1 = torch.randn(B, H, K, D, dtype=dtype, device=device)
    Vs_2 = torch.randn(B, H, K, D, dtype=dtype, device=device)
    dropout_rate = 0.0 # Not relevant for gather equivalence
    return Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2, dropout_rate

# --- Main Test Logic ---
def run_test():
    # --- CPU Execution ---
    print("\nRunning forward pass on CPU...")
    Q_cpu, R_cpu, S_cpu, Vq_1_cpu, Vq_2_cpu, Vr_1_cpu, Vr_2_cpu, Vs_1_cpu, Vs_2_cpu, dr_cpu = generate_inputs(device_cpu)
    
    assert Q_cpu.device.type == 'cpu', "Input tensors not on CPU!"
    
    try:
        Y_tuple_cpu = manual_att3ntion.forward(Q_cpu, R_cpu, S_cpu, Vq_1_cpu, Vq_2_cpu, Vr_1_cpu, Vr_2_cpu, Vs_1_cpu, Vs_2_cpu, dr_cpu)
        Y_q_cpu, Y_r_cpu, Y_s_cpu, _, _, _ = Y_tuple_cpu # Extract gather outputs
        print("CPU forward pass completed.")
    except Exception as e:
        print(f"Error during CPU forward pass: {e}")
        sys.exit(1)

    # --- CUDA Execution & Comparison ---
    if not torch.cuda.is_available():
        print("\nCUDA not available. Skipping CUDA execution and comparison.")
        return

    print("\nMoving inputs to CUDA device...")
    try:
        Q_cuda = Q_cpu.to(device_cuda)
        R_cuda = R_cpu.to(device_cuda)
        S_cuda = S_cpu.to(device_cuda)
        Vq_1_cuda = Vq_1_cpu.to(device_cuda)
        Vq_2_cuda = Vq_2_cpu.to(device_cuda) # Needed for signature
        Vr_1_cuda = Vr_1_cpu.to(device_cuda)
        Vr_2_cuda = Vr_2_cpu.to(device_cuda) # Needed for signature
        Vs_1_cuda = Vs_1_cpu.to(device_cuda)
        Vs_2_cuda = Vs_2_cpu.to(device_cuda) # Needed for signature
        dr_cuda = dr_cpu 
        assert Q_cuda.device.type == 'cuda', "Input tensors not on CUDA!"
        print("Inputs moved to CUDA.")
    except Exception as e:
        print(f"Error moving inputs to CUDA: {e}")
        sys.exit(1)

    print("Running forward pass on CUDA...")
    try:
        Y_tuple_cuda = manual_att3ntion.forward(Q_cuda, R_cuda, S_cuda, Vq_1_cuda, Vq_2_cuda, Vr_1_cuda, Vr_2_cuda, Vs_1_cuda, Vs_2_cuda, dr_cuda)
        Y_q_cuda, Y_r_cuda, Y_s_cuda, _, _, _ = Y_tuple_cuda # Extract gather outputs
        print("CUDA forward pass completed.")
    except Exception as e:
        print(f"Error during CUDA forward pass: {e}")
        sys.exit(1)

    print("\nComparing CPU and CUDA gather outputs...")
    # Move CUDA results back to CPU for comparison
    Y_q_cuda_cpu = Y_q_cuda.cpu()
    Y_r_cuda_cpu = Y_r_cuda.cpu()
    Y_s_cuda_cpu = Y_s_cuda.cpu()

    # Check shapes
    shape_match = True
    if Y_q_cpu.shape != Y_q_cuda_cpu.shape:
        print(f"ERROR: Y_q shape mismatch! CPU: {Y_q_cpu.shape}, CUDA: {Y_q_cuda_cpu.shape}")
        shape_match = False
    if Y_r_cpu.shape != Y_r_cuda_cpu.shape:
        print(f"ERROR: Y_r shape mismatch! CPU: {Y_r_cpu.shape}, CUDA: {Y_r_cuda_cpu.shape}")
        shape_match = False
    if Y_s_cpu.shape != Y_s_cuda_cpu.shape:
        print(f"ERROR: Y_s shape mismatch! CPU: {Y_s_cpu.shape}, CUDA: {Y_s_cuda_cpu.shape}")
        shape_match = False
        
    if not shape_match:
        print("Exiting due to shape mismatches.")
        sys.exit(1)
    else:
        print("Shapes match.")

    # Check numerical equivalence
    yq_close = torch.allclose(Y_q_cpu, Y_q_cuda_cpu, rtol=rtol, atol=atol)
    yr_close = torch.allclose(Y_r_cpu, Y_r_cuda_cpu, rtol=rtol, atol=atol)
    ys_close = torch.allclose(Y_s_cpu, Y_s_cuda_cpu, rtol=rtol, atol=atol)

    print(f"Comparing Y_q: {'PASS' if yq_close else 'FAIL'}")
    if not yq_close:
        print(f"  Max difference (Y_q): {(Y_q_cpu - Y_q_cuda_cpu).abs().max()}")
        
    print(f"Comparing Y_r: {'PASS' if yr_close else 'FAIL'}")
    if not yr_close:
        print(f"  Max difference (Y_r): {(Y_r_cpu - Y_r_cuda_cpu).abs().max()}")
        
    print(f"Comparing Y_s: {'PASS' if ys_close else 'FAIL'}")
    if not ys_close:
        print(f"  Max difference (Y_s): {(Y_s_cpu - Y_s_cuda_cpu).abs().max()}")

    if yq_close and yr_close and ys_close:
        print("\n*** Gather Equivalence Test Passed! ***")
    else:
        print("\n*** Gather Equivalence Test Failed! ***")
        sys.exit(1)

# --- Test Backward grad_Vq_1 --- 
def run_backward_vq1_test():
    print("\n-------------------------------------")
    print("Testing Backward grad_Vq_1 Equivalence (CPU vs CUDA)")
    print("-------------------------------------")
    
    # --- CPU Execution ---
    print("\nGenerating inputs for backward pass on CPU...")
    Q_cpu, R_cpu, S_cpu, Vq_1_cpu, Vq_2_cpu, Vr_1_cpu, Vr_2_cpu, Vs_1_cpu, Vs_2_cpu, dr_cpu = generate_inputs(device_cpu)
    
    # Generate grad_output matching the expected structure accessed by the kernels
    # The C++ code accesses grad_output[j] and grad_output[k]. 
    # We assume a combined gradient tensor where the 3rd dim fits max(I, J, K).
    # This might need adjustment depending on the exact autograd implementation.
    N_grad = max(I, J, K) # Max sequence length accessed in grad_Vq1
    grad_output_cpu = torch.randn(B, H, N_grad, D, dtype=dtype, device=device_cpu)
    print(f"Generated grad_output_cpu with shape: {grad_output_cpu.shape}")

    # Make tensors require grad for CPU backward computation if needed by the C++ impl (though manual impl might bypass this)
    # Q_cpu.requires_grad_(True)
    # R_cpu.requires_grad_(True)
    # S_cpu.requires_grad_(True)
    # Vq_1_cpu.requires_grad_(True)
    # ... etc for other inputs ...

    print("Running backward pass on CPU...")
    try:
        grads_tuple_cpu = manual_att3ntion.backward(
            grad_output_cpu,
            Q_cpu, R_cpu, S_cpu, 
            Vq_1_cpu, Vq_2_cpu, Vr_1_cpu, Vr_2_cpu, Vs_1_cpu, Vs_2_cpu, 
            dr_cpu
        )
        grad_Vq_1_cpu = grads_tuple_cpu[3] # grad_Vq_1 is the 4th element (index 3)
        print("CPU backward pass completed.")
    except Exception as e:
        print(f"Error during CPU backward pass: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # --- CUDA Execution & Comparison ---
    if not torch.cuda.is_available():
        print("\nCUDA not available. Skipping CUDA backward execution and comparison.")
        return

    print("\nMoving inputs to CUDA device for backward pass...")
    try:
        Q_cuda = Q_cpu.to(device_cuda)
        R_cuda = R_cpu.to(device_cuda)
        S_cuda = S_cpu.to(device_cuda)
        Vq_1_cuda = Vq_1_cpu.to(device_cuda)
        Vq_2_cuda = Vq_2_cpu.to(device_cuda) 
        Vr_1_cuda = Vr_1_cpu.to(device_cuda)
        Vr_2_cuda = Vr_2_cpu.to(device_cuda) 
        Vs_1_cuda = Vs_1_cpu.to(device_cuda)
        Vs_2_cuda = Vs_2_cpu.to(device_cuda) 
        grad_output_cuda = grad_output_cpu.to(device_cuda)
        dr_cuda = dr_cpu 
        print("Inputs moved to CUDA for backward pass.")
    except Exception as e:
        print(f"Error moving inputs to CUDA for backward pass: {e}")
        sys.exit(1)

    print("Running backward pass on CUDA...")
    try:
        grads_tuple_cuda = manual_att3ntion.backward(
            grad_output_cuda,
            Q_cuda, R_cuda, S_cuda, 
            Vq_1_cuda, Vq_2_cuda, Vr_1_cuda, Vr_2_cuda, Vs_1_cuda, Vs_2_cuda, 
            dr_cuda
        )
        grad_Vq_1_cuda = grads_tuple_cuda[3] # grad_Vq_1 is the 4th element (index 3)
        print("CUDA backward pass completed.")
    except Exception as e:
        print(f"Error during CUDA backward pass: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\nComparing CPU and CUDA grad_Vq_1 outputs...")
    # Move CUDA result back to CPU for comparison
    grad_Vq_1_cuda_cpu = grad_Vq_1_cuda.cpu()

    # Check shapes
    if grad_Vq_1_cpu.shape != grad_Vq_1_cuda_cpu.shape:
        print(f"ERROR: grad_Vq_1 shape mismatch! CPU: {grad_Vq_1_cpu.shape}, CUDA: {grad_Vq_1_cuda_cpu.shape}")
        print("Exiting due to shape mismatch.")
        sys.exit(1)
    else:
        print(f"Shapes match: {grad_Vq_1_cpu.shape}")

    # Check numerical equivalence
    vq1_grads_close = torch.allclose(grad_Vq_1_cpu, grad_Vq_1_cuda_cpu, rtol=rtol, atol=atol)

    print(f"Comparing grad_Vq_1: {'PASS' if vq1_grads_close else 'FAIL'}")
    if not vq1_grads_close:
        print(f"  Max difference (grad_Vq_1): {(grad_Vq_1_cpu - grad_Vq_1_cuda_cpu).abs().max()}")
        # Optional: Print specific differing values
        # diff_mask = ~torch.isclose(grad_Vq_1_cpu, grad_Vq_1_cuda_cpu, rtol=rtol, atol=atol)
        # print("CPU vals:", grad_Vq_1_cpu[diff_mask][:10])
        # print("CUDA vals:", grad_Vq_1_cuda_cpu[diff_mask][:10])

    if vq1_grads_close:
        print("\n*** grad_Vq_1 Equivalence Test Passed! ***")
    else:
        print("\n*** grad_Vq_1 Equivalence Test Failed! ***")
        sys.exit(1)

# --- Test Backward grad_Vr_2 --- 
def run_backward_vr2_test():
    print("\n-------------------------------------")
    print("Testing Backward grad_Vr_2 Equivalence (CPU vs CUDA)")
    print("-------------------------------------")
    
    print("\nGenerating inputs for backward pass on CPU...")
    Q_cpu, R_cpu, S_cpu, Vq_1_cpu, Vq_2_cpu, Vr_1_cpu, Vr_2_cpu, Vs_1_cpu, Vs_2_cpu, dr_cpu = generate_inputs(device_cpu)
    
    # grad_Vr_2 depends on grad_output[i] and grad_output[k]
    N_grad = max(I, J, K) 
    grad_output_cpu = torch.randn(B, H, N_grad, D, dtype=dtype, device=device_cpu)
    print(f"Generated grad_output_cpu with shape: {grad_output_cpu.shape}")

    print("Running backward pass on CPU...")
    try:
        grads_tuple_cpu = manual_att3ntion.backward(
            grad_output_cpu, Q_cpu, R_cpu, S_cpu, 
            Vq_1_cpu, Vq_2_cpu, Vr_1_cpu, Vr_2_cpu, Vs_1_cpu, Vs_2_cpu, dr_cpu
        )
        grad_Vr_2_cpu = grads_tuple_cpu[6] # grad_Vr_2 is the 7th element (index 6)
        print("CPU backward pass completed.")
    except Exception as e:
        print(f"Error during CPU backward pass: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if not torch.cuda.is_available():
        print("\nCUDA not available. Skipping CUDA backward execution and comparison.")
        return

    print("\nMoving inputs to CUDA device for backward pass...")
    try:
        Q_cuda, R_cuda, S_cuda = Q_cpu.to(device_cuda), R_cpu.to(device_cuda), S_cpu.to(device_cuda)
        Vq_1_cuda, Vq_2_cuda = Vq_1_cpu.to(device_cuda), Vq_2_cpu.to(device_cuda)
        Vr_1_cuda, Vr_2_cuda = Vr_1_cpu.to(device_cuda), Vr_2_cpu.to(device_cuda)
        Vs_1_cuda, Vs_2_cuda = Vs_1_cpu.to(device_cuda), Vs_2_cpu.to(device_cuda)
        grad_output_cuda = grad_output_cpu.to(device_cuda)
        dr_cuda = dr_cpu 
        print("Inputs moved to CUDA for backward pass.")
    except Exception as e:
        print(f"Error moving inputs to CUDA for backward pass: {e}")
        sys.exit(1)

    print("Running backward pass on CUDA...")
    try:
        grads_tuple_cuda = manual_att3ntion.backward(
            grad_output_cuda, Q_cuda, R_cuda, S_cuda, 
            Vq_1_cuda, Vq_2_cuda, Vr_1_cuda, Vr_2_cuda, Vs_1_cuda, Vs_2_cuda, dr_cuda
        )
        grad_Vr_2_cuda = grads_tuple_cuda[6] # grad_Vr_2 is the 7th element (index 6)
        print("CUDA backward pass completed.")
    except Exception as e:
        print(f"Error during CUDA backward pass: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\nComparing CPU and CUDA grad_Vr_2 outputs...")
    grad_Vr_2_cuda_cpu = grad_Vr_2_cuda.cpu()

    if grad_Vr_2_cpu.shape != grad_Vr_2_cuda_cpu.shape:
        print(f"ERROR: grad_Vr_2 shape mismatch! CPU: {grad_Vr_2_cpu.shape}, CUDA: {grad_Vr_2_cuda_cpu.shape}")
        sys.exit(1)
    else:
        print(f"Shapes match: {grad_Vr_2_cpu.shape}")

    vr2_grads_close = torch.allclose(grad_Vr_2_cpu, grad_Vr_2_cuda_cpu, rtol=rtol, atol=atol)

    print(f"Comparing grad_Vr_2: {'PASS' if vr2_grads_close else 'FAIL'}")
    if not vr2_grads_close:
        print(f"  Max difference (grad_Vr_2): {(grad_Vr_2_cpu - grad_Vr_2_cuda_cpu).abs().max()}")

    if vr2_grads_close:
        print("\n*** grad_Vr_2 Equivalence Test Passed! ***")
    else:
        print("\n*** grad_Vr_2 Equivalence Test Failed! ***")
        sys.exit(1)

# --- Test Backward grad_Vs_2 --- 
def run_backward_vs2_test():
    print("\n-------------------------------------")
    print("Testing Backward grad_Vs_2 Equivalence (CPU vs CUDA)")
    print("-------------------------------------")
    
    print("\nGenerating inputs for backward pass on CPU...")
    Q_cpu, R_cpu, S_cpu, Vq_1_cpu, Vq_2_cpu, Vr_1_cpu, Vr_2_cpu, Vs_1_cpu, Vs_2_cpu, dr_cpu = generate_inputs(device_cpu)
    
    # grad_Vs_2 depends on grad_output[i] and grad_output[j]
    N_grad = max(I, J, K) 
    grad_output_cpu = torch.randn(B, H, N_grad, D, dtype=dtype, device=device_cpu)
    print(f"Generated grad_output_cpu with shape: {grad_output_cpu.shape}")

    print("Running backward pass on CPU...")
    try:
        grads_tuple_cpu = manual_att3ntion.backward(
            grad_output_cpu, Q_cpu, R_cpu, S_cpu, 
            Vq_1_cpu, Vq_2_cpu, Vr_1_cpu, Vr_2_cpu, Vs_1_cpu, Vs_2_cpu, dr_cpu
        )
        grad_Vs_2_cpu = grads_tuple_cpu[8] # grad_Vs_2 is the 9th element (index 8)
        print("CPU backward pass completed.")
    except Exception as e:
        print(f"Error during CPU backward pass: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if not torch.cuda.is_available():
        print("\nCUDA not available. Skipping CUDA backward execution and comparison.")
        return

    print("\nMoving inputs to CUDA device for backward pass...")
    try:
        Q_cuda, R_cuda, S_cuda = Q_cpu.to(device_cuda), R_cpu.to(device_cuda), S_cpu.to(device_cuda)
        Vq_1_cuda, Vq_2_cuda = Vq_1_cpu.to(device_cuda), Vq_2_cpu.to(device_cuda)
        Vr_1_cuda, Vr_2_cuda = Vr_1_cpu.to(device_cuda), Vr_2_cpu.to(device_cuda)
        Vs_1_cuda, Vs_2_cuda = Vs_1_cpu.to(device_cuda), Vs_2_cpu.to(device_cuda)
        grad_output_cuda = grad_output_cpu.to(device_cuda)
        dr_cuda = dr_cpu 
        print("Inputs moved to CUDA for backward pass.")
    except Exception as e:
        print(f"Error moving inputs to CUDA for backward pass: {e}")
        sys.exit(1)

    print("Running backward pass on CUDA...")
    try:
        grads_tuple_cuda = manual_att3ntion.backward(
            grad_output_cuda, Q_cuda, R_cuda, S_cuda, 
            Vq_1_cuda, Vq_2_cuda, Vr_1_cuda, Vr_2_cuda, Vs_1_cuda, Vs_2_cuda, dr_cuda
        )
        grad_Vs_2_cuda = grads_tuple_cuda[8] # grad_Vs_2 is the 9th element (index 8)
        print("CUDA backward pass completed.")
    except Exception as e:
        print(f"Error during CUDA backward pass: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\nComparing CPU and CUDA grad_Vs_2 outputs...")
    grad_Vs_2_cuda_cpu = grad_Vs_2_cuda.cpu()

    if grad_Vs_2_cpu.shape != grad_Vs_2_cuda_cpu.shape:
        print(f"ERROR: grad_Vs_2 shape mismatch! CPU: {grad_Vs_2_cpu.shape}, CUDA: {grad_Vs_2_cuda_cpu.shape}")
        sys.exit(1)
    else:
        print(f"Shapes match: {grad_Vs_2_cpu.shape}")

    vs2_grads_close = torch.allclose(grad_Vs_2_cpu, grad_Vs_2_cuda_cpu, rtol=rtol, atol=atol)

    print(f"Comparing grad_Vs_2: {'PASS' if vs2_grads_close else 'FAIL'}")
    if not vs2_grads_close:
        print(f"  Max difference (grad_Vs_2): {(grad_Vs_2_cpu - grad_Vs_2_cuda_cpu).abs().max()}")

    if vs2_grads_close:
        print("\n*** grad_Vs_2 Equivalence Test Passed! ***")
    else:
        print("\n*** grad_Vs_2 Equivalence Test Failed! ***")
        sys.exit(1)

# --- Test Backward grad_Q --- 
def run_backward_q_test():
    print("\n-------------------------------------")
    print("Testing Backward grad_Q Equivalence (CPU vs CUDA)")
    print("-------------------------------------")
    
    print("\nGenerating inputs for backward pass on CPU...")
    Q_cpu, R_cpu, S_cpu, Vq_1_cpu, Vq_2_cpu, Vr_1_cpu, Vr_2_cpu, Vs_1_cpu, Vs_2_cpu, dr_cpu = generate_inputs(device_cpu)
    
    # grad_Q depends on grad_output[i], grad_output[j], and grad_output[k]
    N_grad = max(I, J, K) 
    grad_output_cpu = torch.randn(B, H, N_grad, D, dtype=dtype, device=device_cpu)
    print(f"Generated grad_output_cpu with shape: {grad_output_cpu.shape}")

    print("Running backward pass on CPU...")
    try:
        grads_tuple_cpu = manual_att3ntion.backward(
            grad_output_cpu, Q_cpu, R_cpu, S_cpu, 
            Vq_1_cpu, Vq_2_cpu, Vr_1_cpu, Vr_2_cpu, Vs_1_cpu, Vs_2_cpu, dr_cpu
        )
        grad_Q_cpu = grads_tuple_cpu[0] # grad_Q is the 1st element (index 0)
        print("CPU backward pass completed.")
    except Exception as e:
        print(f"Error during CPU backward pass: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if not torch.cuda.is_available():
        print("\nCUDA not available. Skipping CUDA backward execution and comparison.")
        return

    print("\nMoving inputs to CUDA device for backward pass...")
    try:
        Q_cuda, R_cuda, S_cuda = Q_cpu.to(device_cuda), R_cpu.to(device_cuda), S_cpu.to(device_cuda)
        Vq_1_cuda, Vq_2_cuda = Vq_1_cpu.to(device_cuda), Vq_2_cpu.to(device_cuda)
        Vr_1_cuda, Vr_2_cuda = Vr_1_cpu.to(device_cuda), Vr_2_cpu.to(device_cuda)
        Vs_1_cuda, Vs_2_cuda = Vs_1_cpu.to(device_cuda), Vs_2_cpu.to(device_cuda)
        grad_output_cuda = grad_output_cpu.to(device_cuda)
        dr_cuda = dr_cpu 
        print("Inputs moved to CUDA for backward pass.")
    except Exception as e:
        print(f"Error moving inputs to CUDA for backward pass: {e}")
        sys.exit(1)

    print("Running backward pass on CUDA...")
    try:
        grads_tuple_cuda = manual_att3ntion.backward(
            grad_output_cuda, Q_cuda, R_cuda, S_cuda, 
            Vq_1_cuda, Vq_2_cuda, Vr_1_cuda, Vr_2_cuda, Vs_1_cuda, Vs_2_cuda, dr_cuda
        )
        grad_Q_cuda = grads_tuple_cuda[0] # grad_Q is the 1st element (index 0)
        print("CUDA backward pass completed.")
    except Exception as e:
        print(f"Error during CUDA backward pass: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\nComparing CPU and CUDA grad_Q outputs...")
    grad_Q_cuda_cpu = grad_Q_cuda.cpu()

    if grad_Q_cpu.shape != grad_Q_cuda_cpu.shape:
        print(f"ERROR: grad_Q shape mismatch! CPU: {grad_Q_cpu.shape}, CUDA: {grad_Q_cuda_cpu.shape}")
        sys.exit(1)
    else:
        print(f"Shapes match: {grad_Q_cpu.shape}")

    q_grads_close = torch.allclose(grad_Q_cpu, grad_Q_cuda_cpu, rtol=rtol, atol=atol)

    print(f"Comparing grad_Q: {'PASS' if q_grads_close else 'FAIL'}")
    if not q_grads_close:
        print(f"  Max difference (grad_Q): {(grad_Q_cpu - grad_Q_cuda_cpu).abs().max()}")

    if q_grads_close:
        print("\n*** grad_Q Equivalence Test Passed! ***")
    else:
        print("\n*** grad_Q Equivalence Test Failed! ***")
        sys.exit(1)


# --- Run the test --- 
if __name__ == '__main__':
    run_test() 
    run_backward_vq1_test() # Add the call to the new test function 
    run_backward_vr2_test()
    run_backward_vs2_test() 
    run_backward_q_test() 