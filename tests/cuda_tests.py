import torch
import os
import sys

# --- Configuration ---
B = 2
H = 4
I = 10
J = 12
K = 14
D = 64
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
        Y_q_cpu, Y_r_cpu, Y_s_cpu, Y_q_cpu_, Y_r_cpu_, Y_s_cpu_ = Y_tuple_cpu # Extract all outputs
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
        Y_q_cuda, Y_r_cuda, Y_s_cuda, Y_q_cuda_, Y_r_cuda_, Y_s_cuda_ = Y_tuple_cuda # Extract all outputs
        print("CUDA forward pass completed.")
    except Exception as e:
        print(f"Error during CUDA forward pass: {e}")
        sys.exit(1)

    print("\nComparing CPU and CUDA gather outputs...")
    # Move CUDA results back to CPU for comparison
    Y_q_cuda_cpu = Y_q_cuda.cpu()
    Y_r_cuda_cpu = Y_r_cuda.cpu()
    Y_s_cuda_cpu = Y_s_cuda.cpu()
    Y_q_cuda_cpu_ = Y_q_cuda_.cpu() # Move Y_q_cuda to CPU
    Y_r_cuda_cpu_ = Y_r_cuda_.cpu() # Move Y_r_cuda to CPU
    Y_s_cuda_cpu_ = Y_s_cuda_.cpu() # Move Y_s_cuda to CPU

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
    if Y_q_cpu_.shape != Y_q_cuda_cpu_.shape: # Add shape check for Y_q_
        print(f"ERROR: Y_q_ shape mismatch! CPU: {Y_q_cpu_.shape}, CUDA: {Y_q_cuda_cpu_.shape}")
        shape_match = False
    if Y_r_cpu_.shape != Y_r_cuda_cpu_.shape: # Add shape check for Y_r_
        print(f"ERROR: Y_r_ shape mismatch! CPU: {Y_r_cpu_.shape}, CUDA: {Y_r_cuda_cpu_.shape}")
        shape_match = False
    if Y_s_cpu_.shape != Y_s_cuda_cpu_.shape: # Add shape check for Y_s_
        print(f"ERROR: Y_s_ shape mismatch! CPU: {Y_s_cpu_.shape}, CUDA: {Y_s_cuda_cpu_.shape}")
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
    yq_prime_close = torch.allclose(Y_q_cpu_, Y_q_cuda_cpu_, rtol=rtol, atol=atol) # Add comparison for Y_q_
    yr_prime_close = torch.allclose(Y_r_cpu_, Y_r_cuda_cpu_, rtol=rtol, atol=atol) # Add comparison for Y_r_
    ys_prime_close = torch.allclose(Y_s_cpu_, Y_s_cuda_cpu_, rtol=rtol, atol=atol) # Add comparison for Y_s_

    print(f"Comparing Y_q: {'PASS' if yq_close else 'FAIL'}")
    if not yq_close:
        print(f"  Max difference (Y_q): {(Y_q_cpu - Y_q_cuda_cpu).abs().max()}")
        
    print(f"Comparing Y_r: {'PASS' if yr_close else 'FAIL'}")
    if not yr_close:
        print(f"  Max difference (Y_r): {(Y_r_cpu - Y_r_cuda_cpu).abs().max()}")
        
    print(f"Comparing Y_s: {'PASS' if ys_close else 'FAIL'}")
    if not ys_close:
        print(f"  Max difference (Y_s): {(Y_s_cpu - Y_s_cuda_cpu).abs().max()}")

    print(f"Comparing Y_q': {'PASS' if yq_prime_close else 'FAIL'}") # Add print for Y_q_
    if not yq_prime_close:
        print(f"  Max difference (Y_q'): {(Y_q_cpu_ - Y_q_cuda_cpu_).abs().max()}")

    print(f"Comparing Y_r': {'PASS' if yr_prime_close else 'FAIL'}") # Add print for Y_r_
    if not yr_prime_close:
        print(f"  Max difference (Y_r'): {(Y_r_cpu_ - Y_r_cuda_cpu_).abs().max()}")
        
    print(f"Comparing Y_s': {'PASS' if ys_prime_close else 'FAIL'}") # Add print for Y_s_
    if not ys_prime_close:
        print(f"  Max difference (Y_s'): {(Y_s_cpu_ - Y_s_cuda_cpu_).abs().max()}")

    if yq_close and yr_close and ys_close and yq_prime_close and yr_prime_close and ys_prime_close: # Update overall check
        print("\n*** Gather Equivalence Test Passed! ***")
        print("\n*** Gather & Scatter Equivalence Test Passed! ***")
    else:
        print("\n*** Gather Equivalence Test Failed! ***")
        print("\n*** Scatter Equivalence Test Failed! ***")
        sys.exit(1)

# --- Run the test --- 
if __name__ == '__main__':
    run_test() 