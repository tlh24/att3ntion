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

# Import Extension
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

try:
    import hyper_attn_cpp_manual as manual_att3ntion
    print("Successfully imported C++/CUDA extension.")
except ImportError:
    print("\nError: Failed to import the C++/CUDA extension 'hyper_attn_cpp_manual'.")
    print("Please ensure the extension has been compiled successfully and the name matches setup.py.")
    sys.exit(1)

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
    dropout_rate = 0.0
    return Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2, dropout_rate

def print_table(headers, rows):
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, item in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(item)))
    
    header = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    separator = "-+-".join("-" * w for w in col_widths)
    print(f"| {header} |")
    print(f"|-{separator}-|")
    
    for row in rows:
        row_str = " | ".join(str(item).ljust(col_widths[i]) for i, item in enumerate(row))
        print(f"| {row_str} |")

def run_test():
    print("\nRunning forward pass on CPU...")
    Q_cpu, R_cpu, S_cpu, Vq_1_cpu, Vq_2_cpu, Vr_1_cpu, Vr_2_cpu, Vs_1_cpu, Vs_2_cpu, dr_cpu = generate_inputs(device_cpu)
    
    assert Q_cpu.device.type == 'cpu', "Input tensors not on CPU!"
    
    try:
        Y_tuple_cpu = manual_att3ntion.forward(Q_cpu, R_cpu, S_cpu, Vq_1_cpu, Vq_2_cpu, Vr_1_cpu, Vr_2_cpu, Vs_1_cpu, Vs_2_cpu, dr_cpu)
        Y_q_cpu, Y_r_cpu, Y_s_cpu, Y_q__cpu, Y_r__cpu, Y_s__cpu = Y_tuple_cpu
        print("CPU forward pass completed.")
    except Exception as e:
        print(f"Error during CPU forward pass: {e}")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("\nCUDA not available. Skipping CUDA execution and comparison.")
        return

    print("\nMoving inputs to CUDA device...")
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
        dr_cuda = dr_cpu 
        assert Q_cuda.device.type == 'cuda', "Input tensors not on CUDA!"
        print("Inputs moved to CUDA.")
    except Exception as e:
        print(f"Error moving inputs to CUDA: {e}")
        sys.exit(1)

    print("Running forward pass on CUDA...")
    try:
        Y_tuple_cuda = manual_att3ntion.forward(Q_cuda, R_cuda, S_cuda, Vq_1_cuda, Vq_2_cuda, Vr_1_cuda, Vr_2_cuda, Vs_1_cuda, Vs_2_cuda, dr_cuda)
        Y_q_cuda, Y_r_cuda, Y_s_cuda, Y_q__cuda, Y_r__cuda, Y_s__cuda = Y_tuple_cuda
        print("CUDA forward pass completed.")
    except Exception as e:
        print(f"Error during CUDA forward pass: {e}")
        sys.exit(1)

    print("\nComparing CPU and CUDA outputs...")
    Y_q_cuda_cpu = Y_q_cuda.cpu()
    Y_r_cuda_cpu = Y_r_cuda.cpu()
    Y_s_cuda_cpu = Y_s_cuda.cpu()
    Y_q__cuda_cpu = Y_q__cuda.cpu()
    Y_r__cuda_cpu = Y_r__cuda.cpu()
    Y_s__cuda_cpu = Y_s__cuda.cpu()

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
    if Y_q__cpu.shape != Y_q__cuda_cpu.shape:
        print(f"ERROR: Y_q_ shape mismatch! CPU: {Y_q__cpu.shape}, CUDA: {Y_q__cuda_cpu.shape}")
        shape_match = False
    if Y_r__cpu.shape != Y_r__cuda_cpu.shape:
        print(f"ERROR: Y_r_ shape mismatch! CPU: {Y_r__cpu.shape}, CUDA: {Y_r__cuda_cpu.shape}")
        shape_match = False
    if Y_s__cpu.shape != Y_s__cuda_cpu.shape:
        print(f"ERROR: Y_s_ shape mismatch! CPU: {Y_s__cpu.shape}, CUDA: {Y_s__cuda_cpu.shape}")
        shape_match = False
        
    if not shape_match:
        print("Exiting due to shape mismatches.")
        sys.exit(1)
    else:
        print("Shapes match.")

    # Check numerical equivalence for gather outputs
    yq_close = torch.allclose(Y_q_cpu, Y_q_cuda_cpu, rtol=rtol, atol=atol)
    yr_close = torch.allclose(Y_r_cpu, Y_r_cuda_cpu, rtol=rtol, atol=atol)
    ys_close = torch.allclose(Y_s_cpu, Y_s_cuda_cpu, rtol=rtol, atol=atol)
    
    # Check numerical equivalence for scatter outputs
    yq__close = torch.allclose(Y_q__cpu, Y_q__cuda_cpu, rtol=rtol, atol=atol)
    yr__close = torch.allclose(Y_r__cpu, Y_r__cuda_cpu, rtol=rtol, atol=atol)
    ys__close = torch.allclose(Y_s__cpu, Y_s__cuda_cpu, rtol=rtol, atol=atol)

    forward_results = [
        ["Y_q", "PASS" if yq_close else "FAIL", (Y_q_cpu - Y_q_cuda_cpu).abs().max().item() if not yq_close else 0],
        ["Y_r", "PASS" if yr_close else "FAIL", (Y_r_cpu - Y_r_cuda_cpu).abs().max().item() if not yr_close else 0],
        ["Y_s", "PASS" if ys_close else "FAIL", (Y_s_cpu - Y_s_cuda_cpu).abs().max().item() if not ys_close else 0],
        ["Y_q_", "PASS" if yq__close else "FAIL", (Y_q__cpu - Y_q__cuda_cpu).abs().max().item() if not yq__close else 0],
        ["Y_r_", "PASS" if yr__close else "FAIL", (Y_r__cpu - Y_r__cuda_cpu).abs().max().item() if not yr__close else 0],
        ["Y_s_", "PASS" if ys__close else "FAIL", (Y_s__cpu - Y_s__cuda_cpu).abs().max().item() if not ys__close else 0]
    ]
    
    print("\nForward Pass Results:")
    print_table(["Output", "Status", "Max Diff"], forward_results)

    all_passed = yq_close and yr_close and ys_close and yq__close and yr__close and ys__close
    
    if all_passed:
        print("\n*** Forward Pass Equivalence Test Passed! ***")
    else:
        print("\n*** Forward Pass Equivalence Test Failed! ***")
        sys.exit(1)

def run_backward_test():
    print("\n-------------------------------------")
    print("Testing Backward Equivalence (CPU vs CUDA)")
    print("-------------------------------------")
    
    print("\nGenerating inputs for backward pass on CPU...")
    Q_cpu, R_cpu, S_cpu, Vq_1_cpu, Vq_2_cpu, Vr_1_cpu, Vr_2_cpu, Vs_1_cpu, Vs_2_cpu, dr_cpu = generate_inputs(device_cpu)
    
    N_grad = max(I, J, K)
    grad_output_cpu = torch.randn(B, H, N_grad, D, dtype=dtype, device=device_cpu)
    print(f"Generated grad_output_cpu with shape: {grad_output_cpu.shape}")

    print("Running backward pass on CPU...")
    try:
        grads_tuple_cpu = manual_att3ntion.backward(
            grad_output_cpu,
            Q_cpu, R_cpu, S_cpu, 
            Vq_1_cpu, Vq_2_cpu, Vr_1_cpu, Vr_2_cpu, Vs_1_cpu, Vs_2_cpu, 
            dr_cpu
        )
        
        grad_names = ["grad_Q", "grad_R", "grad_S", "grad_Vq_1", "grad_Vq_2", 
                     "grad_Vr_1", "grad_Vr_2", "grad_Vs_1", "grad_Vs_2"]
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
        print("CUDA backward pass completed.")
    except Exception as e:
        print(f"Error during CUDA backward pass: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\nComparing CPU and CUDA gradient outputs...")
    
    results = []
    all_passed = True
    
    for i, name in enumerate(grad_names):
        try:
            grad_cuda_cpu = grads_tuple_cuda[i].cpu()
            grad_cpu = grads_tuple_cpu[i]
            
            shape_match = grad_cpu.shape == grad_cuda_cpu.shape
            if not shape_match:
                results.append([name, "FAIL", f"Shape mismatch: CPU {grad_cpu.shape}, CUDA {grad_cuda_cpu.shape}"])
                all_passed = False
                continue
                
            is_close = torch.allclose(grad_cpu, grad_cuda_cpu, rtol=rtol, atol=atol)
            max_diff = (grad_cpu - grad_cuda_cpu).abs().max().item() if not is_close else 0
            
            results.append([name, "PASS" if is_close else "FAIL", max_diff])
            if not is_close:
                all_passed = False
        except Exception as e:
            results.append([name, "FAIL", f"Error: {str(e)}"])
            all_passed = False
    
    print("\nBackward Pass Results:")
    print_table(["Gradient", "Status", "Max Diff/Error"], results)
    
    if all_passed:
        print("\n*** Backward Pass Equivalence Test Passed! ***")
    else:
        print("\n*** Backward Pass Equivalence Test Failed! ***")
        sys.exit(1)

if __name__ == '__main__':
    run_test()
    run_backward_test() 