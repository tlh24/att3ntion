import torch
import hyper_attn_cpp_manual

# Set up test case with specific dimensions that include our target indices
B, H, I, J, K, D = 1, 1, 3, 4, 3, 5

# Create consistent random data for testing
torch.manual_seed(42)
Q = torch.randn(B, H, I, D, requires_grad=True)
R = torch.randn(B, H, J, D, requires_grad=True)
S = torch.randn(B, H, K, D, requires_grad=True)
Vq_1 = torch.randn(B, H, I, D, requires_grad=True)
Vq_2 = torch.randn(B, H, I, D, requires_grad=True)
Vr_1 = torch.randn(B, H, J, D, requires_grad=True)
Vr_2 = torch.randn(B, H, J, D, requires_grad=True)
Vs_1 = torch.randn(B, H, K, D, requires_grad=True)
Vs_2 = torch.randn(B, H, K, D, requires_grad=True)

# Create identical copies for CPU and CUDA
Q_cpu = Q.clone().detach().requires_grad_(True)
R_cpu = R.clone().detach().requires_grad_(True)
S_cpu = S.clone().detach().requires_grad_(True)
Vq_1_cpu = Vq_1.clone().detach().requires_grad_(True)
Vq_2_cpu = Vq_2.clone().detach().requires_grad_(True)
Vr_1_cpu = Vr_1.clone().detach().requires_grad_(True)
Vr_2_cpu = Vr_2.clone().detach().requires_grad_(True)
Vs_1_cpu = Vs_1.clone().detach().requires_grad_(True)
Vs_2_cpu = Vs_2.clone().detach().requires_grad_(True)

Q_cuda = Q.clone().detach().cuda().requires_grad_(True)
R_cuda = R.clone().detach().cuda().requires_grad_(True)
S_cuda = S.clone().detach().cuda().requires_grad_(True)
Vq_1_cuda = Vq_1.clone().detach().cuda().requires_grad_(True)
Vq_2_cuda = Vq_2.clone().detach().cuda().requires_grad_(True)
Vr_1_cuda = Vr_1.clone().detach().cuda().requires_grad_(True)
Vr_2_cuda = Vr_2.clone().detach().cuda().requires_grad_(True)
Vs_1_cuda = Vs_1.clone().detach().cuda().requires_grad_(True)
Vs_2_cuda = Vs_2.clone().detach().cuda().requires_grad_(True)

# Forward pass CPU
print("Running CPU forward pass...")
Y_q_cpu, Y_r_cpu, Y_s_cpu, Y_q_cpu_, Y_r_cpu_, Y_s_cpu_ = hyper_attn_cpp_manual.forward(
    Q_cpu, R_cpu, S_cpu,
    Vq_1_cpu, Vq_2_cpu,
    Vr_1_cpu, Vr_2_cpu,
    Vs_1_cpu, Vs_2_cpu,
    0.0
)

# Forward pass CUDA
print("Running CUDA forward pass...")
Y_q_cuda, Y_r_cuda, Y_s_cuda, Y_q_cuda_, Y_r_cuda_, Y_s_cuda_ = hyper_attn_cpp_manual.forward(
    Q_cuda, R_cuda, S_cuda,
    Vq_1_cuda, Vq_2_cuda,
    Vr_1_cuda, Vr_2_cuda,
    Vs_1_cuda, Vs_2_cuda,
    0.0
)

# Print shapes to debug
print("\nOutput tensor shapes:")
print(f"CPU: Y_q: {Y_q_cpu.shape}, Y_r: {Y_r_cpu.shape}, Y_s: {Y_s_cpu.shape}")
print(f"CPU: Y_q_: {Y_q_cpu_.shape}, Y_r_: {Y_r_cpu_.shape}, Y_s_: {Y_s_cpu_.shape}")
print(f"CUDA: Y_q: {Y_q_cuda.shape}, Y_r: {Y_r_cuda.shape}, Y_s: {Y_s_cuda.shape}")
print(f"CUDA: Y_q_: {Y_q_cuda_.shape}, Y_r_: {Y_r_cuda_.shape}, Y_s_: {Y_s_cuda_.shape}")

# Create gradient for backward pass - use ones with appropriate shapes for each output
print("\nCreating gradient tensors...")
grad_output_cpu = torch.ones_like(Y_q_cpu)  # Just use Y_q dimensions for simplicity
grad_output_cuda = torch.ones_like(Y_q_cuda).cuda()

# Backward pass CPU
print("\nRunning CPU backward pass (with debug output)...")
grad_Q_cpu, grad_R_cpu, grad_S_cpu, grad_Vq1_cpu, grad_Vq2_cpu, grad_Vr1_cpu, grad_Vr2_cpu, grad_Vs1_cpu, grad_Vs2_cpu = hyper_attn_cpp_manual.backward(
    grad_output_cpu, 
    Q_cpu, R_cpu, S_cpu,
    Vq_1_cpu, Vq_2_cpu,
    Vr_1_cpu, Vr_2_cpu,
    Vs_1_cpu, Vs_2_cpu,
    0.0
)



# Backward pass CUDA
print("\nRunning CUDA backward pass (with debug output)...")
grad_Q_cuda, grad_R_cuda, grad_S_cuda, grad_Vq1_cuda, grad_Vq2_cuda, grad_Vr1_cuda, grad_Vr2_cuda, grad_Vs1_cuda, grad_Vs2_cuda = hyper_attn_cpp_manual.backward(
    grad_output_cuda, 
    Q_cuda, R_cuda, S_cuda,
    Vq_1_cuda, Vq_2_cuda,
    Vr_1_cuda, Vr_2_cuda,
    Vs_1_cuda, Vs_2_cuda,
    0.0
)

# Extract the specific value we're debugging: b=0,h=0,i=1,j=2,k=1,d=3
print("\nComparing specific index values:")
print(f"CPU gradQ[0,0,1,3]: {grad_Q_cpu[0,0,1,3].item()}")
print(f"CUDA gradQ[0,0,1,3]: {grad_Q_cuda[0,0,1,3].item()}")
print(f"Ratio: {grad_Q_cpu[0,0,1,3].item() / grad_Q_cuda[0,0,1,3].item() if grad_Q_cuda[0,0,1,3].item() != 0 else 'N/A'}")
print(f"Absolute difference: {abs(grad_Q_cpu[0,0,1,3].item() - grad_Q_cuda[0,0,1,3].item())}")

# Print a few more indices for context
print("\nChecking a few more gradient values:")
for i in range(I):
    for d in range(D):
        if i == 1 and d == 3:
            continue  # Already printed above
        print(f"CPU gradQ[0,0,{i},{d}]: {grad_Q_cpu[0,0,i,d].item():.8f}, CUDA: {grad_Q_cuda[0,0,i,d].item():.8f}, Diff: {abs(grad_Q_cpu[0,0,i,d].item() - grad_Q_cuda[0,0,i,d].item()):.8f}") 