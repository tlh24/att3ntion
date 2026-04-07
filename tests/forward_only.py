import math
import torch
import os
import sys
import time
from pynvml import *
import pdb
import matplotlib.pyplot as plt

# Import extensions (assuming they are compiled and accessible)
# The order of imports matters for shared library loading, ensure pytorch-related ones are first if there are conflicts.
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

try:
    import hyper_attn_cpp_manual as manual_att3ntion
    import hyper_attn_cpp_reference
    print("Successfully imported C++/CUDA extensions.")
except ImportError:
    print("\nError: Failed to import the C++/CUDA extension 'hyper_attn_cpp_manual' or 'hyper_attn_cpp_reference'.")
    print("Please ensure the extensions have been compiled successfully and the names match setup.py.")
    sys.exit(1)


# --- Configuration ---
B = 1
H = 1
I = 16
J = 16
K = 16
D = 32
dtype = torch.float32
device_cpu = torch.device("cpu")
device_cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rtol = 1e-4
atol = 1e-5

print(f"Testing Scatter Equivalence (CPU vs CUDA)")
print(f"Parameters: B={B}, H={H}, I={I}, J={J}, K={K}, D={D}")
print(f"CPU Device: {device_cpu}")
print(f"CUDA Device: {device_cuda}")

# ---> Verify LD_LIBRARY_PATH before import <---
print(f"LD_LIBRARY_PATH before import: {os.environ.get('LD_LIBRARY_PATH')}")


def generate_inputs(device):
	"""Generates random input tensors on the specified device."""
	Q = torch.randn(B, H, I, D, dtype=dtype, device=device) * 1
	R = torch.randn(B, H, J, D, dtype=dtype, device=device) * 1
	S = torch.randn(B, H, K, D, dtype=dtype, device=device) * 1
	Vq_1 = torch.randn(B, H, I, D, dtype=dtype, device=device)
	Vq_2 = torch.randn(B, H, I, D, dtype=dtype, device=device)
	Vr_1 = torch.randn(B, H, J, D, dtype=dtype, device=device)
	Vr_2 = torch.randn(B, H, J, D, dtype=dtype, device=device)
	Vs_1 = torch.randn(B, H, K, D, dtype=dtype, device=device)
	Vs_2 = torch.randn(B, H, K, D, dtype=dtype, device=device)
	if False:
		Q = torch.zeros(B, H, I, D, dtype=dtype, device=device)
		R = torch.zeros(B, H, J, D, dtype=dtype, device=device)
		S = torch.zeros(B, H, K, D, dtype=dtype, device=device)
		# k = 9
		# Q[0,0,:k,:k] = torch.arange(-5, -5 + k*k).reshape(k, k)
		# R[0,0,:k,:k] = torch.arange(-10, -10 + k*k).reshape(k, k)
		# S[0,0,:k,:k] = torch.arange(5, 5 + k*k).reshape(k, k)
		of = 0
		tok = 0
		l = 4
		Q[0,0,tok,of:of+l] = torch.arange(0, l) * 4
		R[0,0,tok,of:of+l] = torch.arange(0, l) * 4
		S[0,0,tok,of:of+l] = torch.arange(0, l) * 4
	if False:
		Vr_2 = torch.ones(B, H, J, D, dtype=dtype, device=device)
		Vs_2 = torch.ones(B, H, J, D, dtype=dtype, device=device) * 2
		# k = 9
		# Vr_2[0,0,:k,:k] = torch.arange(-5, -5 + k*k).reshape(k, k)
		# Vs_2[0,0,:k,:k] = torch.arange(-5, -5 + k*k).reshape(k, k)
		# Vr_2[0,0,-4:,:] = torch.randn(4, D) * 100.0
		# Vs_2[0,0,-4:,:] = torch.randn(4, D) * 100.0
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

def python_att3ntion(Q, R, S, Vq, Vq_, Vr, Vr_, Vs, Vs_, dr):
	# sanity check - reference python implementation!
	d = Q.shape[-1]
	# compute 3-way attention scores of shape [b, h, i, j, k]
	dot_product = torch.einsum('bhid,bhjd,bhkd->bhijk', Q, R, S)
	dot_product = dot_product / (math.sqrt(d))

	# Compute attention weights for each position
	# Aq - gathering to position i (softmax over j,k)
	dot_product_q = dot_product
	Aq = torch.softmax(dot_product_q.flatten(3, 4), dim=-1).reshape(dot_product.shape)

	# Ar - gathering to position j (softmax over i,k)
	dot_product_r = dot_product.permute(0, 1, 3, 2, 4) # [b, h, j, i, k]
	Ar = torch.softmax(dot_product_r.flatten(3, 4), dim=-1).reshape(dot_product.shape)
	Ar = Ar.permute(0, 1, 3, 2, 4)

	# As - gathering to position k (softmax over i,j)
	dot_product_s = dot_product.permute(0, 1, 4, 2, 3)  # [b, h, k, i, j]
	As = torch.softmax(dot_product_s.flatten(3, 4), dim=-1).reshape(dot_product.shape)
	As = As.permute(0, 1, 3, 4, 2)

	# gather
	Y_q = torch.einsum('bhijk,bhjd,bhkd->bhid', Aq, Vr, Vs)
	Y_r = torch.einsum('bhijk,bhid,bhkd->bhjd', Ar, Vq, Vs)
	Y_s = torch.einsum('bhijk,bhid,bhjd->bhkd', As, Vq, Vr)

	# scatter
	# 'diamond' op is mul
	Y_q_ = torch.einsum('bhijk,bhjd,bhijk,bhkd->bhid', Ar, Vr_, As, Vs_)
	Y_r_ = torch.einsum('bhijk,bhid,bhijk,bhkd->bhjd', Aq, Vq_, As, Vs_)
	Y_s_ = torch.einsum('bhijk,bhid,bhijk,bhjd->bhkd', Aq, Vq_, Ar, Vr_)

	return (Y_q, Y_r, Y_s, Y_q_, Y_r_, Y_s_)


def run_test():
	print("\nRunning forward pass on CPU...")
	Q_cpu, R_cpu, S_cpu, Vq_1_cpu, Vq_2_cpu, Vr_1_cpu, Vr_2_cpu, Vs_1_cpu, Vs_2_cpu, dr_cpu = generate_inputs(device_cpu)

	assert Q_cpu.device.type == 'cpu', "Input tensors not on CPU!"

	Y_tuple_py = python_att3ntion(Q_cpu, R_cpu, S_cpu, Vq_1_cpu, Vq_2_cpu, Vr_1_cpu, Vr_2_cpu, Vs_1_cpu, Vs_2_cpu, dr_cpu)
	Y_q_py, Y_r_py, Y_s_py, Y_q__py, Y_r__py, Y_s__py = Y_tuple_py

	try:
		Y_tuple_cpu = manual_att3ntion.forward(Q_cpu, R_cpu, S_cpu, Vq_1_cpu, Vq_2_cpu, Vr_1_cpu, Vr_2_cpu, Vs_1_cpu, Vs_2_cpu, dr_cpu)
		Y_q_cpu, Y_r_cpu, Y_s_cpu, Y_q__cpu, Y_r__cpu, Y_s__cpu = Y_tuple_cpu
		print("CPU forward pass completed.")
	except Exception as e:
		print(f"Error during CPU forward pass: {e}")
		sys.exit(1)

	# check python vs cpp
	yq_close = torch.allclose(Y_q_cpu, Y_q_py, rtol=rtol, atol=atol)
	yr_close = torch.allclose(Y_r_cpu, Y_r_py, rtol=rtol, atol=atol)
	ys_close = torch.allclose(Y_s_cpu, Y_s_py, rtol=rtol, atol=atol)
	yq__close = torch.allclose(Y_q__cpu, Y_q__py, rtol=rtol, atol=atol)
	yr__close = torch.allclose(Y_r__cpu, Y_r__py, rtol=rtol, atol=atol)
	ys__close = torch.allclose(Y_s__cpu, Y_s__py, rtol=rtol, atol=atol)

	py_vs_cpp_results = [
		["Y_q", "PASS" if yq_close else "FAIL", (Y_q_cpu - Y_q_py).abs().max().item() if not yq__close else 0],
		["Y_r", "PASS" if yr_close else "FAIL", (Y_r_cpu - Y_r_py).abs().max().item() if not yr__close else 0],
		["Y_s", "PASS" if ys_close else "FAIL", (Y_s_cpu - Y_s_py).abs().max().item() if not ys__close else 0],
		["Y_q_", "PASS" if yq__close else "FAIL", (Y_q__cpu - Y_q__py).abs().max().item() if not yq__close else 0],
		["Y_r_", "PASS" if yr__close else "FAIL", (Y_r__cpu - Y_r__py).abs().max().item() if not yr__close else 0],
		["Y_s_", "PASS" if ys__close else "FAIL", (Y_s__cpu - Y_s__py).abs().max().item() if not ys__close else 0]
	]
	print("\nCpp vs Python Pass Results:")
	print_table(["Output", "Status", "Max Diff"], py_vs_cpp_results)

	if not torch.cuda.is_available():
		print("\nCUDA not available. Skipping CUDA execution and comparison.")
		return True

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
	x = Y_q__cuda.cpu() - Y_q__cpu
	x = x.squeeze()
	fig,axs = plt.subplots(1, 4)
	im = axs[0].imshow(Y_q__cpu.squeeze())
	plt.colorbar(im, ax=axs[0])
	axs[0].set_title('CPU')
	im = axs[1].imshow(Y_q__cuda.cpu().squeeze())
	plt.colorbar(im, ax=axs[1])
	axs[1].set_title('CUDA')
	im = axs[2].imshow(x)
	plt.colorbar(im, ax=axs[2])
	axs[2].set_title('Diff')
	x = Y_q__cuda.cpu() / Y_q__cpu
	x = x.squeeze()
	im = axs[3].imshow(x)
	plt.colorbar(im, ax=axs[3])
	axs[3].set_title('Ratio')
	plt.show()

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
		return False
	else:
		print("Shapes match.")

	# Check numerical equivalence for gather outputs - SKIPPED
	yq_close = True
	yr_close = True
	ys_close = True

	# Check numerical equivalence for scatter outputs
	yq__close = torch.allclose(Y_q__cpu, Y_q__cuda_cpu, rtol=rtol, atol=atol)
	yr__close = True #torch.allclose(Y_r__cpu, Y_r__cuda_cpu, rtol=rtol, atol=atol)
	ys__close = True #torch.allclose(Y_s__cpu, Y_s__cuda_cpu, rtol=rtol, atol=atol)

	forward_results = [
		["Y_q", "SKIPPED", "N/A"],
		["Y_r", "SKIPPED", "N/A"],
		["Y_s", "SKIPPED", "N/A"],
		["Y_q_", "PASS" if yq__close else "FAIL", (Y_q__cpu - Y_q__cuda_cpu).abs().max().item() if not yq__close else 0],
		["Y_r_", "SKIPPED", "N/A"], # "PASS" if yr__close else "FAIL", (Y_r__cpu - Y_r__cuda_cpu).abs().max().item() if not yr__close else 0],
		["Y_s_", "SKIPPED", "N/A"] # "PASS" if ys__close else "FAIL", (Y_s__cpu - Y_s__cuda_cpu).abs().max().item() if not ys__close else 0]
	]

	all_passed = yq__close #and yr__close and ys__close

	if not all_passed: # Only print detailed results if some tests failed
		print("\nForward Pass Results:")
		print_table(["Output", "Status", "Max Diff"], forward_results)
		print("\n*** Scatter Equivalence Test Failed! ***")
		return False
	else:
		print("\nScatter-only Equivalence Test Passed.")
		return True

configs = [ # dims: b, h, i, j, k, d
    # (1, 1, 4, 4, 4, 64),
    # (1, 1, 6, 6, 6, 64),
    # (1, 1, 8, 8, 8, 64),
    # (1, 1, 12, 12, 12, 64),
    (1, 1, 16, 16, 16, 64),  
    # (1, 1, 24, 24, 24, 64),
    (1, 1, 32, 32, 32, 64),  
    (1, 1, 48, 48, 48, 64),
    (1, 1, 64, 64, 64, 64),
    (1, 1, 96, 96, 96, 64),
    (1, 1, 128, 128, 128, 64),
    (1, 1, 256, 256, 256, 64),
    (1, 1, 512, 512, 512, 64),
    (1, 2, 1024, 1024, 1024, 32),

]

def benchmark():
    dropout_rate = 0.0

    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARKS (FORWARD PASS, YQ SCATTER ONLY)")
    print("=" * 80)

    print("\n--- Custom CUDA & PyTorch C++ Reference Benchmarks ---")
    header_custom = (f"{'Seq Len':<10} | "
                     f"{'CUDA ms':<12} | {'Torch ms':<12} | "
                     f"{'CUDA TFLOP/s':<15} | {'Torch TFLOP/s':<15} | "
                     f"{'CUDA Peak MB':<12} | {'Torch Peak MB':<12}")
    print(header_custom)
    print("-" * len(header_custom))

    for B, H, I_dim, J_dim, K_dim, D_dim in configs:
        flops = (B * H) * (4 * I_dim * J_dim * K_dim * D_dim + 3 * J_dim * K_dim * D_dim)
        try:
            Q = torch.rand(B, H, I_dim, D_dim, device='cuda', dtype=torch.float32)
            R = torch.rand(B, H, J_dim, D_dim, device='cuda', dtype=torch.float32)
            S = torch.rand(B, H, K_dim, D_dim, device='cuda', dtype=torch.float32)
            Vq_1 = torch.rand(B, H, I_dim, D_dim, device='cuda', dtype=torch.float32)
            Vq_2 = torch.rand(B, H, I_dim, D_dim, device='cuda', dtype=torch.float32)
            Vr_1 = torch.rand(B, H, J_dim, D_dim, device='cuda', dtype=torch.float32)
            Vr_2 = torch.rand(B, H, J_dim, D_dim, device='cuda', dtype=torch.float32)
            Vs_1 = torch.rand(B, H, K_dim, D_dim, device='cuda', dtype=torch.float32)
            Vs_2 = torch.rand(B, H, K_dim, D_dim, device='cuda', dtype=torch.float32)
        except Exception as e:
            print(f"{I_dim:<10} | Error initializing tensors: {e}")
            continue

        peak_mem_manual_cuda_mb = 0.0
        manual_vram_used_mb = 0.0
        peak_mem_pytorch_ref_mb = 0.0
        pytorch_ref_vram_used_mb = 0.0

        # --- Manual CUDA Benchmark ---
        try:
            torch.cuda.reset_peak_memory_stats()
            handle = nvmlDeviceGetHandleByIndex(0)
            pre_used = nvmlDeviceGetMemoryInfo(handle).used
            
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            # Only run and measure for Yq_scatter
            Y_q_mc, _, _, Y_q__mc, _, _ = manual_att3ntion.forward(
                Q.clone(), R.clone(), S.clone(), Vq_1.clone(), Vq_2.clone(),
                Vr_1.clone(), Vr_2.clone(), Vs_1.clone(), Vs_2.clone(), dropout_rate)
            
            torch.cuda.synchronize()
            total_time_manual_cuda = time.perf_counter() - start_time
            post_used = nvmlDeviceGetMemoryInfo(handle).used
            manual_vram_used_mb = (post_used - pre_used) / (1024 * 1024)
            peak_mem_manual_cuda_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        except torch.cuda.OutOfMemoryError:
            total_time_manual_cuda = float('inf')
        except Exception as e:
            total_time_manual_cuda = float('nan')
            print(f"Error in manual attn: {e}")

        # --- PyTorch C++ Reference Benchmark ---
        try:
            Q_ref, R_ref, S_ref = Q.clone(), R.clone(), S.clone()
            Vq_1_ref, Vq_2_ref = Vq_1.clone(), Vq_2.clone()
            Vr_1_ref, Vr_2_ref = Vr_1.clone(), Vr_2.clone()
            Vs_1_ref, Vs_2_ref = Vs_1.clone(), Vs_2.clone()

            torch.cuda.reset_peak_memory_stats()
            handle = nvmlDeviceGetHandleByIndex(0)
            pre_used = nvmlDeviceGetMemoryInfo(handle).used

            torch.cuda.synchronize()
            start_time = time.perf_counter()

            # Only run and measure for Yq_scatter
            Y_q_ref, _, _, Y_q__ref, _, _ = hyper_attn_cpp_reference.forward(
                Q_ref, R_ref, S_ref, Vq_1_ref, Vq_2_ref, Vr_1_ref, Vr_2_ref, Vs_1_ref, Vs_2_ref, dropout_rate)
            
            torch.cuda.synchronize()
            total_time_pytorch_ref = time.perf_counter() - start_time
            post_used = nvmlDeviceGetMemoryInfo(handle).used
            pytorch_ref_vram_used_mb = (post_used - pre_used) / (1024 * 1024)
            peak_mem_pytorch_ref_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        except torch.cuda.OutOfMemoryError:
            total_time_pytorch_ref = float('inf')
        except Exception as e:
            total_time_pytorch_ref = float('nan')
            print(f"Error in torch ref: {e}")

        cuda_tflops, torch_tflops = 0.0, 0.0
        if total_time_manual_cuda > 0 and not (total_time_manual_cuda == float('inf') or total_time_manual_cuda == float('nan')):
            cuda_tflops = (flops / total_time_manual_cuda) / 1e12
        if total_time_pytorch_ref > 0 and not (total_time_pytorch_ref == float('inf') or total_time_pytorch_ref == float('nan')):
            torch_tflops = (flops / total_time_pytorch_ref) / 1e12

        cuda_time_str = f"{total_time_manual_cuda * 1000:<12.4f}"
        cuda_mem_str = f"{peak_mem_manual_cuda_mb:<12.2f}"
        cuda_tflops_str = f"{cuda_tflops:<15.4f}"
        if total_time_manual_cuda == float('inf'):
            cuda_time_str = f"{'OOM':<12}"
            cuda_mem_str = f"{'N/A':<12}"
            cuda_tflops_str = f"{'N/A':<15}"
        elif total_time_manual_cuda == float('nan'):
            cuda_time_str = f"{'Error':<12}"
            cuda_mem_str = f"{'N/A':<12}"
            cuda_tflops_str = f"{'N/A':<15}"

        torch_time_str = f"{total_time_pytorch_ref * 1000:<12.4f}"
        torch_mem_str = f"{peak_mem_pytorch_ref_mb:<12.2f}"
        torch_tflops_str = f"{torch_tflops:<15.4f}"
        if total_time_pytorch_ref == float('inf'):
            torch_time_str = f"{'OOM':<12}"
            torch_mem_str = f"{'N/A':<12}"
            torch_tflops_str = f"{'N/A':<15}"
        elif total_time_pytorch_ref == float('nan'):
            torch_time_str = f"{'Error':<12}"
            torch_mem_str = f"{'N/A':<12}"
            torch_tflops_str = f"{'N/A':<15}"
        
        print(f"{I_dim:<10} | {cuda_time_str} | {torch_time_str} | {cuda_tflops_str} | {torch_tflops_str} | {cuda_mem_str} | {torch_mem_str}")

    print("-" * len(header_custom))


if __name__ == '__main__':
    nvmlInit()
    
    print("=" * 60)
    print("CUDA EQUIVALENCE TESTING (Yq_SCATTER FORWARD ONLY)")
    print("=" * 60)
    
    forward_passed = run_test()
    
    if forward_passed:
        print("\n" + "=" * 60)
        print("ALL Yq_SCATTER EQUIVALENCE TESTS PASSED! Proceeding with benchmarks...")
        print("=" * 60)
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            benchmark()
        else:
            print("CUDA not available. Skipping benchmarks.")
    else:
        print("\n" + "=" * 60)
        print("Yq_SCATTER EQUIVALENCE TESTS FAILED - SKIPPING BENCHMARKS")
        print("Please fix the implementation issues before benchmarking.")
        print("=" * 60)
        sys.exit(1)
    
    nvmlShutdown() 
