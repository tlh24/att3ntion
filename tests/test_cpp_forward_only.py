import torch
import hyper_attn_cpp_manual
import hyper_attn_cpp_reference
import pytest
import time

def make_inputs(B, H, L, D, seed=42):
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

def compare_forward_configs():
    print("\n===== Comparing Forward Pass Across Multiple Configurations ====")

    configs = [
        # (B, H, L, D)
        (1, 1, 3, 4),    
        (2, 2, 8, 16),   
        (1, 4, 5, 8),    
        (4, 1, 10, 6),   
        (1, 1, 1, 1),    
        (2, 2, 2, 2),    
    ]

    all_passed = True
    total_time_ref = 0
    total_time_opt = 0

    for i, config in enumerate(configs):
        B, H, L, D = config
        seed = 42 + i 
        print(f"\n--- Config {i+1}/{len(configs)}: B={B}, H={H}, L={L}, D={D} ---")

        Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2 = make_inputs(B, H, L, D, seed=seed)

        start_time = time.time()
        try:
            Yq_ref, Yr_ref, Ys_ref, Yq_scatter_ref, Yr_scatter_ref, Ys_scatter_ref = hyper_attn_cpp_reference.forward(Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2)
            Y_ref_sum = Yq_ref + Yr_ref + Ys_ref + Yq_scatter_ref + Yr_scatter_ref + Ys_scatter_ref
        except Exception as e:
            print(f"Reference forward pass failed: {e}")
            all_passed = False
            continue # Skip to next config
        time_ref = time.time() - start_time
        total_time_ref += time_ref

        start_time = time.time()
        try:
            Y_opt_sum = hyper_attn_cpp_manual.forward(Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2)
        except Exception as e:
            print(f"Manual forward pass failed: {e}")
            all_passed = False
            continue 
        time_opt = time.time() - start_time
        total_time_opt += time_opt

        close = torch.allclose(Y_ref_sum, Y_opt_sum, atol=1e-4, rtol=1e-4)
        diff = (Y_ref_sum - Y_opt_sum).abs().max().item()
        status = '✅ passed' if close else '❌ FAILED'
        print(f"Result: {status} | Max diff: {diff:.6f} | Time Ref: {time_ref*1000:.2f}ms, Opt: {time_opt*1000:.2f}ms")

        if not close: 
            all_passed = False
            print(f"  Summed Reference:\n{Y_ref_sum}")
            print(f"  Summed Optimized:\n{Y_opt_sum}")

    print("\n===== Forward Pass Comparison Summary ====")
    if all_passed:
        print("✅ All configurations passed!")
    else:
        print("❌ Some configurations failed.")

    avg_time_ref = (total_time_ref / len(configs)) * 1000 if configs else 0
    avg_time_opt = (total_time_opt / len(configs)) * 1000 if configs else 0
    print(f"Average Time Reference: {avg_time_ref:.2f}ms")
    print(f"Average Time Optimized: {avg_time_opt:.2f}ms")

    return all_passed

if __name__ == "__main__":
    passed = compare_forward_configs()
