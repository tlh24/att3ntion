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

def compare():
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


if __name__ == "__main__":
    compare()
