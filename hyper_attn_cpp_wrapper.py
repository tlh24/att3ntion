import torch
import torch.nn as nn
import math
import hyper_attn_cpp_manual
from torch.autograd import Function

class HyperAttentionAutograd(Function):
    """
    Bridge between PyTorch autograd and the manual C++ forward/backward passes.
    """

    @staticmethod
    def forward(ctx, Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2, dropout_rate=0.0):
        """
        Calls the C++ forward pass and saves necessary tensors for backward.
        Args:
            ctx: Context object to save tensors.
            Q, R, S, V*_*: Input tensors for the attention mechanism.
            dropout_rate: Dropout rate (passed to C++ if needed, but not differentiable).
        Returns:
            The output tensor from the C++ forward pass.
        """
        Q = Q.contiguous()
        R = R.contiguous()
        S = S.contiguous()
        Vq_1 = Vq_1.contiguous()
        Vq_2 = Vq_2.contiguous()
        Vr_1 = Vr_1.contiguous()
        Vr_2 = Vr_2.contiguous()
        Vs_1 = Vs_1.contiguous()
        Vs_2 = Vs_2.contiguous()

        print("--- Preparing to call C++ forward ---", flush=True)
        print(f"Q: {Q.shape}, {Q.dtype}, {Q.device}", flush=True)
        print(f"R: {R.shape}, {R.dtype}, {R.device}", flush=True)
        print(f"S: {S.shape}, {S.dtype}, {S.device}", flush=True)
        print(f"Vq_1: {Vq_1.shape}, {Vq_1.dtype}, {Vq_1.device}", flush=True)
        print(f"Vq_2: {Vq_2.shape}, {Vq_2.dtype}, {Vq_2.device}", flush=True)
        print(f"Vr_1: {Vr_1.shape}, {Vr_1.dtype}, {Vr_1.device}", flush=True)
        print(f"Vr_2: {Vr_2.shape}, {Vr_2.dtype}, {Vr_2.device}", flush=True)
        print(f"Vs_1: {Vs_1.shape}, {Vs_1.dtype}, {Vs_1.device}", flush=True)
        print(f"Vs_2: {Vs_2.shape}, {Vs_2.dtype}, {Vs_2.device}", flush=True)
        print(f"dropout_rate: {dropout_rate}", flush=True)
        print("--------------------------------------", flush=True)

        output = hyper_attn_cpp_manual.forward(
            Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2, dropout_rate
        )

        print(f"--- C++ forward returned. Checking output... ---", flush=True)
        try:
            print(f"Output shape: {output.shape}", flush=True)
            print(f"Output dtype: {output.dtype}", flush=True)
            print(f"Output device: {output.device}", flush=True)
            print("--- Output check seems OK --- ", flush=True)
        except Exception as e:
            print(f"!!! Error accessing C++ output tensor: {e}", flush=True)
            # Force exit or raise to ensure crash info is related to this point
            import sys
            sys.exit(1) 

        print("--- Preparing to save for backward --- ", flush=True)
        ctx.save_for_backward(Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2)
        print("--- Saved for backward --- ", flush=True)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Calls the C++ backward pass using saved tensors and the incoming gradient.
        Args:
            ctx: Context object with saved tensors.
            grad_output: Gradient of the loss w.r.t. the output of the forward pass.
        Returns:
            A tuple of gradients corresponding *exactly* to the inputs of the
            forward function (Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2, dropout_rate).
        """
        grad_output = grad_output.contiguous()

        Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2 = ctx.saved_tensors

        grad_tuple = hyper_attn_cpp_manual.backward(
            grad_output, Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2 #,dropout_rate
        )

        grad_Q, grad_R, grad_S, grad_Vq_1, grad_Vq_2, grad_Vr_1, grad_Vr_2, grad_Vs_1, grad_Vs_2 = grad_tuple

        return (
            grad_Q,
            grad_R,
            grad_S,
            grad_Vq_1,
            grad_Vq_2,
            grad_Vr_1,
            grad_Vr_2,
            grad_Vs_1,
            grad_Vs_2,
        )

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
        
        self.Wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.Wr = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.Ws = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        
        self.Wv_q = nn.Linear(d_model, n_heads * self.head_dim * 2, bias=True)
        self.Wv_r = nn.Linear(d_model, n_heads * self.head_dim * 2, bias=True)
        self.Wv_s = nn.Linear(d_model, n_heads * self.head_dim * 2, bias=True)
        
        self.Wo = nn.Linear(n_heads * self.head_dim, d_model, bias=True)
       
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
        
        #core 3-way attention
        y = HyperAttentionAutograd.apply( 
            Q, R, S,
            Vq_1, Vq_2,
            Vr_1, Vr_2,
            Vs_1, Vs_2,
        )
        
        y = y.permute(0, 2, 1, 3).contiguous().view(batch_size, ntok, self.n_heads * self.head_dim) 
        #The previous reshape assumed summing over heads, let's adjust to typical attention output handling
        # y = y.permute(0, 2, 1, 3).sum(dim=2).squeeze() # is this correct? doublecheck

        y = self.gelu(y) 
        y = self.Wo(y) 

        return y 