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
            The output tensor from the C++ forward pass (summed).
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

        outputs_tuple = hyper_attn_cpp_manual.forward(
            Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2, dropout_rate
        )

        # Sum the tuple of tensors from the C++ backend to return a single tensor
        if not isinstance(outputs_tuple, torch.Tensor):
            if isinstance(outputs_tuple, tuple) and all(isinstance(t, torch.Tensor) for t in outputs_tuple):
                 final_output = sum(outputs_tuple)
            else:
                raise TypeError(f"C++ forward expected to return a tuple of Tensors or a single Tensor, but got {type(outputs_tuple)}")
        else: # It's already a single tensor
            final_output = outputs_tuple

        ctx.save_for_backward(Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2)
        
        return final_output

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

        # Ensure grad_tuple from C++ has the correct number of elements (9 for the 9 tensor inputs)
        if not (isinstance(grad_tuple, tuple) and len(grad_tuple) == 9):
            raise ValueError(f"C++ backward expected to return a tuple of 9 gradients, got {len(grad_tuple) if isinstance(grad_tuple, tuple) else type(grad_tuple)}")

        grad_Q, grad_R, grad_S, grad_Vq_1, grad_Vq_2, grad_Vr_1, grad_Vr_2, grad_Vs_1, grad_Vs_2 = grad_tuple

        debug_names = [
            "grad_Q", "grad_R", "grad_S",
            "grad_Vq_1", "grad_Vq_2",
            "grad_Vr_1", "grad_Vr_2",
            "grad_Vs_1", "grad_Vs_2",
        ]
        for name, tensor in zip(debug_names, grad_tuple):
            if tensor is not None and tensor.is_floating_point():
                if not torch.isfinite(tensor).all():
                    raise RuntimeError(f"{name} contains non-finite values")

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
            None, #dropout doesn't need a gradient
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
        
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
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
            self.dropout.p
            )
        
            
        y = y.permute(0, 2, 1, 3).contiguous().view(batch_size, ntok, self.n_heads * self.head_dim) 
        #The previous reshape assumed summing over heads, let's adjust to typical attention output handling
        # y = y.permute(0, 2, 1, 3).sum(dim=2).squeeze() # is this correct? doublecheck

        y = self.gelu(y) 
        y = self.Wo(y) 

        return y 