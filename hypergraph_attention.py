import torch
import torch.nn as nn
import math
import hyper_attn_cpp_manual
import hyper_attn_cpp_reference
from torch.autograd import Function

def _pad_to_multiple(tensor, multiple, dim=2):
    """Pad tensor along specified dimension to be a multiple of `multiple`."""
    size = tensor.size(dim)
    if size % multiple == 0:
        return tensor, 0
    pad_size = multiple - (size % multiple)
    # Create padding tuple: (last_dim_left, last_dim_right, ..., dim_left, dim_right)
    # For 4D tensor [B, H, N, D] and dim=2, we need to pad N
    ndim = tensor.ndim
    pad = [0] * (2 * ndim)
    # PyTorch pad works from last dim backwards
    pad_idx = 2 * (ndim - 1 - dim)
    pad[pad_idx + 1] = pad_size  # pad on the right side
    return torch.nn.functional.pad(tensor, pad), pad_size


class HyperAttentionAutograd(Function):
    """
    Bridge between PyTorch autograd and the manual C++ forward/backward passes.
    Saves softmax statistics from forward pass to avoid redundant computation in backward.
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
        # CUDA kernels require sequence length to be a multiple of 16 (TILE_I)
        TILE_SIZE = 16
        orig_seq_len = Q.size(2)
        
        Q, pad_q = _pad_to_multiple(Q.contiguous(), TILE_SIZE, dim=2)
        R, pad_r = _pad_to_multiple(R.contiguous(), TILE_SIZE, dim=2)
        S, pad_s = _pad_to_multiple(S.contiguous(), TILE_SIZE, dim=2)
        Vq_1, _ = _pad_to_multiple(Vq_1.contiguous(), TILE_SIZE, dim=2)
        Vq_2, _ = _pad_to_multiple(Vq_2.contiguous(), TILE_SIZE, dim=2)
        Vr_1, _ = _pad_to_multiple(Vr_1.contiguous(), TILE_SIZE, dim=2)
        Vr_2, _ = _pad_to_multiple(Vr_2.contiguous(), TILE_SIZE, dim=2)
        Vs_1, _ = _pad_to_multiple(Vs_1.contiguous(), TILE_SIZE, dim=2)
        Vs_2, _ = _pad_to_multiple(Vs_2.contiguous(), TILE_SIZE, dim=2)

        outputs_tuple = hyper_attn_cpp_manual.forward(
            Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2, dropout_rate
        )

        # Forward now returns 12 tensors: 6 outputs + 6 softmax stats
        if isinstance(outputs_tuple, tuple) and len(outputs_tuple) == 12:
            Y_q, Y_r, Y_s, Y_q_, Y_r_, Y_s_, m_i, l_i, m_j, l_j, m_k, l_k = outputs_tuple
            final_output = Y_q + Y_r + Y_s + Y_q_ + Y_r_ + Y_s_
        else:
            raise TypeError(f"C++ forward expected to return a tuple of 12 Tensors, but got {type(outputs_tuple)} with len {len(outputs_tuple) if isinstance(outputs_tuple, tuple) else 'N/A'}")

        # Remove padding from output to match original sequence length
        if orig_seq_len != final_output.size(2):
            final_output = final_output[:, :, :orig_seq_len, :]

        # Save input tensors AND softmax stats for backward pass
        # Also save original sequence length for backward padding
        ctx.save_for_backward(Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2,
                              m_i, l_i, m_j, l_j, m_k, l_k)
        ctx.orig_seq_len = orig_seq_len
        
        return final_output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Calls the C++ backward pass using saved tensors and pre-computed softmax stats.
        This avoids redundant computation of softmax statistics.
        Args:
            ctx: Context object with saved tensors.
            grad_output: Gradient of the loss w.r.t. the summed output.
        Returns:
            A tuple of gradients corresponding *exactly* to the inputs of the
            forward function (Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2, dropout_rate).
        """
        # Retrieve saved tensors including softmax stats (these are already padded)
        Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2, \
            m_i, l_i, m_j, l_j, m_k, l_k = ctx.saved_tensors
        orig_seq_len = ctx.orig_seq_len
        
        # Pad grad_output to match the padded sequence length used in forward
        TILE_SIZE = 16
        grad_output, _ = _pad_to_multiple(grad_output.contiguous(), TILE_SIZE, dim=2)

        # Use backward with pre-computed softmax stats from forward pass
        grad_tuple = hyper_attn_cpp_manual.backward(
            grad_output, Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2,
            m_i, l_i, m_j, l_j, m_k, l_k
        )

        # Ensure grad_tuple from C++ has the correct number of elements (9 for the 9 tensor inputs)
        if not (isinstance(grad_tuple, tuple) and len(grad_tuple) == 9):
            raise ValueError(f"C++ backward expected to return a tuple of 9 gradients, got {len(grad_tuple) if isinstance(grad_tuple, tuple) else type(grad_tuple)}")

        grad_Q, grad_R, grad_S, grad_Vq_1, grad_Vq_2, grad_Vr_1, grad_Vr_2, grad_Vs_1, grad_Vs_2 = grad_tuple

        # Remove padding from gradients to match original sequence length
        if orig_seq_len != grad_Q.size(2):
            grad_Q = grad_Q[:, :, :orig_seq_len, :]
            grad_R = grad_R[:, :, :orig_seq_len, :]
            grad_S = grad_S[:, :, :orig_seq_len, :]
            grad_Vq_1 = grad_Vq_1[:, :, :orig_seq_len, :]
            grad_Vq_2 = grad_Vq_2[:, :, :orig_seq_len, :]
            grad_Vr_1 = grad_Vr_1[:, :, :orig_seq_len, :]
            grad_Vr_2 = grad_Vr_2[:, :, :orig_seq_len, :]
            grad_Vs_1 = grad_Vs_1[:, :, :orig_seq_len, :]
            grad_Vs_2 = grad_Vs_2[:, :, :orig_seq_len, :]

        debug_names = [
            "grad_Q", "grad_R", "grad_S",
            "grad_Vq_1", "grad_Vq_2",
            "grad_Vr_1", "grad_Vr_2",
            "grad_Vs_1", "grad_Vs_2",
        ]
        grads = (grad_Q, grad_R, grad_S, grad_Vq_1, grad_Vq_2, grad_Vr_1, grad_Vr_2, grad_Vs_1, grad_Vs_2)
        for name, tensor in zip(debug_names, grads):
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
        
        # Core 3-way attention - returns summed output
        y = HyperAttentionAutograd.apply( 
            Q, R, S,
            Vq_1, Vq_2,
            Vr_1, Vr_2,
            Vs_1, Vs_2,
            self.dropout.p
        )
        
        # Reshape: concatenate heads (CPP uses head_dim = d_model // n_heads)
        y = y.permute(0, 2, 1, 3).contiguous().view(batch_size, ntok, self.n_heads * self.head_dim)
        
        y = self.Wo(y) 

        return y 


class HypergraphAttentionCPPReference(nn.Module):
    """
    PyTorch wrapper for cpp/torch_att3ntion.cpp (reference implementation).
    Uses PyTorch autograd for backward pass.
    """
    def __init__(self, d_model, n_heads, dropout_rate=0):
        super(HypergraphAttentionCPPReference, self).__init__()
        
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
        
        # Core 3-way attention using C++ reference (uses autograd for backward)
        outputs_tuple = hyper_attn_cpp_reference.forward(
            Q.contiguous(), R.contiguous(), S.contiguous(),
            Vq_1.contiguous(), Vq_2.contiguous(),
            Vr_1.contiguous(), Vr_2.contiguous(),
            Vs_1.contiguous(), Vs_2.contiguous(),
            self.dropout.p
        )
        
        # Sum the 6 output tensors
        y = sum(outputs_tuple)
        
        # Reshape: concatenate heads
        y = y.permute(0, 2, 1, 3).contiguous().view(batch_size, ntok, self.n_heads * self.head_dim)
        
        y = self.Wo(y) 

        return y