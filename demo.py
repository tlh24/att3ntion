#!/usr/bin/env python3
"""
att3ntion Demo: Hypergraph Attention with Flash-style Memory Efficiency

This demo showcases:
1. Basic usage of the HypergraphAttention layer
2. Memory scaling comparison (O(N) vs naive O(N³))
3. Timing comparison: CUDA kernels vs PyTorch reference
4. Training on a compositional arithmetic task

Usage:
    python demo.py                    # Quick sanity check
    python demo.py --memory           # Memory scaling benchmark
    python demo.py --benchmark        # CUDA vs Torch timing comparison
    python demo.py --train            # Train on modular arithmetic
    python demo.py --all              # Run everything
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ============================================================================
# BASIC USAGE DEMO
# ============================================================================

def demo_basic_usage():
    """Demonstrate basic usage of HypergraphAttention."""
    print("\n" + "=" * 60)
    print("🔺 att3ntion: 3-Way Hypergraph Attention")
    print("=" * 60)
    
    from att3ntion import HypergraphAttention
    
    # Create layer
    # CUDA kernel requires head_dim (d_model/n_heads) to be a multiple of 4
    d_model = 64
    n_heads = 4  # head_dim = 64/4 = 16
    layer = HypergraphAttention(d_model=d_model, n_heads=n_heads)
    
    print(f"\nLayer config: d_model={d_model}, n_heads={n_heads}")
    print(f"Parameters: {sum(p.numel() for p in layer.parameters()):,}")
    
    # Move to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    layer = layer.to(device)
    print(f"Device: {device}")
    
    # Forward pass
    batch_size, seq_len = 2, 32
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    
    with torch.no_grad():
        y = layer(x)
    
    print(f"\nInput shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")
    print("✓ Forward pass successful!")
    
    # Test backward pass
    x.requires_grad = True
    y = layer(x)
    loss = y.sum()
    loss.backward()
    
    print("✓ Backward pass successful!")
    print("\n" + "-" * 60)


# ============================================================================
# MEMORY SCALING BENCHMARK
# ============================================================================

def demo_memory_scaling():
    """Show O(N) memory scaling vs naive O(N³)."""
    print("\n" + "=" * 60)
    print("📊 Memory Scaling: att3ntion vs Naive Implementation")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("⚠ CUDA not available. Skipping memory benchmark.")
        return
    
    from att3ntion import HypergraphAttention
    
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Testing sequence lengths: 32, 64, 128, 256, 512")
    print("\nNaive 3-way attention would require O(N³) memory for the")
    print("attention tensor A[batch, heads, i, j, k]. att3ntion uses")
    print("flash-style tiling to achieve O(N) memory.\n")
    
    print(f"{'N':>6} | {'Actual (MB)':>12} | {'Naive O(N³)':>12} | {'Savings':>10}")
    print("-" * 50)
    
    # CUDA kernel requires head_dim (d_model/n_heads) to be a multiple of 4
    d_model = 64
    n_heads = 4  # head_dim = 64/4 = 16
    batch_size = 4
    
    for seq_len in [32, 64, 128, 256, 512]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        layer = HypergraphAttention(d_model=d_model, n_heads=n_heads).cuda()
        x = torch.randn(batch_size, seq_len, d_model, device='cuda', requires_grad=True)
        
        # Forward + backward
        y = layer(x)
        loss = y.sum()
        loss.backward()
        
        torch.cuda.synchronize()
        actual_mb = torch.cuda.max_memory_allocated() / 1e6
        
        # Naive would need: batch * heads * N * N * N * 4 bytes (float32)
        naive_mb = batch_size * n_heads * (seq_len ** 3) * 4 / 1e6
        
        savings = naive_mb / actual_mb if actual_mb > 0 else 0
        
        print(f"{seq_len:>6} | {actual_mb:>10.1f}MB | {naive_mb:>10.1f}MB | {savings:>8.0f}x")
        
        del layer, x, y
        torch.cuda.empty_cache()
    
    print("\n" + "-" * 60)


# ============================================================================
# TIMING BENCHMARK: CUDA vs TORCH
# ============================================================================

def benchmark_fn(fn, warmup=5, iters=20):
    """Benchmark a function using CUDA events. Returns (median_ms, min_ms, max_ms)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    times.sort()
    median = times[len(times) // 2]
    return median, min(times), max(times)


def _naive_gather_forward(Q, R, S, Vq, Vr, Vs):
    """Naive PyTorch hypergraph attention (gather branch only): materializes
    the full O(N³) score tensor, three joint softmaxes, three contractions.
    This is what running the same math in plain PyTorch would cost."""
    B, H, N, _ = Q.shape
    scale = 1.0 / (Q.shape[-1] ** 0.5)
    x = torch.einsum('bhid,bhjd,bhkd->bhijk', Q, R, S) * scale
    Aq = torch.softmax(x.reshape(B, H, N, -1), -1).reshape(B, H, N, N, N)
    xr = x.transpose(2, 3)  # [b,h,j,i,k]
    Ar = torch.softmax(xr.reshape(B, H, N, -1), -1).reshape(B, H, N, N, N)
    xs = x.permute(0, 1, 4, 2, 3)  # [b,h,k,i,j]
    As = torch.softmax(xs.reshape(B, H, N, -1), -1).reshape(B, H, N, N, N)
    Y_q = torch.einsum('bhijk,bhjd,bhkd->bhid', Aq, Vr, Vs)
    Y_r = torch.einsum('bhjik,bhid,bhkd->bhjd', Ar, Vq, Vs)
    Y_s = torch.einsum('bhkij,bhid,bhjd->bhkd', As, Vq, Vr)
    return Y_q, Y_r, Y_s


def demo_benchmark():
    """Compare CUDA kernel timing vs naive PyTorch (gather-only: scatter
    branch unused, matching current models — the CUDA path skips scatter
    work and runs the tensor-core backward when eligible)."""
    print("\n" + "=" * 60)
    print("⏱️  Timing Benchmark: CUDA Kernels vs Naive PyTorch (no scatter)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("⚠ CUDA not available. Skipping timing benchmark.")
        return

    try:
        import att3ntion._cuda_kernels as cuda_ext
    except ImportError:
        print("⚠ CUDA extension not found. Run: python setup.py develop")
        return

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print("\n'Torch' = naive PyTorch materializing the O(N³) score tensor.")
    print("'Time ×' = torch_ms / cuda_ms (higher = CUDA faster).")
    print("Backward includes the torch forward re-run autograd requires.\n")

    # Test configurations: (N, D) pairs. D=64 rows use the tensor-core
    # forward gathers and backward; D=32 rows use the scalar kernels.
    configs = [
        ("N32_D32",  32,  32),
        ("N64_D32",  64,  32),
        ("N64_D64",  64,  64),
        ("N128_D32", 128, 32),
        ("N128_D64", 128, 64),
        ("N256_D64", 256, 64),
    ]

    B, H = 1, 2  # batch, heads
    warmup, iters = 5, 20

    print(f"{'Config':<10} │ {'Forward':^26} │ {'Backward':^26} │ {'Step (f+b)':^26}")
    print(f"{'':10} │ {'CUDA':>7} {'Torch':>8} {'Time ×':>7} │ {'CUDA':>7} {'Torch':>8} {'Time ×':>7} │ {'CUDA':>7} {'Torch':>8} {'Time ×':>7}")
    print("─" * 100)

    for name, N, D in configs:
        torch.cuda.empty_cache()
        torch.manual_seed(42)

        Q, R, S, Vq_1, Vr_1, Vs_1 = (
            torch.randn(B, H, N, D, device='cuda') for _ in range(6))
        # Scatter unused: zero V2 operands (forward skips the scatter
        # kernels) and zero scatter cotangents (backward takes the TC path).
        zero_bf = torch.zeros(B, H, N, D, device='cuda', dtype=torch.bfloat16)
        cuda_inputs = tuple(t.to(torch.bfloat16) for t in (Q, R, S)) + (
            Vq_1.to(torch.bfloat16), zero_bf,
            Vr_1.to(torch.bfloat16), zero_bf,
            Vs_1.to(torch.bfloat16), zero_bf)
        grad_Y_q, grad_Y_r, grad_Y_s = (
            torch.randn(B, H, N, D, device='cuda', dtype=torch.bfloat16)
            for _ in range(3))

        def run_cuda_fwd():
            return cuda_ext.forward(*cuda_inputs, 0.0)

        def run_ref_fwd():
            with torch.no_grad():
                return _naive_gather_forward(Q, R, S, Vq_1, Vr_1, Vs_1)

        cuda_fwd_ms, _, _ = benchmark_fn(run_cuda_fwd, warmup, iters)
        ref_fwd_ms, _, _ = benchmark_fn(run_ref_fwd, warmup, iters)

        # Forward outputs: stats for backward + Y for the TC path's
        # collapsed correction sums.
        fwd_out = cuda_ext.forward(*cuda_inputs, 0.0)
        Y_q, Y_r, Y_s = fwd_out[0], fwd_out[1], fwd_out[2]
        m_i, l_i, m_j, l_j, m_k, l_k = fwd_out[6:12]

        def run_cuda_bwd():
            return cuda_ext.backward(
                grad_Y_q, grad_Y_r, grad_Y_s, zero_bf, zero_bf, zero_bf,
                *cuda_inputs, m_i, l_i, m_j, l_j, m_k, l_k, 0.0,
                None, Y_q, Y_r, Y_s
            )

        gq, gr, gs = grad_Y_q.float(), grad_Y_r.float(), grad_Y_s.float()

        def run_ref_bwd():
            ref_inputs = [t.detach().clone().requires_grad_(True)
                          for t in (Q, R, S, Vq_1, Vr_1, Vs_1)]
            yq, yr, ys = _naive_gather_forward(*ref_inputs)
            ((yq * gq).sum() + (yr * gr).sum() + (ys * gs).sum()).backward()

        cuda_bwd_ms, _, _ = benchmark_fn(run_cuda_bwd, warmup, iters)
        ref_bwd_ms, _, _ = benchmark_fn(run_ref_bwd, warmup, iters)

        def fmt_x(r):
            return f"{r:.1f}×" if r >= 1 else f"{r:.2f}×"

        fwd_x = ref_fwd_ms / cuda_fwd_ms
        bwd_x = ref_bwd_ms / cuda_bwd_ms
        step_cuda = cuda_fwd_ms + cuda_bwd_ms
        step_ref = ref_fwd_ms + ref_bwd_ms
        print(f"{name:<10} │ {cuda_fwd_ms:>6.2f} {ref_fwd_ms:>7.2f} {fmt_x(fwd_x):>7} │ "
              f"{cuda_bwd_ms:>6.2f} {ref_bwd_ms:>7.2f} {fmt_x(bwd_x):>7} │ "
              f"{step_cuda:>6.2f} {step_ref:>7.2f} {fmt_x(step_ref / step_cuda):>7}")

        del Q, R, S, Vq_1, Vr_1, Vs_1, grad_Y_q, grad_Y_r, grad_Y_s, fwd_out
        torch.cuda.empty_cache()

    print("─" * 100)
    print("\nTimes in ms. Naive PyTorch also allocates the O(N³) score tensor")
    print("(see --memory); att3ntion never materializes it.")
    print("\n" + "-" * 60)


# ============================================================================
# TRAINING DEMO: COMPOSITIONAL ARITHMETIC
# ============================================================================

def modOp(va, vb, op, md):
    """Perform modular arithmetic operation."""
    if op == 0:
        return (va + vb) % md, '+'
    elif op == 1:
        return (va - vb) % md, '-'
    elif op == 2:
        return (va * vb) % md, '*'
    else:
        return (va // vb) % md if vb != 0 else 0, '/'


def generate_data(num_samples, modulo=19):
    """
    Generate compositional arithmetic data.
    Task: Given (a op₁ b) op₃ (c op₂ d), predict the final result.
    
    This requires understanding 3-way relationships between operands
    and operators - exactly what hypergraph attention excels at.
    
    Note: Sequence length is 16 (padded from 12) to satisfy CUDA kernel
    constraint that sequence length must be a multiple of 16.
    """
    # Use 16 tokens (padded from 12) to satisfy CUDA TILE_I=16 constraint
    x = np.zeros((num_samples, 16, 32), dtype=np.float32)
    
    for b in range(num_samples):
        va = np.random.randint(modulo)
        vb = np.random.randint(modulo)
        vc = np.random.randint(modulo)
        vd = np.random.randint(modulo)
        op1 = np.random.randint(4)
        op2 = np.random.randint(4)
        op3 = np.random.randint(4)
        
        ve, _ = modOp(va, vb, op1, modulo)
        vf, _ = modOp(vc, vd, op2, modulo)
        vg, _ = modOp(ve, vf, op3, modulo)
        
        # Encode as one-hot: positions 0-3 for operators, 4+ for values
        x[b, 0, va + 4] = 1   # a
        x[b, 1, op1] = 1      # op1
        x[b, 2, vb + 4] = 1   # b
        x[b, 3, ve + 4] = 1   # (a op1 b)
        x[b, 4, vc + 4] = 1   # c
        x[b, 5, op2] = 1      # op2
        x[b, 6, vd + 4] = 1   # d
        x[b, 7, vf + 4] = 1   # (c op2 d)
        x[b, 8, ve + 4] = 1   # intermediate result 1
        x[b, 9, op3] = 1      # op3
        x[b, 10, vf + 4] = 1  # intermediate result 2
        x[b, 11, vg + 4] = 1  # final result (target)
        # Positions 12-15 remain zero (padding)
    
    return x


class ArithmeticModel(nn.Module):
    """Simple model using hypergraph attention for arithmetic reasoning."""
    
    # Target position in the sequence (position 11 contains the target, 12-15 are padding)
    TARGET_POS = 11
    
    def __init__(self, hidden_dim=128, num_heads=4, n_layers=2, modulo=19):
        super().__init__()
        from att3ntion import HypergraphAttention
        
        self.embedding = nn.Linear(32, hidden_dim)
        self.modulo = modulo
        
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'attention': HypergraphAttention(hidden_dim, num_heads),
                'norm1': nn.LayerNorm(hidden_dim),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_dim, 3 * hidden_dim),
                    nn.ReLU(),
                    nn.Linear(3 * hidden_dim, hidden_dim)
                ),
                'norm2': nn.LayerNorm(hidden_dim),
            }))
        
        self.classifier = nn.Linear(hidden_dim, modulo)
    
    def forward(self, x):
        x = self.embedding(x)
        
        for layer in self.layers:
            x = layer['norm1'](x + layer['attention'](x))
            x = layer['norm2'](x + layer['ffn'](x))
        
        # Predict from target position (not last, since we have padding)
        return self.classifier(x[:, self.TARGET_POS])


def demo_training(epochs=200, batch_size=64):
    """Train on compositional arithmetic task."""
    print("\n" + "=" * 60)
    print("🧮 Training Demo: Compositional Arithmetic")
    print("=" * 60)
    
    print("\nTask: Learn to compute (a op₁ b) op₃ (c op₂ d) mod 19")
    print("Example: (3 + 5) * (2 - 1) = 8 * 1 = 8")
    print("\nThis requires understanding 3-way relationships between")
    print("operands and operators - ideal for hypergraph attention.\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    modulo = 19
    # CUDA kernel requires head_dim (hidden_dim/num_heads) to be a multiple of 4
    hidden_dim = 128
    num_heads = 4  # head_dim = 128/4 = 32
    
    # Generate data
    print(f"Generating {batch_size * 50} training samples...")
    data = generate_data(batch_size * 50, modulo)
    
    # Prepare dataset
    # Target is at position 11 (positions 12-15 are padding)
    TARGET_POS = 11
    inputs = data.copy()
    inputs[:, TARGET_POS, :] = 0  # Mask target
    targets = np.argmax(data[:, TARGET_POS, 4:4+modulo], axis=1)
    
    dataset = TensorDataset(
        torch.FloatTensor(inputs),
        torch.LongTensor(targets)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    model = ArithmeticModel(hidden_dim, num_heads, n_layers=2, modulo=modulo).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\nTraining for {epochs} epochs...")
    print(f"{'Epoch':>6} | {'Loss':>8} | {'Accuracy':>10} | {'Time':>8}")
    print("-" * 45)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        start = time.time()
        
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            correct += (pred.argmax(dim=1) == y).sum().item()
            total += y.size(0)
        
        elapsed = time.time() - start
        acc = 100 * correct / total
        avg_loss = total_loss / len(loader)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"{epoch+1:>6} | {avg_loss:>8.4f} | {acc:>9.1f}% | {elapsed:>7.2f}s")
    
    print("-" * 45)
    print(f"\nFinal accuracy: {acc:.1f}%")
    
    if acc > 90:
        print("✓ Model successfully learned compositional arithmetic!")
    elif acc > 50:
        print("~ Model is learning, try more epochs for better results.")
    else:
        print("⚠ Model needs more training or tuning.")
    
    print("\n" + "-" * 60)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='att3ntion Demo: Hypergraph Attention',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python demo.py              # Quick sanity check
    python demo.py --memory     # Memory scaling benchmark
    python demo.py --benchmark  # CUDA vs Torch timing comparison
    python demo.py --train      # Train on arithmetic task
    python demo.py --all        # Run all demos
        """
    )
    parser.add_argument('--memory', action='store_true',
                        help='Run memory scaling benchmark')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run CUDA vs Torch timing comparison')
    parser.add_argument('--train', action='store_true',
                        help='Train on compositional arithmetic')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs (default: 50)')
    parser.add_argument('--all', action='store_true',
                        help='Run all demos')
    
    args = parser.parse_args()
    
    # Always run basic demo
    demo_basic_usage()
    
    if args.memory or args.all:
        demo_memory_scaling()
    
    if args.benchmark or args.all:
        demo_benchmark()
    
    if args.train or args.all:
        demo_training(epochs=args.epochs)
    
    if not (args.memory or args.benchmark or args.train or args.all):
        print("\nTip: Run with --memory, --benchmark, --train, or --all for more demos!")
    
    print("\n✨ Demo complete!\n")


if __name__ == '__main__':
    main()
