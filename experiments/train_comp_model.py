#!/usr/bin/env python3
"""
Compare CUDA kernels vs Torch C++ reference implementation on comp_model.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# Add parent directory to path
current_script_path = Path(__file__).resolve()
script_dir = str(current_script_path.parent)
parent_dir_str = str(current_script_path.parent.parent)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
if parent_dir_str not in sys.path:
    sys.path.insert(0, parent_dir_str)

from att3ntion import HypergraphAttention, _HypergraphAttentionTorch, QuickGELU


# Data generation functions (copied from gen_data_comp.py to avoid matplotlib import)
def randint(k):
    return np.random.randint(k)

def modOp(va, vb, op, md):
    vc0 = (va + vb) % md
    vc1 = (va - vb) % md
    vc2 = (va * vb) % md
    if vb == 0:
        vc3 = 0
    else:
        vc3 = (va // vb) % md
    
    match op:
        case 0:
            vc = vc0
            ops = '+'
        case 1:
            vc = vc1
            ops = '-'
        case 2:
            vc = vc2
            ops = '*'
        case 3:
            vc = vc3
            ops = '/'
    return vc, ops

def genData1(bs, md, do_print=False):
    '''Task 1: learn the rules of modulo arithmetic.'''
    assert(md < 32-4)
    x = np.zeros((bs, 4, 32), dtype=int)
    
    for b in range(bs):
        va = randint(md)
        vb = randint(md)
        op = np.random.randint(4)
        vc, ops = modOp(va, vb, op, md)
        
        x[b,0,va+4] = 1
        x[b,1,op  ] = 1
        x[b,2,vb+4] = 1
        x[b,3,vc+4] = 1
        
        if do_print:
            print(f"{va} {ops} {vb} = {vc}")
    return x

def genData2(bs, md, do_print=False):
    '''Task 2: learn to compose arithmetic. out = (a op b) op (c op d)'''
    assert(md < 32-4)
    x = np.zeros((bs, 12, 32), dtype=int)
    
    for b in range(bs):
        va = randint(md)
        vb = randint(md)
        vc = randint(md)
        vd = randint(md)
        op1 = np.random.randint(4)
        op2 = np.random.randint(4)
        op3 = np.random.randint(4)
        
        ve, ops1 = modOp(va, vb, op1, md)
        vf, ops2 = modOp(vc, vd, op2, md)
        vg, ops3 = modOp(ve, vf, op3, md)
        
        x[b,0,va+4] = 1
        x[b,1,op1 ] = 1
        x[b,2,vb+4] = 1
        x[b,3,ve+4] = 1
        x[b,4,vc+4] = 1
        x[b,5,op2 ] = 1
        x[b,6,vd+4] = 1
        x[b,7,vf+4] = 1
        x[b,8,ve+4] = 1
        x[b,9,op3 ] = 1
        x[b,10,vf+4] = 1
        x[b,11,vg+4] = 1
        
        if do_print:
            print(f"({va} {ops1} {vb}) {ops3} ({vc} {ops2} {vd}) = {ve} {ops3} {vf} = {vg}")
    return x


class CompModelComparison(nn.Module):
    """Model with hypergraph attention layer for comparison testing."""
    def __init__(self, hidden_dim: int, num_heads: int, n_layers: int, attn_impl: str = 'torch_cpp', modulo: int = 11):
        super().__init__()
        input_dim = 32
        self.embedding_proj = nn.Linear(input_dim, hidden_dim)
        self.attn_impl = attn_impl
        self.modulo = modulo
        
        self.repeated_layers = nn.ModuleList()
        for _ in range(n_layers):
            # Select attention implementation
            if attn_impl == 'cuda':
                attention_layer = HypergraphAttention(hidden_dim, num_heads)
            elif attn_impl == 'torch_cpp':
                attention_layer = _HypergraphAttentionTorch(hidden_dim, num_heads)
            else:
                raise ValueError(f"Unknown attention implementation: {attn_impl}")

            norm1_layer = nn.LayerNorm(hidden_dim)
            ffn_layer = nn.Sequential(
                nn.Linear(hidden_dim, 3 * hidden_dim),
                nn.ReLU(),
                nn.Linear(3 * hidden_dim, hidden_dim)
            )
            norm2_layer = nn.LayerNorm(hidden_dim)

            self.repeated_layers.append(
                nn.ModuleDict({
                    'attention': attention_layer,
                    'norm1': norm1_layer,
                    'ffn': ffn_layer,
                    'norm2': norm2_layer,
                })
            )
        
        self.value_classifier = nn.Linear(hidden_dim, 28)
        self.gelu = QuickGELU()
        
    def forward(self, x):
        x = self.embedding_proj(x)

        for layer_block in self.repeated_layers:
            attn_output = layer_block['attention'](x)
            x = layer_block['norm1'](x + attn_output)
            ffn_output = layer_block['ffn'](x)
            x = layer_block['norm2'](x + ffn_output)
        
        value_pred = self.value_classifier(x[:, -1])
        return value_pred


def prepare_data(data_tensor, device, modulo):
    inputs = data_tensor.copy()
    # Mask the value at last position
    inputs[:, -1, :] = 0
    
    value_targets = np.argmax(data_tensor[:, -1, 4:4+modulo], axis=1)
    
    return (torch.FloatTensor(inputs).to(device), 
            torch.LongTensor(value_targets).to(device))


def time_epoch(model, train_loader, optimizer, criterion, device, modulo, max_batches=None):
    """Time a single training epoch and return metrics."""
    model.train()
    start_time = time.time()
    
    total_loss = 0
    correct_vals = 0
    total = 0
    
    for batch_idx, (inputs_np,) in enumerate(train_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
            
        inputs, value_targets = prepare_data(inputs_np.numpy(), device, modulo)
        
        optimizer.zero_grad()
        value_pred = model(inputs)
        
        loss = criterion(value_pred, value_targets)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total += inputs.size(0)
        correct_vals += (torch.argmax(value_pred, dim=1) == value_targets).sum().item()
    
    elapsed_time = time.time() - start_time
    num_batches = max_batches if max_batches else len(train_loader)
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    val_accuracy = 100 * correct_vals / total if total > 0 else 0
    
    return elapsed_time, avg_loss, val_accuracy


def run_training_comparison(num_epochs=50, batch_size=64, hidden_dim=64, 
                           num_heads=4, device='cuda', modulo=19, 
                           max_batches=None, warmup_epochs=2, n_layers=2,
                           data_gen='genData2'):
    """Run training for both implementations and collect metrics."""
    
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate data
    num_samples = batch_size * 50  # 50 batches per epoch
    print(f"Generating training data ({num_samples} samples, {num_samples // batch_size} batches/epoch)...")
    
    if data_gen == 'genData1':
        data = genData1(num_samples, modulo)
        seq_len = 4
    else:  # genData2
        data = genData2(num_samples, modulo)
        seq_len = 12
    
    print(f"Sequence length N={seq_len}, hidden_dim={hidden_dim}, num_heads={num_heads}")
    
    dataset = TensorDataset(torch.tensor(data))
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    results = {}
    
    for impl_name in ['torch_cpp', 'cuda']:
        print(f"\n{'='*60}")
        print(f"Training with {impl_name.upper()} implementation")
        print(f"{'='*60}")
        
        # Reset seeds
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create model
        model = CompModelComparison(hidden_dim, num_heads, n_layers=n_layers, 
                                    attn_impl=impl_name, modulo=modulo).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        
        # Storage for metrics
        epoch_times = []
        losses = []
        val_accuracies = []
        
        # Warmup epochs
        print(f"\nRunning {warmup_epochs} warmup epoch(s)...", flush=True)
        for epoch in range(warmup_epochs):
            time_epoch(model, train_loader, optimizer, criterion, device, modulo, max_batches)
            print(f"  Warmup {epoch+1}/{warmup_epochs} done", flush=True)
        
        # Actual training with timing
        print(f"\nTraining for {num_epochs} epochs...", flush=True)
        for epoch in range(num_epochs):
            elapsed, loss, val_acc = time_epoch(
                model, train_loader, optimizer, criterion, device, modulo, max_batches
            )
            
            epoch_times.append(elapsed)
            losses.append(loss)
            val_accuracies.append(val_acc)
            
            print(f'  [{epoch+1:3d}/{num_epochs}] {elapsed:.2f}s | Loss: {loss:.4f} | Val: {val_acc:5.1f}%', flush=True)
        
        results[impl_name] = {
            'epoch_times': epoch_times,
            'losses': losses,
            'val_accuracies': val_accuracies,
            'avg_time': np.mean(epoch_times),
            'std_time': np.std(epoch_times),
            'final_loss': losses[-1],
            'final_val_acc': val_accuracies[-1]
        }
        
        print(f"\n{impl_name.upper()} Summary:")
        print(f"  Average epoch time: {results[impl_name]['avg_time']:.3f} ± {results[impl_name]['std_time']:.3f}s")
        print(f"  Final loss: {results[impl_name]['final_loss']:.4f}")
        print(f"  Final val accuracy: {results[impl_name]['final_val_acc']:.2f}%")
    
    return results


def print_summary(results):
    """Print summary statistics."""
    speedup = results['torch_cpp']['avg_time'] / results['cuda']['avg_time']
    
    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"\nTorch C++ Implementation:")
    print(f"  Avg epoch time: {results['torch_cpp']['avg_time']:.3f} ± {results['torch_cpp']['std_time']:.3f}s")
    print(f"  Total time: {sum(results['torch_cpp']['epoch_times']):.2f}s")
    print(f"  Final loss: {results['torch_cpp']['final_loss']:.4f}")
    print(f"  Final val acc: {results['torch_cpp']['final_val_acc']:.2f}%")
    
    print(f"\nCUDA Implementation:")
    print(f"  Avg epoch time: {results['cuda']['avg_time']:.3f} ± {results['cuda']['std_time']:.3f}s")
    print(f"  Total time: {sum(results['cuda']['epoch_times']):.2f}s")
    print(f"  Final loss: {results['cuda']['final_loss']:.4f}")
    print(f"  Final val acc: {results['cuda']['final_val_acc']:.2f}%")
    
    print(f"\nSpeedup: {speedup:.2f}x")
    print(f"Time saved per epoch: {results['torch_cpp']['avg_time'] - results['cuda']['avg_time']:.3f}s")
    print(f"Total time saved: {sum(results['torch_cpp']['epoch_times']) - sum(results['cuda']['epoch_times']):.2f}s")
    print(f"{'='*60}\n")


def save_results(results, output_dir):
    """Save results to a text file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'comp_model_results.txt')
    
    speedup = results['torch_cpp']['avg_time'] / results['cuda']['avg_time']
    
    with open(filepath, 'w') as f:
        f.write("CUDA vs Torch C++ Implementation Comparison (comp_model)\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("PERFORMANCE SUMMARY\n")
        f.write("-" * 60 + "\n\n")
        
        f.write("Torch C++ Implementation:\n")
        f.write(f"  Avg epoch time: {results['torch_cpp']['avg_time']:.3f} ± {results['torch_cpp']['std_time']:.3f}s\n")
        f.write(f"  Total time: {sum(results['torch_cpp']['epoch_times']):.2f}s\n")
        f.write(f"  Final loss: {results['torch_cpp']['final_loss']:.4f}\n")
        f.write(f"  Final val acc: {results['torch_cpp']['final_val_acc']:.2f}%\n\n")
        
        f.write("CUDA Implementation:\n")
        f.write(f"  Avg epoch time: {results['cuda']['avg_time']:.3f} ± {results['cuda']['std_time']:.3f}s\n")
        f.write(f"  Total time: {sum(results['cuda']['epoch_times']):.2f}s\n")
        f.write(f"  Final loss: {results['cuda']['final_loss']:.4f}\n")
        f.write(f"  Final val acc: {results['cuda']['final_val_acc']:.2f}%\n\n")
        
        f.write(f"Speedup: {speedup:.2f}x\n")
        f.write(f"Time saved per epoch: {results['torch_cpp']['avg_time'] - results['cuda']['avg_time']:.3f}s\n")
        f.write(f"Total time saved: {sum(results['torch_cpp']['epoch_times']) - sum(results['cuda']['epoch_times']):.2f}s\n")
    
    print(f"\nResults saved to: {filepath}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare CUDA vs Torch C++ implementation on comp_model'
    )
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--hidden-dim', type=int, default=64,
                       help='Hidden dimension (default: 64)')
    parser.add_argument('--num-heads', type=int, default=4,
                       help='Number of attention heads (default: 4)')
    parser.add_argument('--n-layers', type=int, default=2,
                       help='Number of attention layers (default: 2)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cpu', 'cuda'],
                       help='Device to use (default: cuda)')
    parser.add_argument('--modulo', type=int, default=19,
                       help='Modulo for arithmetic (default: 19)')
    parser.add_argument('--max-batches', type=int, default=None,
                       help='Max batches per epoch for quick testing')
    parser.add_argument('--warmup-epochs', type=int, default=2,
                       help='Number of warmup epochs (default: 2)')
    parser.add_argument('--output-dir', type=str, default='./comparison_plots',
                       help='Directory to save results (default: ./comparison_plots)')
    parser.add_argument('--data-gen', type=str, default='genData2',
                       choices=['genData1', 'genData2'],
                       help='Data generator to use (genData1: N=4, genData2: N=12)')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Run comparison
    results = run_training_comparison(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        n_layers=args.n_layers,
        device=args.device,
        modulo=args.modulo,
        max_batches=args.max_batches,
        warmup_epochs=args.warmup_epochs,
        data_gen=args.data_gen
    )
    
    # Print and save results
    print_summary(results)
    save_results(results, args.output_dir)
