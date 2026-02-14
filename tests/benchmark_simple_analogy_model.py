#!/usr/bin/env python3
"""
Compare CUDA kernels vs PyTorch reference implementation on the simple analogy model.
Generates performance comparison graphs.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
from pathlib import Path

# Try to import matplotlib, but don't fail if it's not available
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, will save results to text file only")

# Add parent directory to path
current_script_path = Path(__file__).resolve()
parent_dir = current_script_path.parent.parent
parent_dir_str = str(parent_dir)
if parent_dir_str not in sys.path:
    sys.path.insert(0, parent_dir_str)

from tests.simple_analogy_model import train_model, SimpleAnalogyModel, prepare_data
from gen_data import genData
from torch.utils.data import DataLoader, TensorDataset


def time_epoch(model, train_loader, optimizer, criterion, device, max_batches=None):
    """Time a single training epoch and return metrics."""
    model.train()
    start_time = time.time()
    
    total_loss = 0
    correct_ops = 0
    correct_vals = 0
    total = 0
    
    for batch_idx, (inputs_np,) in enumerate(train_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
            
        inputs, op_targets, value_targets = prepare_data(inputs_np.numpy(), device)
        
        optimizer.zero_grad()
        op_pred1, op_pred5, value_pred = model(inputs)
        
        loss = (criterion(op_pred1, op_targets[:, 0]) + 
               criterion(op_pred5, op_targets[:, 1]) + 
               criterion(value_pred, value_targets))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total += inputs.size(0)
        
        correct_ops += ((torch.argmax(op_pred1, dim=1) == op_targets[:, 0]).sum().item() +
                      (torch.argmax(op_pred5, dim=1) == op_targets[:, 1]).sum().item()) / 2
        correct_vals += (torch.argmax(value_pred, dim=1) == value_targets).sum().item()
    
    elapsed_time = time.time() - start_time
    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
    op_accuracy = 100 * correct_ops / total if total > 0 else 0
    val_accuracy = 100 * correct_vals / total if total > 0 else 0
    
    return elapsed_time, avg_loss, op_accuracy, val_accuracy


def run_training_comparison(num_epochs=50, batch_size=128, hidden_dim=64, 
                           num_heads=4, device='cuda', modulo=19, 
                           max_batches=None, warmup_epochs=2):
    """Run training for both implementations and collect metrics."""
    
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate data - use smaller dataset for faster epochs
    num_samples = batch_size * 50  # 50 batches per epoch instead of 1000
    print(f"Generating training data ({num_samples} samples, {num_samples // batch_size} batches/epoch)...")
    data = genData(num_samples, modulo)
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
        model = SimpleAnalogyModel(hidden_dim, num_heads, n_layers=2, attn_impl=impl_name).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        
        # Storage for metrics
        epoch_times = []
        losses = []
        op_accuracies = []
        val_accuracies = []
        
        # Warmup epochs (not counted in metrics)
        print(f"\nRunning {warmup_epochs} warmup epoch(s)...", flush=True)
        for epoch in range(warmup_epochs):
            time_epoch(model, train_loader, optimizer, criterion, device, max_batches)
            print(f"  Warmup {epoch+1}/{warmup_epochs} done", flush=True)
        
        # Actual training with timing
        print(f"\nTraining for {num_epochs} epochs...", flush=True)
        for epoch in range(num_epochs):
            elapsed, loss, op_acc, val_acc = time_epoch(
                model, train_loader, optimizer, criterion, device, max_batches
            )
            
            epoch_times.append(elapsed)
            losses.append(loss)
            op_accuracies.append(op_acc)
            val_accuracies.append(val_acc)
            
            # Print every epoch for progress tracking
            print(f'  [{epoch+1:3d}/{num_epochs}] {elapsed:.2f}s | Loss: {loss:.4f} | Op: {op_acc:5.1f}% | Val: {val_acc:5.1f}%', flush=True)
        
        results[impl_name] = {
            'epoch_times': epoch_times,
            'losses': losses,
            'op_accuracies': op_accuracies,
            'val_accuracies': val_accuracies,
            'avg_time': np.mean(epoch_times),
            'std_time': np.std(epoch_times),
            'final_loss': losses[-1],
            'final_op_acc': op_accuracies[-1],
            'final_val_acc': val_accuracies[-1]
        }
        
        print(f"\n{impl_name.upper()} Summary:")
        print(f"  Average epoch time: {results[impl_name]['avg_time']:.3f} ± {results[impl_name]['std_time']:.3f}s")
        print(f"  Final loss: {results[impl_name]['final_loss']:.4f}")
        print(f"  Final op accuracy: {results[impl_name]['final_op_acc']:.2f}%")
        print(f"  Final val accuracy: {results[impl_name]['final_val_acc']:.2f}%")
    
    return results


def plot_comparison(results, output_dir='./comparison_plots'):
    """Generate comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    epochs = np.arange(1, len(results['torch_cpp']['losses']) + 1)
    
    # Save results to text file
    save_results_to_file(results, output_dir)
    
    if not HAS_MATPLOTLIB:
        print(f"\nResults saved to {output_dir}/results.txt")
        print_summary(results)
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('CUDA Kernels vs Torch C++ Reference Comparison', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Training Loss
    ax = axes[0, 0]
    ax.plot(epochs, results['torch_cpp']['losses'], 'b-', label='Torch C++', linewidth=2, alpha=0.7)
    ax.plot(epochs, results['cuda']['losses'], 'r-', label='CUDA', linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Training Loss Over Time', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Operator Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, results['torch_cpp']['op_accuracies'], 'b-', label='Torch C++', linewidth=2, alpha=0.7)
    ax.plot(epochs, results['cuda']['op_accuracies'], 'r-', label='CUDA', linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('Operator Prediction Accuracy', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Value Accuracy
    ax = axes[1, 0]
    ax.plot(epochs, results['torch_cpp']['val_accuracies'], 'b-', label='Torch C++', linewidth=2, alpha=0.7)
    ax.plot(epochs, results['cuda']['val_accuracies'], 'r-', label='CUDA', linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('Value Prediction Accuracy', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Epoch Time Comparison (Bar chart with error bars)
    ax = axes[1, 1]
    implementations = ['Torch C++', 'CUDA']
    avg_times = [results['torch_cpp']['avg_time'], results['cuda']['avg_time']]
    std_times = [results['torch_cpp']['std_time'], results['cuda']['std_time']]
    
    bars = ax.bar(implementations, avg_times, yerr=std_times, 
                   color=['blue', 'red'], alpha=0.7, capsize=10)
    ax.set_ylabel('Time (seconds)', fontsize=11)
    ax.set_title('Average Epoch Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, avg_time, std_time in zip(bars, avg_times, std_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std_time,
                f'{avg_time:.3f}s',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add speedup annotation
    speedup = results['torch_cpp']['avg_time'] / results['cuda']['avg_time']
    ax.text(0.5, max(avg_times) * 0.5, 
            f'Speedup: {speedup:.2f}x',
            ha='center', va='center',
            fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'implementation_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    
    # Also create a detailed timing plot
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(epochs, results['torch_cpp']['epoch_times'], 'b-', label='Torch C++', 
             linewidth=2, alpha=0.7, marker='o', markersize=3)
    ax2.plot(epochs, results['cuda']['epoch_times'], 'r-', label='CUDA', 
             linewidth=2, alpha=0.7, marker='s', markersize=3)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Time (seconds)', fontsize=12)
    ax2.set_title('Per-Epoch Training Time Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timing_plot_path = os.path.join(output_dir, 'epoch_timing_comparison.png')
    plt.savefig(timing_plot_path, dpi=300, bbox_inches='tight')
    print(f"Timing plot saved to: {timing_plot_path}")
    
    print_summary(results)


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
    print(f"  Final op acc: {results['torch_cpp']['final_op_acc']:.2f}%")
    print(f"  Final val acc: {results['torch_cpp']['final_val_acc']:.2f}%")
    
    print(f"\nCUDA Implementation:")
    print(f"  Avg epoch time: {results['cuda']['avg_time']:.3f} ± {results['cuda']['std_time']:.3f}s")
    print(f"  Total time: {sum(results['cuda']['epoch_times']):.2f}s")
    print(f"  Final loss: {results['cuda']['final_loss']:.4f}")
    print(f"  Final op acc: {results['cuda']['final_op_acc']:.2f}%")
    print(f"  Final val acc: {results['cuda']['final_val_acc']:.2f}%")
    
    print(f"\nSpeedup: {speedup:.2f}x")
    print(f"Time saved per epoch: {results['torch_cpp']['avg_time'] - results['cuda']['avg_time']:.3f}s")
    print(f"Total time saved: {sum(results['torch_cpp']['epoch_times']) - sum(results['cuda']['epoch_times']):.2f}s")
    print(f"{'='*60}\n")


def save_results_to_file(results, output_dir):
    """Save results to a text file."""
    filepath = os.path.join(output_dir, 'results.txt')
    
    with open(filepath, 'w') as f:
        f.write("CUDA vs Torch C++ Implementation Comparison\n")
        f.write("=" * 60 + "\n\n")
        
        speedup = results['torch_cpp']['avg_time'] / results['cuda']['avg_time']
        
        f.write("PERFORMANCE SUMMARY\n")
        f.write("-" * 60 + "\n\n")
        
        f.write("Torch C++ Implementation:\n")
        f.write(f"  Avg epoch time: {results['torch_cpp']['avg_time']:.3f} ± {results['torch_cpp']['std_time']:.3f}s\n")
        f.write(f"  Total time: {sum(results['torch_cpp']['epoch_times']):.2f}s\n")
        f.write(f"  Final loss: {results['torch_cpp']['final_loss']:.4f}\n")
        f.write(f"  Final op acc: {results['torch_cpp']['final_op_acc']:.2f}%\n")
        f.write(f"  Final val acc: {results['torch_cpp']['final_val_acc']:.2f}%\n\n")
        
        f.write("CUDA Implementation:\n")
        f.write(f"  Avg epoch time: {results['cuda']['avg_time']:.3f} ± {results['cuda']['std_time']:.3f}s\n")
        f.write(f"  Total time: {sum(results['cuda']['epoch_times']):.2f}s\n")
        f.write(f"  Final loss: {results['cuda']['final_loss']:.4f}\n")
        f.write(f"  Final op acc: {results['cuda']['final_op_acc']:.2f}%\n")
        f.write(f"  Final val acc: {results['cuda']['final_val_acc']:.2f}%\n\n")
        
        f.write(f"Speedup: {speedup:.2f}x\n")
        f.write(f"Time saved per epoch: {results['torch_cpp']['avg_time'] - results['cuda']['avg_time']:.3f}s\n")
        f.write(f"Total time saved: {sum(results['torch_cpp']['epoch_times']) - sum(results['cuda']['epoch_times']):.2f}s\n\n")
        
        f.write("\nDETAILED RESULTS\n")
        f.write("-" * 60 + "\n\n")
        
        f.write("Epoch-by-epoch comparison:\n")
        f.write(f"{'Epoch':<8} {'Torch C++ Time':<15} {'CUDA Time':<15} {'Torch C++ Loss':<15} {'CUDA Loss':<15}\n")
        f.write("-" * 70 + "\n")
        
        for i in range(len(results['torch_cpp']['losses'])):
            f.write(f"{i+1:<8} {results['torch_cpp']['epoch_times'][i]:<15.3f} "
                   f"{results['cuda']['epoch_times'][i]:<15.3f} "
                   f"{results['torch_cpp']['losses'][i]:<15.4f} "
                   f"{results['cuda']['losses'][i]:<15.4f}\n")
    
    print(f"\nDetailed results saved to: {filepath}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare CUDA vs PyTorch implementation performance'
    )
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--hidden-dim', type=int, default=64,
                       help='Hidden dimension (default: 64)')
    parser.add_argument('--num-heads', type=int, default=4,
                       help='Number of attention heads (default: 4)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cpu', 'cuda'],
                       help='Device to use (default: cuda)')
    parser.add_argument('--modulo', type=int, default=19,
                       help='Modulo for arithmetic (default: 19)')
    parser.add_argument('--max-batches', type=int, default=None,
                       help='Max batches per epoch for quick testing')
    parser.add_argument('--warmup-epochs', type=int, default=2,
                       help='Number of warmup epochs before timing (default: 2)')
    parser.add_argument('--output-dir', type=str, default='./comparison_plots',
                       help='Directory to save plots (default: ./comparison_plots)')
    
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
        device=args.device,
        modulo=args.modulo,
        max_batches=args.max_batches,
        warmup_epochs=args.warmup_epochs
    )
    
    # Generate plots
    plot_comparison(results, output_dir=args.output_dir)
