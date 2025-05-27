import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import time

# Add the parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from hyper_attn_cpp_wrapper import HypergraphAttentionCPP
except ImportError as e:
    print(f"Error importing HypergraphAttentionCPP: {e}")
    print("Please ensure the compiled extension and wrapper module are accessible.")
    sys.exit(1)

def train_and_visualize():
    """
    Extended training script with detailed logging
    """
    print("\n" + "="*50)
    print("HYPERGRAPH ATTENTION C++ EXTENSION TRAINING MONITOR")
    print("="*50)

    # Training parameters
    batch_size = 2
    seq_len = 4 
    d_model = 8 
    n_heads = 2 
    num_steps = 50  # Increased for better visualization
    log_interval = 1  # Log detailed metrics every N steps
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Device selection with detailed info
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(0)
        print(f"🚀 CUDA GPU detected: {device_name}")
    else:
        device = torch.device('cpu')
        
    print(f"🔧 Training device: {device}")

    # Initialize model with error handling and info
    try:
        print("\n📦 Initializing HypergraphAttentionCPP model...")
        start_time = time.time()
        model = HypergraphAttentionCPP(d_model, n_heads).to(device)
        init_time = time.time() - start_time
        print(f"✅ Model initialized in {init_time:.4f}s")
        print(f"📊 Model configuration: d_model={d_model}, n_heads={n_heads}")
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        print(f"📏 Total parameters: {param_count:,}")
        
        # Print model structure
        print("\n📋 Model Structure:")
        for name, param in model.named_parameters():
            print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        sys.exit(1)

    # Create synthetic data
    print("\n🔢 Generating synthetic training data...")
    x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    target_y = torch.randn(batch_size, seq_len, d_model, device=device)
    print(f"   Input shape: {tuple(x.shape)}, device: {x.device}, dtype: {x.dtype}")
    print(f"   Target shape: {tuple(target_y.shape)}, device: {target_y.device}, dtype: {target_y.dtype}")

    # Setup loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    print(f"📈 Using optimizer: {optimizer.__class__.__name__} with lr=0.01")

    # For tracking training progress
    losses = []
    grad_norms = []
    
    # Print header for training log
    print("\n" + "="*50)
    print("🏃 BEGINNING TRAINING LOOP")
    print("="*50)
    print(f"{'Step':^6} | {'Loss':^12} | {'Improv %':^9} | {'Grad Norm':^12} | {'Step Time (ms)':^14} | {'Fwd (ms)':^10} | {'Bwd (ms)':^10}")
    print("-" * 90)
    
    start_time = time.time()
    initial_loss = None
    
    for step in range(num_steps):
        step_start_time = time.time()
        
        # Training mode
        model.train()
        optimizer.zero_grad()
        
        # Forward pass with timing and error handling
        try:
            forward_start = time.time()
            # model(x) should now return a single tensor after internal summing
            output = model(x) 
            forward_time = time.time() - forward_start
        except Exception as e:
            print(f"❌ Forward pass failed on step {step+1}: {e}")
            print(f"Input tensor info: shape={x.shape}, device={x.device}, dtype={x.dtype}")
            print("Exiting training loop.")
            break
            
        # Check output shape
        if output.shape != target_y.shape:
            print(f"⚠️ Output shape mismatch: Expected {target_y.shape}, Got {output.shape}")
        
        # Calculate loss
        loss = criterion(output, target_y)
        current_loss = loss.item()
        losses.append(current_loss)
        
        if initial_loss is None:
            initial_loss = current_loss
            
        # Backward pass with timing and error handling
        try:
            backward_start = time.time()
            loss.backward()
            backward_time = time.time() - backward_start
        except Exception as e:
            print(f"❌ Backward pass failed on step {step+1}: {e}")
            print(f"Error details: {str(e)}")
            print("Exiting training loop.")
            break
        
        # Calculate gradient norms for monitoring
        total_grad_norm = 0.0
        max_grad_param = None
        max_grad_value = 0.0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_grad_norm += param_norm**2
                
                # Track parameter with largest gradient for debugging
                if param_norm > max_grad_value:
                    max_grad_value = param_norm
                    max_grad_param = name
        
        grad_norm = total_grad_norm ** 0.5
        grad_norms.append(grad_norm)
        
        # Optimizer step
        optimizer.step()
        step_time = time.time() - step_start_time
        
        # Log progress
        if step % log_interval == 0 or step == num_steps - 1:
            improvement = (initial_loss - current_loss) / initial_loss * 100
            print(f"{step+1:6d} | {current_loss:12.6f} | {improvement:9.2f} | {grad_norm:12.6f} | {step_time*1000:14.2f} | {forward_time*1000:10.2f} | {backward_time*1000:10.2f}")
            
            # Check for NaN or inf values
            if current_loss != current_loss or abs(current_loss) == float('inf'):  # NaN or inf check
                print("⚠️ NaN/Inf detected in loss! Stopping training.")
                break
        
        # Detailed gradient analysis every 10 steps
        if step % 10 == 0:
            print(f"\n📊 Gradient Analysis at Step {step+1}:")
            print(f"  Largest gradient: {max_grad_param} = {max_grad_value:.6f}")
            print("  Parameter gradients:")
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_min = param.grad.min().item()
                    grad_max = param.grad.max().item()
                    grad_mean = param.grad.mean().item()
                    grad_norm = param.grad.norm().item()
                    print(f"    {name}: min={grad_min:.6f}, max={grad_max:.6f}, mean={grad_mean:.6f}, norm={grad_norm:.6f}")
            
            # Check model output statistics
            with torch.no_grad():
                output_min = output.min().item()
                output_max = output.max().item()
                output_mean = output.mean().item()
                output_std = output.std().item()
                print(f"  Output statistics: min={output_min:.4f}, max={output_max:.4f}, mean={output_mean:.4f}, std={output_std:.4f}")
                
    total_time = time.time() - start_time
    
    # Final report
    print("\n" + "="*50)
    print("🏁 TRAINING SUMMARY")
    print("="*50)
    print(f"Total training time: {total_time:.2f}s")
    
    if len(losses) > 1:
        print(f"Initial loss: {losses[0]:.6f}")
        print(f"Final loss: {losses[-1]:.6f}")
        improvement = (losses[0] - losses[-1]) / losses[0] * 100
        print(f"Overall improvement: {improvement:.2f}%")
        
        # Print loss progression
        print("\n📉 Loss Progression (first/last 5 steps + min/max):")
        # First 5
        for i in range(min(5, len(losses))):
            print(f"  Step {i+1}: {losses[i]:.6f}")
        
        # Add ellipsis if more than 10 steps
        if len(losses) > 10:
            print("  ...")
        
        # Last 5
        for i in range(max(5, len(losses)-5), len(losses)):
            print(f"  Step {i+1}: {losses[i]:.6f}")
            
        # Min/max loss
        min_loss = min(losses)
        min_idx = losses.index(min_loss)
        max_loss = max(losses)
        max_idx = losses.index(max_loss)
        print(f"\n  Minimum loss: {min_loss:.6f} (step {min_idx+1})")
        print(f"  Maximum loss: {max_loss:.6f} (step {max_idx+1})")
        
        # Gradient norm summary
        if grad_norms:
            print("\n📊 Gradient Norm Summary:")
            print(f"  Initial: {grad_norms[0]:.6f}")
            print(f"  Final: {grad_norms[-1]:.6f}")
            print(f"  Min: {min(grad_norms):.6f}")
            print(f"  Max: {max(grad_norms):.6f}")
            print(f"  Mean: {sum(grad_norms)/len(grad_norms):.6f}")
        
        if improvement < 10:
            print("\n⚠️ Warning: Loss didn't decrease significantly. Model may not be learning properly.")
        else:
            print("\n✅ Loss decreased successfully! Gradients appear to be flowing correctly.")
    
    # Final model inspection
    print("\n🔍 Final Model Parameter Statistics:")
    for name, param in model.named_parameters():
        with torch.no_grad():
            param_min = param.min().item()
            param_max = param.max().item()
            param_mean = param.mean().item()
            param_std = param.std().item()
            print(f"  {name}: min={param_min:.4f}, max={param_max:.4f}, mean={param_mean:.4f}, std={param_std:.4f}")
    
    return losses, grad_norms

if __name__ == "__main__":
    train_and_visualize() 