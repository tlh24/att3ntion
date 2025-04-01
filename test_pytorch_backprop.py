import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# import matplotlib.pyplot as plt
import copy
from hyper_attn_pytorch import HypergraphAttention

# Function to check gradient equivalence
def check_gradient_equivalence(model, batch_x, batch_y, atol=1e-5):
    """
    Check if manual backward gradients match PyTorch autograd.
    
    Returns:
        bool: True if gradients match within tolerance
    """
    model.eval()
    # Clone model to avoid affecting the original
    model_autograd = copy.deepcopy(model)
    
    # Make sure gradients are enabled for autograd version
    for param in model_autograd.parameters():
        param.requires_grad_(True)
    
    # AUTOGRAD VERSION
    model_autograd.zero_grad()
    output_autograd = model_autograd(batch_x)
    loss_autograd = nn.MSELoss()(output_autograd, batch_y)
    loss_autograd.backward()
    
    # Save autograd gradients
    autograd_grads = {}
    for name, param in model_autograd.named_parameters():
        if param.grad is not None:
            autograd_grads[name] = param.grad.detach().clone()
    
    # MANUAL VERSION
    model.zero_grad()
    output_manual = model(batch_x)
    loss_manual = nn.MSELoss()(output_manual, batch_y)
    dL_doutput = 2.0 * (output_manual - batch_y) / output_manual.numel()
    
    # Fix for dropout masks - create ones tensors with the same shape as Aq, Ar, As
    batch_size = batch_x.shape[0]
    ntok = batch_x.shape[1]
    n_heads = model.n_heads
    # These are the shapes needed for the dropout masks
    attn_shape = (batch_size, n_heads, ntok, ntok, ntok)
    
    # Add the dropout masks if they don't exist (workaround for the backward method)
    if not hasattr(model, 'dropout_mask_q'):
        model.dropout_mask_q = torch.ones(attn_shape)
    if not hasattr(model, 'dropout_mask_r'):
        model.dropout_mask_r = torch.ones(attn_shape)
    if not hasattr(model, 'dropout_mask_s'):
        model.dropout_mask_s = torch.ones(attn_shape)
    
    # Get manual gradients
    manual_grads = model.backward(batch_x, dL_doutput)
    
    # Compare gradients
    all_match = True
    print("\n=== Gradient Comparison ===")
    
    for name, param in model.named_parameters():
        if name == 'Wq.weight' and 'dWq' in manual_grads:
            manual_grad = manual_grads['dWq'].t()  # Transpose
        elif name == 'Wr.weight' and 'dWr' in manual_grads:
            manual_grad = manual_grads['dWr'].t()
        elif name == 'Ws.weight' and 'dWs' in manual_grads:
            manual_grad = manual_grads['dWs'].t()
        elif name == 'Wv_q.weight' and 'dWv_q' in manual_grads:
            manual_grad = manual_grads['dWv_q'].t()
        elif name == 'Wv_r.weight' and 'dWv_r' in manual_grads:
            manual_grad = manual_grads['dWv_r'].t()
        elif name == 'Wv_s.weight' and 'dWv_s' in manual_grads:
            manual_grad = manual_grads['dWv_s'].t()
        elif name == 'Wo.weight' and 'dWo' in manual_grads:
            manual_grad = manual_grads['dWo'].t()
        elif name == 'Wv_q.bias' and 'dWv_q_bias' in manual_grads:
            manual_grad = manual_grads['dWv_q_bias']
        elif name == 'Wv_r.bias' and 'dWv_r_bias' in manual_grads:
            manual_grad = manual_grads['dWv_r_bias']
        elif name == 'Wv_s.bias' and 'dWv_s_bias' in manual_grads:
            manual_grad = manual_grads['dWv_s_bias']
        elif name == 'Wo.bias' and 'dWo_bias' in manual_grads:
            manual_grad = manual_grads['dWo_bias']
        else:
            continue
        
        autograd_grad = autograd_grads[name]
        
        # Check shapes match
        if manual_grad.shape != autograd_grad.shape:
            print(f"{name}: Shape mismatch - Manual {manual_grad.shape} vs Autograd {autograd_grad.shape}")
            all_match = False
            continue
        
        # Calculate max absolute difference
        diff = (manual_grad - autograd_grad).abs().max().item()
        match = diff <= atol
        status = "✓" if match else "✗"
        
        print(f"{name}: max diff = {diff:.8f} {status}")
        if not match:
            all_match = False
    
    if all_match:
        print("All gradients match within tolerance! ✅")
    else:
        print("Some gradients don't match! ⚠️")
    
    return all_match

# Training function with gradient check and both methods
def train_and_compare(model, num_epochs=10, batch_size=32, learning_rate=0.001):
    """Train with both manual backward and autograd to compare"""
    # Create two identical models
    model_manual = copy.deepcopy(model)
    model_autograd = copy.deepcopy(model)
    model_manual.eval()
    model_autograd.eval()
    
    # Generate data with fixed seed
    torch.manual_seed(42)
    seq_len = 16
    input_dim = model.d_model
    num_samples = 1000
    
    X = torch.randn(num_samples, seq_len, input_dim)
    Y = X + 0.1 * torch.randn_like(X) + 0.5
    
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                            generator=torch.Generator().manual_seed(42))
    
    # Create optimizers
    opt_manual = optim.Adam(model_manual.parameters(), lr=learning_rate)
    opt_autograd = optim.Adam(model_autograd.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Lists to track metrics
    manual_losses = []
    autograd_losses = []
    
    # First batch for detailed gradient check
    first_check_done = False
    
    # Training loop
    for epoch in range(num_epochs):
        manual_epoch_loss = 0.0
        autograd_epoch_loss = 0.0
        
        for i, (batch_x, batch_y) in enumerate(dataloader):
            # AUTOGRAD VERSION
            opt_autograd.zero_grad()
            output_autograd = model_autograd(batch_x)
            loss_autograd = criterion(output_autograd, batch_y)
            loss_autograd.backward()
            opt_autograd.step()
            autograd_epoch_loss += loss_autograd.item()
            
            # MANUAL VERSION
            opt_manual.zero_grad()
            output_manual = model_manual(batch_x)
            loss_manual = criterion(output_manual, batch_y)
            
            # Compute gradient of loss
            dL_doutput = 2.0 * (output_manual - batch_y) / output_manual.numel()
            
            # Add dropout masks if not present
            batch_size = batch_x.shape[0]
            ntok = batch_x.shape[1]
            n_heads = model_manual.n_heads
            attn_shape = (batch_size, n_heads, ntok, ntok, ntok)
            
            if not hasattr(model_manual, 'dropout_mask_q'):
                model_manual.dropout_mask_q = torch.ones(attn_shape)
            if not hasattr(model_manual, 'dropout_mask_r'):
                model_manual.dropout_mask_r = torch.ones(attn_shape)
            if not hasattr(model_manual, 'dropout_mask_s'):
                model_manual.dropout_mask_s = torch.ones(attn_shape)
                
            # Call manual backward
            gradients = model_manual.backward(batch_x, dL_doutput)
            
            # Assign gradients (with transpose where needed)
            for name, param in model_manual.named_parameters():
                if name == 'Wq.weight':
                    param.grad = gradients['dWq'].t()
                elif name == 'Wr.weight':
                    param.grad = gradients['dWr'].t()
                elif name == 'Ws.weight':
                    param.grad = gradients['dWs'].t()
                elif name == 'Wv_q.weight':
                    param.grad = gradients['dWv_q'].t()
                elif name == 'Wv_r.weight':
                    param.grad = gradients['dWv_r'].t()
                elif name == 'Wv_s.weight':
                    param.grad = gradients['dWv_s'].t()
                elif name == 'Wo.weight':
                    param.grad = gradients['dWo'].t()
                elif name == 'Wv_q.bias' and 'dWv_q_bias' in gradients:
                    param.grad = gradients['dWv_q_bias']
                elif name == 'Wv_r.bias' and 'dWv_r_bias' in gradients:
                    param.grad = gradients['dWv_r_bias']
                elif name == 'Wv_s.bias' and 'dWv_s_bias' in gradients:
                    param.grad = gradients['dWv_s_bias']
                elif name == 'Wo.bias' and 'dWo_bias' in gradients:
                    param.grad = gradients['dWo_bias']
            
            opt_manual.step()
            manual_epoch_loss += loss_manual.item()
            
            # Perform detailed gradient check on the first batch of the first epoch
            if not first_check_done and i == 0 and epoch == 0:
                print("\n=== Detailed Gradient Check (First Batch) ===")
                check_model = copy.deepcopy(model)
                check_gradient_equivalence(check_model, batch_x, batch_y)
                first_check_done = True
        
        # Calculate average losses
        manual_avg_loss = manual_epoch_loss / len(dataloader)
        autograd_avg_loss = autograd_epoch_loss / len(dataloader)
        
        manual_losses.append(manual_avg_loss)
        autograd_losses.append(autograd_avg_loss)
        
        # Compare loss values
        loss_diff = abs(manual_avg_loss - autograd_avg_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'  Manual Loss: {manual_avg_loss:.6f}')
        print(f'  Autograd Loss: {autograd_avg_loss:.6f}')
        print(f'  Difference: {loss_diff:.6f}')
    
    # Plot loss curves
    # plt.figure(figsize=(12, 6))
    # plt.plot(manual_losses, label='Manual Backward')
    # plt.plot(autograd_losses, label='PyTorch Autograd')
    # plt.title('Training Loss Comparison')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig('loss_comparison.png')
    # plt.show()
    
    return {
        'manual_model': model_manual,
        'autograd_model': model_autograd,
        'manual_losses': manual_losses,
        'autograd_losses': autograd_losses
    }

# Example usage
if __name__ == "__main__":
    # Create model with smaller dimensions for faster testing
    model = HypergraphAttention(d_model=32, n_heads=2, dropout_rate=0.1)
    
    # Run training with both methods
    results = train_and_compare(model, num_epochs=5, batch_size=32, learning_rate=0.001)
    
    # You can also test gradient equivalence separately
    test_batch_x = torch.randn(4, 16, 32)
    test_batch_y = torch.randn(4, 16, 32)
    check_gradient_equivalence(copy.deepcopy(model), test_batch_x, test_batch_y) 