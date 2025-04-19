import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import pytest

# run this test with: python -m pytest tests/test_cpp_backward_training.py -v -s
#forcing cpu for now

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from hyper_attn_cpp_wrapper import HypergraphAttentionCPP
except ImportError as e:
    print(f"Error importing HypergraphAttentionCPP: {e}")
    print("Please ensure the compiled extension and wrapper module are accessible.")
    pytest.skip("Skipping test due to import error.", allow_module_level=True)

def test_cpp_backward_pass_enables_training():
    """
    Tests if the custom C++ backward pass allows the model to train for a few steps.
    Checks if the loss decreases, indicating gradients are flowing correctly.
    """
    print("\\n=== Testing C++ Backward Pass in Training Loop ===")

    batch_size = 2
    seq_len = 4 
    d_model = 8 
    n_heads = 2 
    num_steps = 10 

    torch.manual_seed(42)

    # Device selection (prefer GPU/MPS if available, fallback to CPU)
    # if torch.cuda.is_available():
    #     device = torch.device('cuda')
    # elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    #     # Note: MPS support for custom C++ extensions might be limited/buggy
    #     # Fallback to CPU if issues arise
    #     try:
    #         # Test MPS allocation
    #         _ = torch.zeros(1, device='mps')
    #         device = torch.device('mps')
    #         print("Using MPS device.")
    #     except Exception as e:
    #         print(f"MPS device test failed ({e}), falling back to CPU.")
    #         device = torch.device('cpu')
    # else:
    #     device = torch.device('cpu')
    # print(f"Using device: {device}")

    # --- FORCE CPU FOR DEBUGGING ---
    device = torch.device('cpu')
    print(f"FORCING CPU for debugging. Using device: {device}")
    # --- END FORCE CPU ---


    try:
        model = HypergraphAttentionCPP(d_model, n_heads).to(device)
    except Exception as e:
        pytest.fail(f"Failed to initialize HypergraphAttentionCPP: {e}")

    x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True) #synthetic input
    target_y = torch.randn(batch_size, seq_len, d_model, device=device) #synthetic target

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    initial_loss = float('inf')
    final_loss = float('inf')
    gradients_present = False

    print("Starting mini training loop...")
    for step in range(num_steps):
        model.train() 
        optimizer.zero_grad()

        try:
            print("About to run forward pass...")
            print(f"Model: {model}")
            print(f"Input shape: {x.shape}, device: {x.device}, requires_grad: {x.requires_grad}")
            output = model(x)
        except Exception as e:
            pytest.fail(f"Forward pass failed during training step {step+1}: {e}")

        if output.shape != target_y.shape:
             pytest.fail(
                 f"Output shape mismatch: Expected {target_y.shape}, Got {output.shape}"
             )

        loss = criterion(output, target_y)

        if step == 0:
            initial_loss = loss.item()
            print(f"Step {step+1}/{num_steps}, Initial Loss: {initial_loss:.6f}")
        else:
             print(f"Step {step+1}/{num_steps}, Loss: {loss.item():.6f}")


        try:
            loss.backward()
        except Exception as e:
            pytest.fail(f"Backward pass failed during training step {step+1}: {e}")

        if step == 0 and model.Wo.weight.grad is not None:
             gradients_present = True
             print("Gradients successfully computed for model.Wo.weight.")
        elif step == 0:
             print("Warning: No gradients found for model.Wo.weight after first backward pass.")


        optimizer.step()

        if step == num_steps - 1:
            final_loss = loss.item()

    print("Training loop finished.")
    print(f"Initial Loss: {initial_loss:.6f}")
    print(f"Final Loss:   {final_loss:.6f}")

    assert gradients_present, "Gradients were not found in model parameters after backward()."
    assert final_loss < initial_loss * 0.9, \
        f"Loss did not decrease significantly. Initial: {initial_loss}, Final: {final_loss}"

    print("✓ Test Passed: Loss decreased, indicating gradients likely flowed correctly.")

if __name__ == "__main__":
    pytest.main([__file__]) # Run the test function using pytest 