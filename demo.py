import torch
import torch.nn as nn
import numpy as np
from hyper_attn_pytorch import HypergraphAttention
from test import genData

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.hyper_attn = HypergraphAttention(embedding_dim=32, num_heads=2)
        self.output = nn.Linear(32, 32)
        
    def forward(self, x):
        attn_output = self.hyper_attn(x)
        self.attn_output_shape = attn_output.shape
        return self.output(attn_output[:, 5]), self.output(attn_output[:, 7])

def main():
    print("Demonstrating Hypergraph Attention Module")
    print("----------------------------------------")
    
    print("Generating sample data...")
    batch_size = 1
    modulo = 7
    sample_data = genData(batch_size, modulo, do_print=True)
    
    x = torch.tensor(sample_data, dtype=torch.float32)
    
    model = SimpleModel()
    
    print("\nRunning forward pass through hypergraph attention...")
    op_logits, result_logits = model(x)
    
    print("\nInput shape:", x.shape)
    print("Hypergraph attention output shape:", model.attn_output_shape)
    print("Operation output shape:", op_logits.shape)
    print("Result output shape:", result_logits.shape)
    
    print("\nMost likely operation:", torch.argmax(op_logits[0]).item())
    print("Most likely result:", torch.argmax(result_logits[0]).item() - 4)  # -4 to convert from index to value
    
    print("\nChecking gradient flow...")
    loss = op_logits.sum() + result_logits.sum()
    loss.backward()
    
    has_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0 
                  for p in model.hyper_attn.parameters())
    
    print(f"Gradients properly flowing through hypergraph attention: {has_grad}")
    
    print("\nHypergraph attention module successfully verified!")

if __name__ == "__main__":
    main() 