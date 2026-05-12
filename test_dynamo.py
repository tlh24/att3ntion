import torch
from att3ntion._autograd import HypergraphAttention

model = HypergraphAttention(d_model=64, n_heads=4).cuda()
model = torch.compile(model)

x = torch.randn(2, 16, 64, device='cuda', requires_grad=True)
y = model(x)
y.sum().backward()
print("Done")
