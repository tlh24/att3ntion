import pytest
import torch

from att3ntion import HypergraphAttention


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hypergraph_attention_compiles():
    model = HypergraphAttention(d_model=64, n_heads=4).cuda()
    compiled = torch.compile(model)

    x = torch.randn(2, 16, 64, device="cuda", requires_grad=True)
    y = compiled(x)
    y.sum().backward()

    assert y.shape == x.shape
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
