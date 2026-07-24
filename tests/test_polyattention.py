import pytest
import torch

from att3ntion import PolyAttention, SelfAttention


@pytest.mark.parametrize("polynomial", ["tree", "strassen", "tensor"])
@pytest.mark.parametrize("head_subspaces", [True, False])
def test_polyattention_t3_forward_backward(polynomial, head_subspaces):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, N, d_model, n_heads = 2, 16, 64, 4

    model = PolyAttention(
        d_model=d_model, n_heads=n_heads,
        head_subspaces=head_subspaces, polynomial=polynomial,
    ).to(device)

    x = torch.randn(B, N, d_model, device=device, requires_grad=True)
    y = model(x, rotary_emb=None)
    y.sum().backward()

    assert y.shape == x.shape
    assert x.grad is not None
    assert torch.isfinite(y).all()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("head_subspaces", [True, False])
def test_polystandard_t2_forward_backward(head_subspaces):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, N, d_model, n_heads = 2, 16, 64, 4

    model = SelfAttention(
        d_model=d_model,
        n_heads=n_heads,
        head_subspaces=head_subspaces,
    ).to(device)

    x = torch.randn(B, N, d_model, device=device, requires_grad=True)
    y = model(x, rotary_emb=None)
    y.sum().backward()

    assert y.shape == x.shape
    assert x.grad is not None
    assert torch.isfinite(y).all()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("polynomial", ["tree", "strassen", "tensor"])
def test_polyattention_t3_softmax_normalizes(polynomial):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, N, d_model, n_heads = 1, 8, 32, 2

    model = PolyAttention(
        d_model=d_model, n_heads=n_heads, head_subspaces=True, polynomial=polynomial,
    ).to(device)
    x = torch.randn(B, N, d_model, device=device).float()

    with torch.no_grad():
        Q = model.Wq(x).reshape(B, N, n_heads, model.d_head).permute(0, 2, 1, 3)
        R = model.Wr(x).reshape(B, N, n_heads, model.d_head).permute(0, 2, 1, 3)
        S = model.Ws(x).reshape(B, N, n_heads, model.d_head).permute(0, 2, 1, 3)
        logits = model._compute_logits(Q, R, S)
        P = torch.softmax(logits.flatten(3, 4), dim=-1).reshape(logits.shape)
        row_sums = P.sum(dim=(-1, -2))
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_polyattention_rejects_bad_polynomial():
    with pytest.raises(ValueError):
        PolyAttention(d_model=32, n_heads=2, polynomial="not_a_real_poly")
