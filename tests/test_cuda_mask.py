import pytest
import torch

from att3ntion import HypergraphAttention, _HypergraphAttentionNaive


def _build_pair(scatter=False, d_model=16, n_heads=4):
    cuda_mod = HypergraphAttention(
        d_model=d_model,
        n_heads=n_heads,
        dropout_rate=0.0,
        scatter=scatter,
    ).to(device="cuda", dtype=torch.float32)
    naive_mod = _HypergraphAttentionNaive(
        d_model=d_model,
        n_heads=n_heads,
        dropout_rate=0.0,
        head_subspaces=True,
        scatter=scatter,
        value_bias=True,
    ).to(device="cuda", dtype=torch.float32)
    naive_mod.load_state_dict(cuda_mod.state_dict())
    return cuda_mod, naive_mod


def _assert_forward_backward_close(cuda_mod, naive_mod, x, mask, atol=3e-2, rtol=3e-2):
    x_cuda = x.clone().detach().requires_grad_(True)
    x_naive = x.clone().detach().requires_grad_(True)

    y_cuda = cuda_mod(x_cuda, mask=mask)
    y_naive = naive_mod(x_naive, None, mask)
    assert torch.allclose(y_cuda, y_naive, atol=atol, rtol=rtol)

    grad_out = torch.randn_like(y_cuda)
    (y_cuda * grad_out).sum().backward()
    (y_naive * grad_out).sum().backward()

    assert torch.allclose(x_cuda.grad, x_naive.grad, atol=atol, rtol=rtol)
    for p_cuda, p_naive in zip(cuda_mod.parameters(), naive_mod.parameters()):
        if p_cuda.grad is None and p_naive.grad is None:
            continue
        assert p_cuda.grad is not None and p_naive.grad is not None
        assert torch.allclose(p_cuda.grad, p_naive.grad, atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("scatter", [False, True])
def test_cuda_mask_random_parity(scatter):
    torch.manual_seed(0)
    B, N, D, H = 2, 19, 64, 2
    cuda_mod, naive_mod = _build_pair(scatter=scatter, d_model=D, n_heads=H)
    x = torch.randn(B, N, D, device="cuda", dtype=torch.float32)
    mask = torch.randint(0, 2, (B, N, N), device="cuda", dtype=torch.bool)
    diag = torch.arange(N, device="cuda")
    mask[:, diag, diag] = True
    _assert_forward_backward_close(cuda_mod, naive_mod, x, mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_mask_broadcast_2d_matches_3d():
    torch.manual_seed(1)
    B, N, D, H = 2, 19, 64, 2
    cuda_mod, _ = _build_pair(scatter=True, d_model=D, n_heads=H)
    x = torch.randn(B, N, D, device="cuda", dtype=torch.float32)
    mask_2d = torch.tril(torch.ones(N, N, device="cuda", dtype=torch.bool))
    mask_3d = mask_2d.unsqueeze(0).expand(B, -1, -1).contiguous()
    y_2d = cuda_mod(x, mask=mask_2d)
    y_3d = cuda_mod(x, mask=mask_3d)
    assert torch.allclose(y_2d, y_3d, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_mask_all_true_matches_unmasked():
    torch.manual_seed(2)
    B, N, D, H = 2, 19, 64, 2
    cuda_mod, _ = _build_pair(scatter=False, d_model=D, n_heads=H)
    x = torch.randn(B, N, D, device="cuda", dtype=torch.float32)
    mask = torch.ones(B, N, N, device="cuda", dtype=torch.bool)
    y_masked = cuda_mod(x, mask=mask)
    y_unmasked = cuda_mod(x, mask=None)
    assert torch.allclose(y_masked, y_unmasked, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_mask_all_masked_row_is_finite_and_matches_naive():
    torch.manual_seed(3)
    B, N, D, H = 2, 19, 64, 2
    cuda_mod, naive_mod = _build_pair(scatter=True, d_model=D, n_heads=H)
    x = torch.randn(B, N, D, device="cuda", dtype=torch.float32)
    mask = torch.tril(torch.ones(B, N, N, device="cuda", dtype=torch.bool))
    mask[:, 5, :] = False
    y_cuda = cuda_mod(x, mask=mask)
    y_naive = naive_mod(x, None, mask)
    assert torch.isfinite(y_cuda).all()
    assert torch.isfinite(y_naive).all()


# The tests above run at d_head = 64/2 = 32, so they exercise only the scalar
# kernels. These use d_model=128 / n_heads=2 -> d_head=64, the one shape the
# tensor-core forward and backward accept, driving the masked TC path through
# the real autograd plumbing (padded mask, collapsed correction sums, ragged N).
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("N", [16, 19, 32, 64])
def test_cuda_mask_tc_head_dim_64_parity(N):
    torch.manual_seed(4 + N)
    B, D, H = 2, 128, 2
    cuda_mod, naive_mod = _build_pair(scatter=False, d_model=D, n_heads=H)
    x = torch.randn(B, N, D, device="cuda", dtype=torch.float32)
    mask = torch.tril(torch.ones(B, N, N, device="cuda", dtype=torch.bool))
    mask[:, 5, :] = False          # a fully masked query row
    _assert_forward_backward_close(cuda_mod, naive_mod, x, mask,
                                   atol=5e-2, rtol=5e-2)
