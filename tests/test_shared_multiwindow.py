"""Exact training at all supported windows, including padded and masked rows."""
import pytest
import torch

from att3ntion._single_gather import window_mask
from att3ntion._single_gather_shared import single_gather_shared_attention
from test_single_gather_shared import NAMES, GRADS, draw, run_reference, check_all


@pytest.mark.parametrize('window', [16, 32, 64, 128])
@pytest.mark.parametrize('n', [17, 65, 129])
@pytest.mark.parametrize('group', [1, 2, 4])
def test_training_across_windows(window, n, group):
    xs, dy = draw(2, 8, n, seed=19)
    mask = window_mask(n, window, 'cuda')[None].expand(2, -1, -1).clone()
    mask[0, n // 2, :] = False
    mask[1, :, ::7] = False
    want_y, want_lse, want_g = run_reference(xs, dy, mask)
    leaves = {name: x.clone().requires_grad_(True) for name, x in xs.items()}
    y = single_gather_shared_attention(*[leaves[name] for name in NAMES], mask,
        fwd_group=group, rs_group=group, window=window)
    y.backward(dy)
    grads = {grad: leaves[name].grad for grad, name in zip(GRADS, NAMES)}
    check_all(y, grads, want_y, want_lse, want_g)
    assert tuple(y.shape) == (2, 8, n, 128)


@pytest.mark.parametrize('window', [16, 64, 128])
def test_empty_window_gradient_is_zero(window):
    xs, dy = draw(1, 8, 33, seed=7)
    leaves = [xs[name].clone().requires_grad_(True) for name in NAMES]
    y = single_gather_shared_attention(*leaves, torch.zeros(33, 33, device='cuda', dtype=torch.bool),
        fwd_group=2, rs_group=4, window=window)
    y.backward(dy)
    assert torch.count_nonzero(y) == 0
    assert all(torch.count_nonzero(x.grad) == 0 for x in leaves)
