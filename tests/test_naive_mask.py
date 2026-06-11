import torch
from att3ntion import _naive
from torch.autograd.functional import jacobian

def _test_jacobian(cls) -> bool:
	"""
	Computes jacobian of attention w.r.t. x, y.
	Tests that the marginal is nonzero exactly where a random mask is True
	"""
	B, C, M = 3, 20, 5
	x = torch.rand((B, C, M))
	mask_BCC = torch.randint(0, 2, (B, C, C)).to(torch.bool)
	mask_BCC[:,torch.arange(C),torch.arange(C)] = True
	attn = cls(d_model=M, n_heads=4)
	fwd = lambda x: attn(x, None, mask_BCC)
	jac_BCMBCM = jacobian(fwd, x) # Bo, Co, Mo, Bi, Ci, Mi

	# only compare pointwise batch elements x[b], y[b]
	jac_BCMCM = torch.einsum('bcmbdn -> bcmdn', jac_BCMBCM)

	# reduce over input-output channel combinations
	active_BCC = (jac_BCMCM != 0).any(axis=(2,4))

	return torch.all(active_BCC == mask_BCC)

@torch.compile
def test_HypergraphAttentionNaive():
	assert bool(_test_jacobian(_naive._HypergraphAttentionNaive))

def test_GraphAttentionNaive():
	assert bool(_test_jacobian(_naive._GraphAttentionNaive))

if __name__ == "__main__":
	torch.set_printoptions(linewidth=220, threshold=10000)
	test_HypergraphAttentionNaive()
	test_GraphAttentionNaive()

