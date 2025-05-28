import torch
from torch.autograd import gradcheck
from hyper_attn_pytorch import HypergraphAttention_Naive
from hyper_attn_full_pytorch import HypergraphAttention
import pdb
import time
import matplotlib.pyplot as plt

def absum(a):
    return torch.sum(torch.abs(a))

device = torch.device("cpu")
torch.manual_seed(int(time.time()))

kwargs = {'dtype': torch.float64,
          'device': device}

debug = True

if debug:
	batch_size = 1 # unlikely to be bugs in this
	n_ctx = 3
	n_heads = 2
	width = 2
else:
	# batch_size = 2
	# n_ctx = 3
	# n_heads = 5
	# width = 7
	batch_size = 1
	n_ctx = 16
	n_heads = 8
	width = 64


x = torch.randn((batch_size, n_ctx, width), requires_grad=True, **kwargs)

ha_naive = HypergraphAttention_Naive(width, n_heads, **kwargs)
ha_baseline = HypergraphAttention(width, n_heads, **kwargs)

variables = [x]

attn_naive = ha_naive(x)
attn_baseline = ha_baseline(x)

# attn_baseline = python.l1attn_baseline.L1AttnFn.apply(*variables)
#
# attn_cpp = l1attn_cpp.L1AttnFn.apply(*variables)
#
# variables_cuda = [q.cuda(), k.cuda()]
# attn_cuda = l1attn_cuda.L1AttnFn.apply(*variables_cuda)

assert(torch.allclose(attn_naive, attn_baseline))
print('Forward: Baseline vs Naive Ok')

# assert(torch.allclose(attn_naive, attn_cpp))
# print('Forward: Cpp vs Naive Ok')

# print(attn_naive[0,:,:,0])
# print(attn_cuda.cpu()[0,:,:,0])
# fig,axs = plt.subplots(1, 3, figsize=(10,5))
# na = attn_naive.detach()[0,:,:,0]
# ca = attn_cuda.detach().cpu()[0,:,:,0]
# axs[0].imshow(attn_naive.detach()[0,:,:,0])
# axs[1].imshow(attn_cuda.detach().cpu()[0,:,:,0])
# axs[2].imshow(na / ca)
# plt.show()
# assert(torch.allclose(attn_naive, attn_cuda.cpu()))
# print('Forward: Cuda vs Naive Ok')

# # note: the gradient dosen't work perfectly with rational numbers (above)
# # because the gradient isn't defined for abs(0)
# # (which occurs with rational numbers)
# q = torch.randn(batch_size, n_ctx, n_heads, width, **kwargs)
# k = torch.randn(batch_size, n_ctx, n_heads, width, **kwargs)
# variables = [q, k]

if gradcheck(ha_baseline, variables, nondet_tol=1e-6):
    print('Backward: Baseline grad Ok')

# if gradcheck(l1attn_cpp.L1AttnFn.apply, variables, nondet_tol=1e-6):
#    print('Backward: Cpp grad Ok')
#
# if gradcheck(l1attn_cuda.L1AttnFn.apply, variables_cuda, nondet_tol=1e-6):
#    print('Backward: Cuda grad Ok')
