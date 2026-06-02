import torch
import torch.nn as nn
import math
import pdb
import numpy as np

from att3ntion._autograd import QuickGELU


class _HypergraphAttentionNaive(nn.Module):
	"""Pure-PyTorch naive O(N^3) implementation for correctness testing."""
	def __init__(
			self, d_model, n_heads, dropout_rate=0,
			head_subspaces=False,
			scatter=False,
			qrs_bias=False,
			value_bias=False,
			out_bias=False,
			**kwargs):
		super().__init__()

		self.d_model = d_model
		self.n_heads = n_heads
		if head_subspaces:
			self.d_head = d_model//n_heads
		else:
			self.d_head = d_model
		self.head_subspaces = head_subspaces
		self.d_val = self.d_head*1
		self.scatter = scatter

		self.Wq = nn.Linear(d_model, self.d_head*n_heads, bias=qrs_bias, **kwargs)
		self.Wr = nn.Linear(d_model, self.d_head*n_heads, bias=qrs_bias, **kwargs)
		self.Ws = nn.Linear(d_model, self.d_head*n_heads, bias=qrs_bias, **kwargs)

		value_proj_multiplier = 2 if self.scatter else 1
		self.Wv_q = nn.Linear(d_model, self.d_val*n_heads*value_proj_multiplier, bias=value_bias, **kwargs)
		self.Wv_r = nn.Linear(d_model, self.d_val*n_heads*value_proj_multiplier, bias=value_bias, **kwargs)
		self.Wv_s = nn.Linear(d_model, self.d_val*n_heads*value_proj_multiplier, bias=value_bias, **kwargs)

		self.Wo = nn.Linear(self.d_model, d_model, bias=out_bias, **kwargs)
		nn.init.normal_(self.Wo.weight, std=1.0 / np.sqrt(d_model))

		self.dropout = nn.Dropout(dropout_rate)
		self.gelu = QuickGELU()

	def forward(self, x, rotary_emb, mask=None):
		out_dtype = x.dtype
		x = x.float()
		batch_size, ntok, d_model = x.shape

		if rotary_emb is not None:
			Q = rotary_emb.rotate_queries_or_keys(self.Wq(x))
			R = rotary_emb.rotate_queries_or_keys(self.Wr(x))
			S = rotary_emb.rotate_queries_or_keys(self.Ws(x))
		else:
			Q = self.Wq(x)
			R = self.Wr(x)
			S = self.Ws(x)

		Q = Q.reshape(batch_size, ntok, self.n_heads, self.d_head).permute(0, 2, 1, 3)
		R = R.reshape(batch_size, ntok, self.n_heads, self.d_head).permute(0, 2, 1, 3)
		S = S.reshape(batch_size, ntok, self.n_heads, self.d_head).permute(0, 2, 1, 3)

		if self.scatter:
			# split the values into scatter and gather components
			Vq_full = self.Wv_q(x)
			Vr_full = self.Wv_r(x)
			Vs_full = self.Wv_s(x)
			Vq, Vq_ = Vq_full.reshape(batch_size, ntok, self.n_heads, self.d_val*2).permute(0, 2, 1, 3).split(self.d_val, dim=-1)
			Vr, Vr_ = Vr_full.reshape(batch_size, ntok, self.n_heads, self.d_val*2).permute(0, 2, 1, 3).split(self.d_val, dim=-1)
			Vs, Vs_ = Vs_full.reshape(batch_size, ntok, self.n_heads, self.d_val*2).permute(0, 2, 1, 3).split(self.d_val, dim=-1)
		else:
			Vq = self.Wv_q(x).reshape(batch_size, ntok, self.n_heads, self.d_val).permute(0, 2, 1, 3)
			Vr = self.Wv_r(x).reshape(batch_size, ntok, self.n_heads, self.d_val).permute(0, 2, 1, 3)
			Vs = self.Wv_s(x).reshape(batch_size, ntok, self.n_heads, self.d_val).permute(0, 2, 1, 3)

		dot_product = torch.einsum('bhid,bhjd,bhkd->bhijk', Q, R, S)
		dot_product = dot_product / (math.sqrt(self.d_head))

		dot_product_q = dot_product.flatten(3, 4) # BHI(JK)
		dot_product_r = dot_product.permute(0, 1, 3, 2, 4).flatten(3, 4) # BHJ(IK)
		dot_product_s = dot_product.permute(0, 1, 4, 2, 3).flatten(3, 4) # BHK(IJ)

		if mask is not None:
			if mask.ndim == 2:
				# mask is the standard 2D matrix (ntok, ntok)
				# add in a (dummy, broadcasted) batch dim
				mask = mask[None,:,:]
			# otherwise can have a different mask per batch element
			assert(mask.ndim == 3)
			valid = mask > 0 # convert to boolean (byte)
			# any i can attend to j,k <= i (likewise for the other 2 permutations).
			valid3 = (valid[:,:,:,None] & valid[:,:,None,:]).flatten(2, 3)
			invalid3 = ~(valid3[:,None,:,:])
			# the three dot_products are permuted and flattened so that the last dim is the softmax dim.
			dot_product_q = dot_product_q.masked_fill(invalid3, float('-inf'))
			dot_product_r = dot_product_r.masked_fill(invalid3, float('-inf'))
			dot_product_s = dot_product_s.masked_fill(invalid3, float('-inf'))

		Aq = torch.softmax(dot_product_q, dim=-1).reshape(dot_product.shape)
		Aq = torch.nan_to_num(Aq, nan=0.0)

		Ar = torch.softmax(dot_product_r, dim=-1).reshape(dot_product.shape)
		Ar = Ar.permute(0, 1, 3, 2, 4)
		Ar = torch.nan_to_num(Ar, nan=0.0)

		As = torch.softmax(dot_product_s, dim=-1).reshape(dot_product.shape)
		As = As.permute(0, 1, 3, 4, 2)
		As = torch.nan_to_num(As, nan=0.0)

		Y_q = torch.einsum('bhijk,bhjd,bhkd->bhid', Aq, Vr, Vs)
		Y_r = torch.einsum('bhijk,bhid,bhkd->bhjd', Ar, Vq, Vs)
		Y_s = torch.einsum('bhijk,bhid,bhjd->bhkd', As, Vq, Vr)
		Y_q = self.gelu(Y_q)
		Y_r = self.gelu(Y_r)
		Y_s = self.gelu(Y_s)
		y = Y_q + Y_r + Y_s

		if self.scatter:
			# note: option for diamond op in scatter being 'add' removed.
			# (see README.md)
			Y_q_ = torch.einsum('bhijk,bhjd,bhijk,bhkd->bhid', Ar, Vr_, As, Vs_)
			Y_r_ = torch.einsum('bhijk,bhid,bhijk,bhkd->bhjd', Aq, Vq_, As, Vs_)
			Y_s_ = torch.einsum('bhijk,bhid,bhijk,bhjd->bhkd', Aq, Vq_, Ar, Vr_)
			Y_q_ = self.gelu(Y_q_)
			Y_r_ = self.gelu(Y_r_)
			Y_s_ = self.gelu(Y_s_)
			y = y + Y_q_ + Y_r_ + Y_s_

		if self.head_subspaces:
			y = y.permute(0, 2, 1, 3).reshape(batch_size, ntok, self.d_model)
		else:
			y = y.permute(0, 2, 1, 3).sum(dim=2).squeeze()
		y = self.Wo(y)
		return y.to(out_dtype)

	def calcFlops(self, x):
		bs, ntok, d_model = x.shape
		f = 0.0
		f += 3 * bs * ntok * d_model**2 * self.n_heads*d_model
		f += 3 * bs * ntok * d_model**2 * self.n_heads*d_model*2
		f += bs * self.n_heads * ntok**3 * d_model * 2
		f += bs * self.n_heads * ntok**3 * 2 * 3
		f += bs * self.n_heads * ntok**3 * d_model * 3
		f += bs * self.n_heads * ntok**3 * d_model * 3 * 3
		f += bs * self.n_heads * ntok * d_model * (6 + 6)
		f += bs * self.n_heads * ntok * d_model**2
		return f


class _GraphAttentionNaive(nn.Module):
	"""Pure-PyTorch naive standard 2-way attention for comparison testing."""
	def __init__(self, d_model, n_heads, dropout_rate=0, head_subspaces=False, **kwargs):
		super().__init__()

		self.d_model = d_model
		self.n_heads = n_heads
		if head_subspaces:
			self.d_head = d_model//n_heads
		else:
			self.d_head = d_model
		self.head_subspaces = head_subspaces

		self.Wq = nn.Linear(d_model, self.d_head*n_heads, bias=False, **kwargs)
		self.Wk = nn.Linear(d_model, self.d_head*n_heads, bias=False, **kwargs)
		self.Wv = nn.Linear(d_model, self.d_head*n_heads, bias=False, **kwargs)
		self.Wo = nn.Linear(d_model, d_model, bias=False, **kwargs)
		nn.init.normal_(self.Wo.weight, std=1.0 / np.sqrt(d_model))

		self.dropout = nn.Dropout(dropout_rate)
		self.gelu = QuickGELU()

	def forward(self, x, rotary_emb, mask=None):
		out_dtype = x.dtype
		x = x.float()
		B, ntok, d_model = x.shape
		q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)

		def split_heads(t):
			return t.view(B, ntok, self.n_heads, self.d_head).transpose(1, 2)
			# shape (B, self.n_heads, ntok, self.d_head)
		q, k, v = split_heads(q), split_heads(k), split_heads(v)

		if rotary_emb is not None:
			q = rotary_emb.rotate_queries_or_keys(q)
			k = rotary_emb.rotate_queries_or_keys(k)

		A = torch.einsum('bhid,bhjd->bhij', q, k) / np.sqrt(self.d_head)
		if mask is not None:  # causal attention
			invalid = mask <= 0
			# this should broadcast properly if mask.ndim is 3 or 2
			A = A.masked_fill(invalid, -torch.inf)
		A = torch.softmax(A, dim=-1)
		y = torch.einsum('bhij,bhjd->bhid', A, v)

		if self.head_subspaces:
			y = y.permute(0,2,1,3)
			y = y.reshape(B, ntok, d_model)
		else:
			y = y.permute(0, 2, 3, 1).sum(dim=3).squeeze()
		y = self.gelu(y)
		y = self.Wo(y)
		return y.to(out_dtype)

	def calcFlops(self, x):
		bs, ntok, d_model = x.shape
		f = 0.0
		f += 3 * bs * ntok * d_model**2 * self.n_heads*d_model
		f += bs * self.n_heads * ntok**2 * d_model
		f += bs * self.n_heads * ntok**2 * 2
		f += bs * self.n_heads * ntok * d_model**2 * 2
		f += bs * self.n_heads * ntok * d_model * (2 + 6)
		f += bs * ntok * d_model**2
		return f
