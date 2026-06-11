import torch
import torch.nn as nn
import math

from att3ntion._autograd import QuickGELU


def _normalize_self_attn_mask(mask, batch_size, ntok, device):
	if mask is None:
		return None

	if mask.ndim == 2:
		if mask.shape != (ntok, ntok):
			raise ValueError(
				f"2D mask must have shape {(ntok, ntok)}, got {tuple(mask.shape)}"
			)
		mask = mask.unsqueeze(0)
	elif mask.ndim != 3:
		raise ValueError(f"mask must have ndim 2 or 3, got ndim={mask.ndim}")

	if mask.shape[-2:] != (ntok, ntok):
		raise ValueError(
			f"mask must have trailing shape {(ntok, ntok)}, got {tuple(mask.shape)}"
		)
	if mask.shape[0] not in (1, batch_size):
		raise ValueError(
			f"mask batch dim must be 1 or {batch_size}, got {mask.shape[0]}"
		)
	if mask.shape[0] == 1 and batch_size > 1:
		mask = mask.expand(batch_size, -1, -1)

	return mask.to(device=device, dtype=torch.bool)


class _HypergraphAttentionNaive(nn.Module):
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

		self.Wo = nn.Linear(self.d_model, d_model, bias=True, **kwargs)

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
				mask = mask[None,:,:]
			assert(mask.ndim == 3)
			valid3 = (mask[:,:,:,None] & mask[:,:,None,:]).flatten(2, 3)
			invalid3 = ~(valid3[:,None,:,:])
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


class PolyAttention(nn.Module):
	SUPPORTED_POLYNOMIALS = ("tree", "strassen", "tensor")

	def __init__(self, d_model, n_heads, dropout_rate=0, head_subspaces=False,
		polynomial="tree", **kwargs):
		super().__init__()

		if polynomial not in self.SUPPORTED_POLYNOMIALS:
			raise ValueError(
				f"polynomial must be one of {self.SUPPORTED_POLYNOMIALS}, got {polynomial!r}"
			)
		self.polynomial = polynomial

		self.d_model = d_model
		self.n_heads = n_heads
		if head_subspaces:
			self.d_head = d_model // n_heads
		else:
			self.d_head = d_model
		self.head_subspaces = head_subspaces
		self.d_val = self.d_head

		self.Wq = nn.Linear(d_model, self.d_head * n_heads, bias=False, **kwargs)
		self.Wr = nn.Linear(d_model, self.d_head * n_heads, bias=False, **kwargs)
		self.Ws = nn.Linear(d_model, self.d_head * n_heads, bias=False, **kwargs)

		self.Wv_r = nn.Linear(d_model, self.d_val * n_heads, bias=True, **kwargs)
		self.Wv_s = nn.Linear(d_model, self.d_val * n_heads, bias=True, **kwargs)

		self.Wo = nn.Linear(self.d_model, d_model, bias=True, **kwargs)

		self.dropout = nn.Dropout(dropout_rate)
		self.gelu = QuickGELU()

	def _compute_logits(self, Q, R, S):
		if self.polynomial == "tree":
			Aqr = torch.einsum('bhid,bhjd->bhij', Q, R)
			Ars = torch.einsum('bhjd,bhkd->bhjk', R, S)
			logits = Aqr.unsqueeze(-1) + Ars.unsqueeze(2)
		elif self.polynomial == "strassen":
			Aqr = torch.einsum('bhid,bhjd->bhij', Q, R)
			Ars = torch.einsum('bhjd,bhkd->bhjk', R, S)
			Aqs = torch.einsum('bhid,bhkd->bhik', Q, S)
			logits = Aqr.unsqueeze(-1) + Ars.unsqueeze(2) + Aqs.unsqueeze(3)
		elif self.polynomial == "tensor":
			logits = torch.einsum('bhid,bhjd,bhkd->bhijk', Q, R, S)
		else:
			raise RuntimeError(f"unreachable: bad polynomial {self.polynomial!r}")
		return logits / math.sqrt(self.d_head)

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

		Vr = self.Wv_r(x)
		Vs = self.Wv_s(x)

		Q = Q.reshape(batch_size, ntok, self.n_heads, self.d_head).permute(0, 2, 1, 3)
		R = R.reshape(batch_size, ntok, self.n_heads, self.d_head).permute(0, 2, 1, 3)
		S = S.reshape(batch_size, ntok, self.n_heads, self.d_head).permute(0, 2, 1, 3)
		Vr = Vr.reshape(batch_size, ntok, self.n_heads, self.d_val).permute(0, 2, 1, 3)
		Vs = Vs.reshape(batch_size, ntok, self.n_heads, self.d_val).permute(0, 2, 1, 3)

		logits = self._compute_logits(Q, R, S)
		flat_logits = logits.flatten(3, 4)

		mask = _normalize_self_attn_mask(mask, batch_size, ntok, x.device)
		if mask is not None:
			valid_pair = (mask[:, :, :, None] & mask[:, :, None, :]).flatten(2, 3)
			invalid_pair = (~valid_pair).unsqueeze(1)
			flat_logits = flat_logits.masked_fill(invalid_pair, float("-inf"))

		P = torch.softmax(flat_logits, dim=-1).reshape(logits.shape)
		P = torch.nan_to_num(P, nan=0.0)

		Y = torch.einsum('bhijk,bhjd,bhkd->bhid', P, Vr, Vs)
		Y = self.gelu(Y)

		if self.head_subspaces:
			y = Y.permute(0, 2, 1, 3).reshape(batch_size, ntok, self.d_model)
		else:
			y = Y.permute(0, 2, 1, 3).sum(dim=2).squeeze()
		y = self.Wo(y)
		return y.to(out_dtype)

	def calcFlops(self, x):
		bs, ntok, d_model = x.shape
		f = 0.0
		f += 3 * bs * ntok * d_model * self.n_heads * self.d_head * 2
		f += 2 * bs * ntok * d_model * self.n_heads * self.d_val * 2
		if self.polynomial == "tree":
			f += 2 * 2 * bs * self.n_heads * ntok * ntok * self.d_head
			f += bs * self.n_heads * ntok**3 * 2
		elif self.polynomial == "strassen":
			f += 3 * 2 * bs * self.n_heads * ntok * ntok * self.d_head
			f += bs * self.n_heads * ntok**3 * 3
		elif self.polynomial == "tensor":
			f += bs * self.n_heads * ntok**3 * self.d_head * 3
		f += bs * self.n_heads * ntok**3 * 3
		f += bs * self.n_heads * ntok**3 * self.d_val * 3
		f += bs * ntok * self.d_model * self.d_model * 2
		return f


class _GraphAttentionNaive(nn.Module):
	def __init__(
		self,
		d_model,
		n_heads,
		dropout_rate=0,
		head_subspaces=False,
		**kwargs,
	):
		super().__init__()

		self.d_model = d_model
		self.n_heads = n_heads
		if head_subspaces:
			self.d_head = d_model//n_heads
		else:
			self.d_head = d_model
		self.head_subspaces = head_subspaces
		self.d_val = self.d_head

		self.Wq = nn.Linear(d_model, self.d_head*n_heads, bias=False, **kwargs)
		self.Wk = nn.Linear(d_model, self.d_head*n_heads, bias=False, **kwargs)

		self.Wv = nn.Linear(d_model, self.d_head*n_heads, bias=True, **kwargs)

		self.Wo = nn.Linear(d_model, d_model, bias=True, **kwargs)

		self.dropout = nn.Dropout(dropout_rate)
		self.gelu = QuickGELU()

	def forward(self, x, rotary_emb, mask=None):
		out_dtype = x.dtype
		x = x.float()
		batch_size, ntok, d_model = x.shape

		if rotary_emb is not None:
			Q = rotary_emb.rotate_queries_or_keys(self.Wq(x))
			K = rotary_emb.rotate_queries_or_keys(self.Wk(x))
		else:
			Q = self.Wq(x)
			K = self.Wk(x)
		V = self.Wv(x)

		Q = Q.reshape(batch_size, ntok, self.n_heads, self.d_head).permute(0, 2, 1, 3)
		K = K.reshape(batch_size, ntok, self.n_heads, self.d_head).permute(0, 2, 1, 3)

		V = V.reshape(batch_size, ntok, self.n_heads, self.d_head).permute(0, 2, 1, 3)

		A = torch.einsum('bhid,bhjd->bhij', Q, K)
		if mask is not None:
			mask = mask[:,None,:,:] # unsqueeze head dimension
			invalid = ~mask
			A = A.masked_fill(invalid, -torch.inf)
		A = torch.softmax(A, dim=-1)
		A = torch.nan_to_num(A, nan=0.0) # needed if all positions of mask are False
		y = torch.einsum('bhij,bhjd->bhid', A, V)

		if self.head_subspaces:
			y = y.permute(0,2,1,3)
			y = y.reshape(batch_size, ntok, d_model)
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


def SelfAttention(d_model, n_heads, dropout_rate=0, head_subspaces=False, **kwargs):
	return _GraphAttentionNaive(
		d_model=d_model,
		n_heads=n_heads,
		dropout_rate=dropout_rate,
		head_subspaces=head_subspaces,
		**kwargs,
	)


_PolyAttentionNaive = PolyAttention
_PolyStandardAttentionNaive = SelfAttention
