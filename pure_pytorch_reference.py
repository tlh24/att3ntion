import torch
import torch.nn as nn
import math

# this is the "naive" implementation
# which uses autograd

class QuickGELU(nn.Module):
	def forward(self, x: torch.Tensor):
		return x * torch.sigmoid(1.702 * x)

class HypergraphAttention_Naive(nn.Module):
	def __init__(self, d_model, n_heads, dropout_rate=0, head_subspaces=False, **kwargs):
		super(HypergraphAttention_Naive, self).__init__()

		# torch.manual_seed(42)
		
		# as with other small transformers, there are no head sub-spaces.
		# Really need to test if this is necessary! 
		
		self.d_model = d_model
		self.n_heads = n_heads
		if head_subspaces:
			self.d_head = d_model//n_heads
		else:
			self.d_head = d_model
		self.head_subspaces = head_subspaces
		self.d_val = self.d_head*1 # expansion!
		
		self.Wq = nn.Linear(d_model, self.d_head*n_heads, bias=False, **kwargs)
		self.Wr = nn.Linear(d_model, self.d_head*n_heads, bias=False, **kwargs)
		self.Ws = nn.Linear(d_model, self.d_head*n_heads, bias=False, **kwargs)
		
		self.Wv_q = nn.Linear(d_model, self.d_val*n_heads*2, bias=True, **kwargs)
		self.Wv_r = nn.Linear(d_model, self.d_val*n_heads*2, bias=True, **kwargs)
		self.Wv_s = nn.Linear(d_model, self.d_val*n_heads*2, bias=True, **kwargs)
		
		self.Wo = nn.Linear(self.d_model, d_model, bias=True, **kwargs)
		
		self.dropout = nn.Dropout(dropout_rate)
		self.gelu = QuickGELU()
		
	def forward(self, x, rotary_emb):
		batch_size, ntok, d_model = x.shape
		
		if rotary_emb is not None:
			Q = rotary_emb.rotate_queries_or_keys(self.Wq(x))
			R = rotary_emb.rotate_queries_or_keys(self.Wr(x))
			S = rotary_emb.rotate_queries_or_keys(self.Ws(x))
		else:
			Q = self.Wq(x)
			R = self.Wr(x)
			S = self.Ws(x)
		
		Vq = self.Wv_q(x)
		Vr = self.Wv_r(x)
		Vs = self.Wv_s(x)
		
		Q = Q.reshape(batch_size, ntok, self.n_heads, self.d_head).permute(0, 2, 1, 3)
		R = R.reshape(batch_size, ntok, self.n_heads, self.d_head).permute(0, 2, 1, 3)
		S = S.reshape(batch_size, ntok, self.n_heads, self.d_head).permute(0, 2, 1, 3)
		# Q,R,S are hence [batch_size, n_heads, ntok, d_head]
		
		# Gather & scatter value projection
		Vq, Vq_ = Vq.reshape(batch_size, ntok, self.n_heads, self.d_val*2).permute(0, 2, 1, 3).split(self.d_val, dim=-1)
		Vr, Vr_ = Vr.reshape(batch_size, ntok, self.n_heads, self.d_val*2).permute(0, 2, 1, 3).split(self.d_val, dim=-1)
		Vs, Vs_ = Vs.reshape(batch_size, ntok, self.n_heads, self.d_val*2).permute(0, 2, 1, 3).split(self.d_val, dim=-1)
		# Vq,Vr,Vs also [batch_size, n_heads, ntok, d_val]
		
		# compute 3-way attention scores of shape [b, h, i, j, k]
		dot_product = torch.einsum('bhid,bhjd,bhkd->bhijk', Q, R, S)
		dot_product = dot_product / (math.sqrt(self.d_head))
		
		# Compute attention weights for each position
		# Aq - gathering to position i (softmax over j,k)
		dot_product_q = dot_product
		Aq = torch.softmax(dot_product_q.flatten(3, 4), dim=-1).reshape(dot_product.shape)
		
		# Ar - gathering to position j (softmax over i,k)
		dot_product_r = dot_product.permute(0, 1, 3, 2, 4) # [b, h, j, i, k]
		Ar = torch.softmax(dot_product_r.flatten(3, 4), dim=-1).reshape(dot_product.shape)
		Ar = Ar.permute(0, 1, 3, 2, 4)  
		
		# As - gathering to position k (softmax over i,j)
		dot_product_s = dot_product.permute(0, 1, 4, 2, 3)  # [b, h, k, i, j]
		As = torch.softmax(dot_product_s.flatten(3, 4), dim=-1).reshape(dot_product.shape)
		As = As.permute(0, 1, 3, 4, 2) 
		
		# Aq = self.dropout(Aq)
		# Ar = self.dropout(Ar)
		# As = self.dropout(As)
		# No dropout for testing backprop
		# self.dropout_mask_q = torch.ones_like(Aq)
		# self.dropout_mask_r = torch.ones_like(Ar)
		# self.dropout_mask_s = torch.ones_like(As)
		gather = True
		scatter = False
		# Gather operations
		# hence the 'diamond' operation is multiply.
		if gather:
			Y_q = torch.einsum('bhijk,bhjd,bhkd->bhid', Aq, Vr, Vs)
			Y_r = torch.einsum('bhijk,bhid,bhkd->bhjd', Ar, Vq, Vs)
			Y_s = torch.einsum('bhijk,bhid,bhjd->bhkd', As, Vq, Vr)
			Y_q = self.gelu(Y_q) # test!
			Y_r = self.gelu(Y_r) # is this better?
			Y_s = self.gelu(Y_s) # doesn't seem like much.
		
		if scatter:
			# Scatter operations
			if False:
				# NOTE optional softmax-transpose: redo A_q, A_r, A_s
				Aq = torch.softmax(dot_product, dim=2) # softmax over i
				Ar = torch.softmax(dot_product, dim=3) # softmax over j
				As = torch.softmax(dot_product, dim=4) # softmax over k

			if False:
			# 'diamond' op is add; this seems very slightly slower to converge
			# in some tasks, it does not converge.
				Y_q_ = torch.einsum('bhijk,bhjd->bhid', Ar, Vr_) + \
						torch.einsum('bhijk,bhkd->bhid', As, Vs_)
				Y_r_ = torch.einsum('bhijk,bhid->bhjd', Aq, Vq_) + \
						torch.einsum('bhijk,bhkd->bhjd', As, Vs_)
				Y_s_ = torch.einsum('bhijk,bhid->bhkd', Aq, Vq_) + \
						torch.einsum('bhijk,bhjd->bhkd', Ar, Vr_)
			else:
				# 'diamond' op is mul
				Y_q_ = torch.einsum('bhijk,bhjd,bhijk,bhkd->bhid', Ar, Vr_, As, Vs_)
				Y_r_ = torch.einsum('bhijk,bhid,bhijk,bhkd->bhjd', Aq, Vq_, As, Vs_)
				Y_s_ = torch.einsum('bhijk,bhid,bhijk,bhjd->bhkd', Aq, Vq_, Ar, Vr_)

			Y_q_ = self.gelu(Y_q_)
			Y_r_ = self.gelu(Y_r_)
			Y_s_ = self.gelu(Y_s_)
			if gather:
				y = Y_q + Y_r + Y_s + Y_q_ + Y_r_ + Y_s_
			else:
				y = Y_q_ + Y_r_ + Y_s_

		else:
			Y_q = self.gelu(Y_q) # test!
			Y_r = self.gelu(Y_r) # is this better?
			Y_s = self.gelu(Y_s) # doesn't seem like much.
			y = Y_q + Y_r + Y_s
		
		# Handle heads based on head_subspaces setting
		if self.head_subspaces:
			# Concatenate heads (standard multi-head attention)
			y = y.permute(0, 2, 1, 3).reshape(batch_size, ntok, self.d_model)
		else:
			# Sum over heads
			y = y.permute(0, 2, 1, 3).sum(dim=2).squeeze()
		y = self.Wo(y)
		# residual path is external to this layer.
		return y 

	def calcFlops(self, x):
		bs, ntok, d_model = x.shape
		f = 0.0
		# QRS proj = [..., d_model] @ [d_model, n_heads*d_model]
		f += 3 * bs * ntok * d_model**2 * self.n_heads*d_model
		# Value proj = [..., d_model] @ [d_model, n_heads*d_model*2]
		f += 3 * bs * ntok * d_model**2 * self.n_heads*d_model*2
		# attn - '2' is from the 3-way multiply
		f += bs * self.n_heads * ntok**3 * d_model * 2
		# 3 softmaxes - 2 is from the softmax itself.
		f += bs * self.n_heads * ntok**3 * 2 * 3
		# gather
		f += bs * self.n_heads * ntok**3 * d_model * 3
		# scatter - 3 comes from the 4 arguments to each
		f += bs * self.n_heads * ntok**3 * d_model * 3 * 3
		# combine & Gelu
		f += bs * self.n_heads * ntok * d_model * (6 + 6)
		# Wo = [..., d_model] @ [d_model, d_model]
		f += bs * self.n_heads * ntok * d_model**2

		return f

class GraphAttention_Naive(nn.Module):
	def __init__(self, d_model, n_heads, dropout_rate=0, head_subspaces=False, **kwargs):
		super(GraphAttention_Naive, self).__init__()

		# as with other small transformers, there are no head sub-spaces.
		# Really need to test if this is necessary!
		self.d_model = d_model
		self.n_heads = n_heads
		if head_subspaces:
			self.d_head = d_model//n_heads
		else:
			self.d_head = d_model
		self.head_subspaces = head_subspaces

		self.Wq = nn.Linear(d_model, self.d_head*n_heads, bias=False, **kwargs)
		self.Wk = nn.Linear(d_model, self.d_head*n_heads, bias=False, **kwargs)

		self.Wv = nn.Linear(d_model, self.d_head*n_heads, bias=True, **kwargs)

		self.Wo = nn.Linear(d_model, d_model, bias=True, **kwargs)

		self.dropout = nn.Dropout(dropout_rate)
		self.gelu = QuickGELU()

	def forward(self, x, rotary_emb):
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
		# Q,K are hence [batch_size, n_heads, ntok, d_head]

		V = V.reshape(batch_size, ntok, self.n_heads, self.d_head).permute(0, 2, 1, 3)
		# V is [batch_size, n_heads, ntok, d_head]

		A = torch.einsum('bhid,bhjd->bhij', Q, K)
		if False: # causal attention
			mask = torch.triu(torch.ones(ntok, ntok), diagonal=1).bool().to(x.device)
			A = A.masked_fill(mask, -torch.inf) # Use a large negative value
		A = torch.softmax(A, dim=-1)
		y = torch.einsum('bhij,bhjd->bhid', A, V)

		# sum along the heads
		if self.head_subspaces:
			y = y.permute(0,2,1,3)
			y = y.reshape(batch_size, ntok, d_model)
		else:
			y = y.permute(0, 2, 3, 1).sum(dim=3).squeeze()
		y = self.gelu(y)
		y = self.Wo(y)
		# residual path is external to this layer.
		return y

	def calcFlops(self, x):
		bs, ntok, d_model = x.shape
		f = 0.0
		# QKV projection
		f += 3 * bs * ntok * d_model**2 * self.n_heads*d_model
		# attention
		f += bs * self.n_heads * ntok**2 * d_model
		# softmax
		f += bs * self.n_heads * ntok**2 * 2
		# V projection
		f += bs * self.n_heads * ntok * d_model**2 * 2
		# sum and gelu
		f += bs * self.n_heads * ntok * d_model * (2 + 6)
		# Wo proj
		f += bs * ntok * d_model**2
		return f

	def calcFlops(self, x):
		bs, ntok, d_model = x.shape
		f = 0.0
		# QKV projection
		f += 3 * bs * ntok * d_model**2 * self.n_heads*d_model
		# attention
		f += bs * self.n_heads * ntok**2 * d_model
		# softmax
		f += bs * self.n_heads * ntok**2 * 2
		# V projection
		f += bs * self.n_heads * ntok * d_model**2 * 2
		# sum and gelu
		f += bs * self.n_heads * ntok * d_model * (2 + 6)
		# Wo proj
		f += bs * ntok * d_model**2
		return f
