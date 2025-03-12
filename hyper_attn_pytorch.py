import torch
import torch.nn as nn
import math

class QuickGELU(nn.Module):
	def forward(self, x: torch.Tensor):
		return x * torch.sigmoid(1.702 * x)

class HypergraphAttention(nn.Module):
	def __init__(self, d_model, n_heads, dropout_rate=0):
		super(HypergraphAttention, self).__init__()

		torch.manual_seed(42)
		
		# as with other small transformers, there are no head subg-spaces.
		# Really need to test if this is necessary! 
		
		self.d_model = d_model
		self.n_heads = n_heads
		self.head_dim = d_model
		
		self.Wq = nn.Linear(d_model, d_model*n_heads, bias=False)
		self.Wr = nn.Linear(d_model, d_model*n_heads, bias=False)
		self.Ws = nn.Linear(d_model, d_model*n_heads, bias=False)
		
		self.Wv_q = nn.Linear(d_model, d_model*n_heads*2, bias=True)
		self.Wv_r = nn.Linear(d_model, d_model*n_heads*2, bias=True)
		self.Wv_s = nn.Linear(d_model, d_model*n_heads*2, bias=True)
		
		self.Wo = nn.Linear(d_model, d_model, bias=True)
		
		self.dropout = nn.Dropout(dropout_rate)
		self.gelu = QuickGELU()
		
	def forward(self, x):
		batch_size, ntok, d_model = x.shape
		
		Q = self.Wq(x)
		R = self.Wr(x)
		S = self.Ws(x)
		
		Vq = self.Wv_q(x)
		Vr = self.Wv_r(x)
		Vs = self.Wv_s(x)
		
		Q = Q.reshape(batch_size, ntok, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
		R = R.reshape(batch_size, ntok, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
		S = S.reshape(batch_size, ntok, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
		# Q,R,S are hence [batch_size, n_heads, ntok, head_dim]
		
		# Gather & scatter value projection
		Vq, Vq_ = Vq.reshape(batch_size, ntok, self.n_heads, self.head_dim*2).permute(0, 2, 1, 3).split(self.head_dim, dim=-1)
		Vr, Vr_ = Vr.reshape(batch_size, ntok, self.n_heads, self.head_dim*2).permute(0, 2, 1, 3).split(self.head_dim, dim=-1)
		Vs, Vs_ = Vs.reshape(batch_size, ntok, self.n_heads, self.head_dim*2).permute(0, 2, 1, 3).split(self.head_dim, dim=-1)
		# Vq,Vr,Vs also [batch_size, n_heads, ntok, head_dim]
		
		
		
		# compute 3-way attention scores of shape [b, h, i, j, k]
		dot_product = torch.einsum('bhid,bhjd,bhkd->bhijk', Q, R, S)
		dot_product = dot_product / (math.sqrt(self.head_dim))
		
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
		
		Aq = self.dropout(Aq)
		Ar = self.dropout(Ar)
		As = self.dropout(As)
		
		# Gather operations
		# hence the 'diamond' operation is multiply.
		Y_q = torch.einsum('bhijk,bhjd,bhkd->bhid', Aq, Vr, Vs)  
		Y_r = torch.einsum('bhijk,bhid,bhkd->bhjd', Ar, Vq, Vs)  
		Y_s = torch.einsum('bhijk,bhid,bhjd->bhkd', As, Vq, Vr) 
		
		# Scatter opeartions
		# Y_q_ = torch.einsum('bhijk,bhjd->bhid', Ar, Vr_) + \
		# 		 torch.einsum('bhijk,bhkd->bhid', As, Vs_)
		# Y_r_ = torch.einsum('bhijk,bhid->bhjd', Aq, Vq_) + \
		# 		 torch.einsum('bhijk,bhkd->bhjd', As, Vs_)
		# Y_s_ = torch.einsum('bhijk,bhid->bhkd', Aq, Vq_) + \
		# 		 torch.einsum('bhijk,bhjd->bhkd', Ar, Vr_)
			 
		Y_q_ = torch.einsum('bhijk,bhjd,bhijk,bhkd->bhid', Ar, Vr_, As, Vs_)
		Y_r_ = torch.einsum('bhijk,bhid,bhijk,bhkd->bhjd', Aq, Vq_, As, Vs_)
		Y_s_ = torch.einsum('bhijk,bhid,bhijk,bhjd->bhkd', Aq, Vq_, Ar, Vr_)
		
		y = Y_q + Y_r + Y_s + Y_q_ + Y_r_ + Y_s_
		
		# sum along the heads
		y = y.permute(0, 2, 1, 3).sum(dim=2).squeeze()
		y = self.gelu(y)
		y = self.Wo(y)
		# residual path is external to this layer.
		return y 
