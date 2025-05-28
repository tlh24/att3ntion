import torch
import torch.nn as nn
import math
import pdb

class QuickGELU(nn.Module):
	def forward(self, x: torch.Tensor):
		return x * torch.sigmoid(1.702 * x)

class HypergraphAttention(nn.Module):
	def __init__(self, d_model, n_heads, dropout_rate=0, **kwargs):
		super(HypergraphAttention, self).__init__()

		torch.manual_seed(42)
		
		# as with other small transformers, there are no head sub-spaces.
		# Really need to test if this is necessary! 
		
		self.d_model = d_model
		self.n_heads = n_heads
		self.head_dim = d_model
		
		self.Wq = nn.Linear(d_model, d_model*n_heads, bias=False, **kwargs)
		self.Wr = nn.Linear(d_model, d_model*n_heads, bias=False, **kwargs)
		self.Ws = nn.Linear(d_model, d_model*n_heads, bias=False, **kwargs)
		
		self.Wv_q = nn.Linear(d_model, d_model*n_heads*2, bias=True, **kwargs)
		self.Wv_r = nn.Linear(d_model, d_model*n_heads*2, bias=True, **kwargs)
		self.Wv_s = nn.Linear(d_model, d_model*n_heads*2, bias=True, **kwargs)
		
		self.Wo = nn.Linear(d_model, d_model, bias=True, **kwargs)
		
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
		
        # Aq = self.dropout(Aq)
		# Ar = self.dropout(Ar)
		# As = self.dropout(As)
		# No dropout for testing backprop
		self.dropout_mask_q = torch.ones_like(Aq)
		self.dropout_mask_r = torch.ones_like(Ar)
		self.dropout_mask_s = torch.ones_like(As)
		
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

	def backward(self, x, dL_dy):
		batch_size, ntok, d_model = x.shape
		
		grads = {}

		# Step 1: Recompute forward pass values (no need to save activations)
		
		Q = self.Wq(x)
		R = self.Wr(x)
		S = self.Ws(x)
		
		Vq = self.Wv_q(x)
		Vr = self.Wv_r(x)
		Vs = self.Wv_s(x)
		
		Q = Q.reshape(batch_size, ntok, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
		R = R.reshape(batch_size, ntok, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
		S = S.reshape(batch_size, ntok, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
		
		Vq, Vq_ = Vq.reshape(batch_size, ntok, self.n_heads, self.head_dim*2).permute(0, 2, 1, 3).split(self.head_dim, dim=-1)
		Vr, Vr_ = Vr.reshape(batch_size, ntok, self.n_heads, self.head_dim*2).permute(0, 2, 1, 3).split(self.head_dim, dim=-1)
		Vs, Vs_ = Vs.reshape(batch_size, ntok, self.n_heads, self.head_dim*2).permute(0, 2, 1, 3).split(self.head_dim, dim=-1)
		
		dot_product = torch.einsum('bhid,bhjd,bhkd->bhijk', Q, R, S)
		dot_product = dot_product / (math.sqrt(self.head_dim))
		
		# Compute attention weights
		dot_product_q = dot_product
		Aq = torch.softmax(dot_product_q.flatten(3, 4), dim=-1).reshape(dot_product.shape)
		
		dot_product_r = dot_product.permute(0, 1, 3, 2, 4)
		Ar = torch.softmax(dot_product_r.flatten(3, 4), dim=-1).reshape(dot_product.shape)
		Ar = Ar.permute(0, 1, 3, 2, 4)
		
		dot_product_s = dot_product.permute(0, 1, 4, 2, 3)
		As = torch.softmax(dot_product_s.flatten(3, 4), dim=-1).reshape(dot_product.shape)
		As = As.permute(0, 1, 3, 4, 2)
		
		# Gather operations
		Y_q = torch.einsum('bhijk,bhjd,bhkd->bhid', Aq, Vr, Vs)
		Y_r = torch.einsum('bhijk,bhid,bhkd->bhjd', Ar, Vq, Vs)
		Y_s = torch.einsum('bhijk,bhid,bhjd->bhkd', As, Vq, Vr)
		
		# Scatter operations
		Y_q_ = torch.einsum('bhijk,bhjd,bhijk,bhkd->bhid', Ar, Vr_, As, Vs_)
		Y_r_ = torch.einsum('bhijk,bhid,bhijk,bhkd->bhjd', Aq, Vq_, As, Vs_)
		Y_s_ = torch.einsum('bhijk,bhid,bhijk,bhjd->bhkd', Aq, Vq_, Ar, Vr_)
		
		y = Y_q + Y_r + Y_s + Y_q_ + Y_r_ + Y_s_
		y = y.permute(0, 2, 1, 3).sum(dim=2).squeeze()
		y_gelu = self.gelu(y)
		output = self.Wo(y_gelu)
		
		# Step 2: Backprop through each layer

		dy_gelu = dL_dy @ self.Wo.weight
		dWo = y_gelu.reshape(-1, d_model).t() @ dL_dy.reshape(-1, d_model)
		grads['dWo'] = dWo

		## GELU - gelu(x) = x * sigmoid(1.702*x)
		sigmoid_val = torch.sigmoid(1.702 * y)
		gelu_deriv = sigmoid_val + y * 1.702 * sigmoid_val * (1 - sigmoid_val)
		dy = dy_gelu * gelu_deriv

		dy = dy.unsqueeze(2).expand(-1, -1, self.n_heads, -1).permute(0, 2, 1, 3)
		
		dY_q = dy
		dY_r = dy
		dY_s = dy
		dY_q_ = dy
		dY_r_ = dy
		dY_s_ = dy

		# Backprop through gather operations (Y_q, Y_r, Y_s)
		# Y_q = torch.einsum('bhijk,bhjd,bhkd->bhid', Aq, Vr, Vs)
		dAq_from_Yq = torch.einsum('bhid,bhjd,bhkd->bhijk', dY_q, Vr, Vs)
		dVr_from_Yq = torch.einsum('bhijk,bhid,bhkd->bhjd', Aq, dY_q, Vs)
		dVs_from_Yq = torch.einsum('bhijk,bhid,bhjd->bhkd', Aq, dY_q, Vr)

		# Y_r = torch.einsum('bhijk,bhid,bhkd->bhjd', Ar, Vq, Vs)
		dAr_from_Yr = torch.einsum('bhjd,bhid,bhkd->bhijk', dY_r, Vq, Vs)
		dVq_from_Yr = torch.einsum('bhijk,bhjd,bhkd->bhid', Ar, dY_r, Vs)
		dVs_from_Yr = torch.einsum('bhijk,bhid,bhjd->bhkd', Ar, Vq, dY_r)

		# Y_s = torch.einsum('bhijk,bhid,bhjd->bhkd', As, Vq, Vr)
		dAs_from_Ys = torch.einsum('bhkd,bhid,bhjd->bhijk', dY_s, Vq, Vr)
		dVq_from_Ys = torch.einsum('bhijk,bhkd,bhjd->bhid', As, dY_s, Vr)
		dVr_from_Ys = torch.einsum('bhijk,bhid,bhkd->bhjd', As, Vq, dY_s)

		# Backprop through scatter operations (Y_q_, Y_r_, Y_s_)
		# Y_q_ = torch.einsum('bhijk,bhjd,bhijk,bhkd->bhid', Ar, Vr_, As, Vs_)
		dAr_from_Yq_ = torch.einsum('bhid,bhjd,bhijk,bhkd->bhijk', dY_q_, Vr_, As, Vs_)
		dVr__from_Yq_ = torch.einsum('bhijk,bhid,bhijk,bhkd->bhjd', Ar, dY_q_, As, Vs_)
		dAs_from_Yq_ = torch.einsum('bhijk,bhjd,bhid,bhkd->bhijk', Ar, Vr_, dY_q_, Vs_)
		dVs__from_Yq_ = torch.einsum('bhijk,bhjd,bhijk,bhid->bhkd', Ar, Vr_, As, dY_q_)
		
		# Y_r_ = torch.einsum('bhijk,bhid,bhijk,bhkd->bhjd', Aq, Vq_, As, Vs_)
		dAq_from_Yr_ = torch.einsum('bhjd,bhid,bhijk,bhkd->bhijk', dY_r_, Vq_, As, Vs_)
		dVq__from_Yr_ = torch.einsum('bhijk,bhjd,bhijk,bhkd->bhid', Aq, dY_r_, As, Vs_)
		dAs_from_Yr_ = torch.einsum('bhijk,bhid,bhjd,bhkd->bhijk', Aq, Vq_, dY_r_, Vs_)
		dVs__from_Yr_ = torch.einsum('bhijk,bhid,bhijk,bhjd->bhkd', Aq, Vq_, As, dY_r_)
		
		# Y_s_ = torch.einsum('bhijk,bhid,bhijk,bhjd->bhkd', Aq, Vq_, Ar, Vr_)
		dAq_from_Ys_ = torch.einsum('bhkd,bhid,bhijk,bhjd->bhijk', dY_s_, Vq_, Ar, Vr_)
		dVq__from_Ys_ = torch.einsum('bhijk,bhkd,bhijk,bhjd->bhid', Aq, dY_s_, Ar, Vr_)
		dAr_from_Ys_ = torch.einsum('bhijk,bhid,bhkd,bhjd->bhijk', Aq, Vq_, dY_s_, Vr_)
		dVr__from_Ys_ = torch.einsum('bhijk,bhid,bhijk,bhkd->bhjd', Aq, Vq_, Ar, dY_s_)

		dAq = dAq_from_Yq + dAq_from_Yr_ + dAq_from_Ys_
		dAr = dAr_from_Yr + dAr_from_Yq_ + dAr_from_Ys_
		dAs = dAs_from_Ys + dAs_from_Yq_ + dAs_from_Yr_

		dAq = dAq * self.dropout_mask_q
		dAr = dAr * self.dropout_mask_r
		dAs = dAs * self.dropout_mask_s
		
		dVq = dVq_from_Yr + dVq_from_Ys
		dVr = dVr_from_Yq + dVr_from_Ys
		dVs = dVs_from_Yq + dVs_from_Yr
		
		dVq_ = dVq__from_Yr_ + dVq__from_Ys_
		dVr_ = dVr__from_Yq_ + dVr__from_Ys_
		dVs_ = dVs__from_Yq_ + dVs__from_Yr_


		# Backprop through softmax operations
		# For Aq - softmax over j,k dimensions
		dAq_flat = dAq.flatten(3, 4)
		Aq_flat = Aq.flatten(3, 4)
		dDot_product_q_flat = Aq_flat * (dAq_flat - (Aq_flat * dAq_flat).sum(dim=-1, keepdim=True))
		dDot_product_q = dDot_product_q_flat.reshape(dot_product.shape)
		
		# For Ar - softmax over i,k dimensions (with permutation)
		dAr_perm = dAr.permute(0, 1, 3, 2, 4)  # [b, h, i, j, k] -> [b, h, j, i, k]
		Ar_perm = Ar.permute(0, 1, 3, 2, 4)
		dAr_flat = dAr_perm.flatten(3, 4)
		Ar_flat = Ar_perm.flatten(3, 4)
		dDot_product_r_flat = Ar_flat * (dAr_flat - (Ar_flat * dAr_flat).sum(dim=-1, keepdim=True))
		dDot_product_r_perm = dDot_product_r_flat.reshape(dAr_perm.shape)
		dDot_product_r = dDot_product_r_perm.permute(0, 1, 3, 2, 4) 
		
		# For As - softmax over i,j dimensions (with different permutation)
		dAs_perm = dAs.permute(0, 1, 4, 2, 3)  # [b, h, i, j, k] -> [b, h, k, i, j]
		As_perm = As.permute(0, 1, 4, 2, 3)
		dAs_flat = dAs_perm.flatten(3, 4)
		As_flat = As_perm.flatten(3, 4)
		
		dDot_product_s_flat = As_flat * (dAs_flat - (As_flat * dAs_flat).sum(dim=-1, keepdim=True))
		dDot_product_s_perm = dDot_product_s_flat.reshape(dAs_perm.shape)
		dDot_product_s = dDot_product_s_perm.permute(0, 1, 3, 4, 2)  # Back to [b, h, i, j, k]
		
		dDot_product = dDot_product_q + dDot_product_r + dDot_product_s
		
		scale = math.sqrt(self.head_dim)
		dDot_product_scaled = dDot_product / scale
		
		dQ = torch.einsum('bhijk,bhjd,bhkd->bhid', dDot_product_scaled, R, S)
		dR = torch.einsum('bhijk,bhid,bhkd->bhjd', dDot_product_scaled, Q, S)
		dS = torch.einsum('bhijk,bhid,bhjd->bhkd', dDot_product_scaled, Q, R)

		dVq_combined = torch.cat([dVq, dVq_], dim=-1)
		dVr_combined = torch.cat([dVr, dVr_], dim=-1)
		dVs_combined = torch.cat([dVs, dVs_], dim=-1)
		
		dQ = dQ.permute(0, 2, 1, 3).reshape(batch_size, ntok, self.n_heads * self.head_dim)
		dR = dR.permute(0, 2, 1, 3).reshape(batch_size, ntok, self.n_heads * self.head_dim)
		dS = dS.permute(0, 2, 1, 3).reshape(batch_size, ntok, self.n_heads * self.head_dim)
		
		dVq_combined = dVq_combined.permute(0, 2, 1, 3).reshape(batch_size, ntok, self.n_heads * self.head_dim * 2)
		dVr_combined = dVr_combined.permute(0, 2, 1, 3).reshape(batch_size, ntok, self.n_heads * self.head_dim * 2)
		dVs_combined = dVs_combined.permute(0, 2, 1, 3).reshape(batch_size, ntok, self.n_heads * self.head_dim * 2)
		
		dWq = x.reshape(-1, d_model).T @ dQ.reshape(-1, self.n_heads * self.head_dim)
		dWr = x.reshape(-1, d_model).T @ dR.reshape(-1, self.n_heads * self.head_dim)
		dWs = x.reshape(-1, d_model).T @ dS.reshape(-1, self.n_heads * self.head_dim)
		
		dWv_q = x.reshape(-1, d_model).T @ dVq_combined.reshape(-1, self.n_heads * self.head_dim * 2)
		dWv_r = x.reshape(-1, d_model).T @ dVr_combined.reshape(-1, self.n_heads * self.head_dim * 2)
		dWv_s = x.reshape(-1, d_model).T @ dVs_combined.reshape(-1, self.n_heads * self.head_dim * 2)
		
		dx_q = dQ @ self.Wq.weight
		dx_r = dR @ self.Wr.weight
		dx_s = dS @ self.Ws.weight
		dx_vq = dVq_combined @ self.Wv_q.weight
		dx_vr = dVr_combined @ self.Wv_r.weight
		dx_vs = dVs_combined @ self.Wv_s.weight
		
		dx = dx_q + dx_r + dx_s + dx_vq + dx_vr + dx_vs
		
		if self.Wv_q.bias is not None:
			dWv_q_bias = dVq_combined.reshape(-1, dVq_combined.size(-1)).sum(dim=0)
			grads['dWv_q_bias'] = dWv_q_bias
			
			dWv_r_bias = dVr_combined.reshape(-1, dVr_combined.size(-1)).sum(dim=0)
			grads['dWv_r_bias'] = dWv_r_bias
			
			dWv_s_bias = dVs_combined.reshape(-1, dVs_combined.size(-1)).sum(dim=0)
			grads['dWv_s_bias'] = dWv_s_bias
			
			dWo_bias = dL_dy.reshape(-1, dL_dy.size(-1)).sum(dim=0)
			grads['dWo_bias'] = dWo_bias


		grads.update({
			'dWq': dWq,
			'dWr': dWr,
			'dWs': dWs,
			'dWv_q': dWv_q,
			'dWv_r': dWv_r,
			'dWv_s': dWv_s,
			'dx': dx
		})
		return grads


