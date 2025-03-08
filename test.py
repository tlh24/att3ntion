import numpy as np

def genData(bs, md, do_print=False): 
	'''
	Problem is a 3-term analogy: 
	if A `op B = C then D `op E = `F
	where op and F need to be `filled in. 
	Values are integers, ops are the usual arithmetic operations. 
	All opearations are over the finite field of integers: mod 'md'
	'''
	def randint(k): 
		return np.random.randint(k)
		
	assert(md < 32-4)
	x = np.zeros((bs, 8, 32), dtype=int)
		
	for b in range(bs): 
		unambig = False
		while not unambig: 
			va = randint(md)
			vb = randint(md-1) + 1 # avoid divide by zero and noops
			op = np.random.randint(4)
			vc0 = (va + vb) % md
			vc1 = (va - vb) % md
			vc2 = (va * vb) % md
			vc3 = (va // vb) % md
			# another way to do this is to sort the 4, calculate the differences, and compare that to zero - this requires 3 comparisons but potentially 4 swaps, so better to just do the 6 comparisons.
			if vc0 != vc1 and vc2 != vc3 and vc0 != vc2 and vc1 != vc3 and vc0 != vc3 and vc1 != vc2: 
				unambig = True
		# in contrast, D op E is always deterministic so don't need to check if the inference is unambiguous.  
		vd = randint(md)
		ve = randint(md-1) + 1
		match op: 
			case 0: 
				vc = vc0
				vf = (vd + ve) % md
			case 1:
				vc = vc1
				vf = (vd - ve) % md
			case 2: 
				vc = vc2
				vf = (vd * ve) % md
			case 3: 
				vc = vc3
				vf = (vd // ve) % md
		# encode. 
		x[b,0,va+4] = 1
		x[b,1,op  ] = 1 # must be masked
		x[b,2,vb+4] = 1
		x[b,3,vc+4] = 1
		
		x[b,4,vd+4] = 1
		x[b,5,op  ] = 1 # must be masked
		x[b,6,ve+4] = 1
		x[b,7,vf+4] = 1 # must be masked
		
		if do_print: 
			match op: 
				case 0: 
					ops = '+'
				case 1: 
					ops = '-'
				case 2: 
					ops = '*'
				case 3: 
					ops = '/'
			print(f"if {va} op {vb} = {vc} then {vd} op {ve} = f  (op = {ops}, f = {vf}")
	# endfor b
	return x

# class QuickGELU(nn.Module):
# 	def forward(self, x: torch.Tensor):
# 		return x * torch.sigmoid(1.702 * x)
# 
# class ResidualAttentionBlock(nn.Module):
# 	def __init__(self, d_model: int, n_head: int):
# 		super().__init__()
# 
# 		self.n_head = n_head
# 		self.d_model = d_model
# 		self.wk = nn.Parameter( 0.005 * torch.ones(n_head, d_model) )
# 
# 		self.wqv = nn.Linear(d_model, 3*n_head*d_model)
# 		self.initWeights(self.wqv)
# 		# add in some identity
# 		with torch.no_grad(): 
# 			for i in range(3): 
# 				self.wqv.weight[i*d_model:(i+1)*d_model, :] += torch.eye(self.d_model, device=self.wqv.weight.device) * 0.01
# 			
# 		self.fanin = nn.Linear(d_model, d_model)
# 
# 		self.l1a_f = l1attn_cuda.L1Attn()
# 
# 		self.gelu = QuickGELU()
# 		self.rms_norm = nn.RMSNorm(d_model)
# 
# 	def initWeights(self, module):
# 		if isinstance(module, nn.Linear):
# 			torch.nn.init.normal_(module.weight, mean=0.0, std=0.005) 
# 			if module.bias is not None:
# 				torch.nn.init.zeros_(module.bias)
# 		
# 	def attentionDP(self, x:torch.Tensor): 
# 		n_head = self.n_head
# 		d_head = self.d_model ## no sub-spaces!
# 		batch_size = x.shape[0]
# 		ntok = x.shape[1]
# 
# 		o = self.wqv(x)
# 		o = torch.reshape(o, (batch_size, ntok, 3*self.n_head, d_head))
# 		q,k,v = torch.split(o, self.n_head, 2)
# 		# q,k,v are shape [batch_size, ntok, n_head, d_head]
# 		
# 		a = torch.einsum('bthw, bshw -> btsh', q, k) / math.sqrt(d_head)
# 		a = F.softmax(a, 1)
# 		b = torch.einsum('btsh, bshw -> bthw', a, v)
# 		b = torch.sum(b, dim=2) # sum along the heads
# 		return b
# 		
# 
# 	def forward(self, x:torch.Tensor, use_dp:bool):
# 		if use_dp: 
# 			y = self.attentionDP( self.rms_norm(x) )
# 		else: 
# 			y = self.attention(x)
# 		y = self.gelu(y)
# 		y = self.fanin(y) # allow sign inversions & mixing; no dim change
# 		return x + y
# 
# class Transformer(nn.Module):
# 	def __init__(self, d_model:int, layers:int, repeat:int, n_head:int):
# 		super().__init__()
# 		self.d_model = d_model
# 		self.n_head = n_head
# 		self.layers = layers
# 		self.repeat = repeat
# 		self.resblocks = nn.ModuleList(\
# 			[ResidualAttentionBlock(d_model, n_head) \
# 				for _ in range(layers)])
# 		self.in_proj = nn.Linear(gendata_dim, d_model, bias=True)
# 		self.out_proj = nn.Linear(d_model, gendata_dim, bias=True)
# 
# 	def forward(self, x:torch.Tensor, use_dp:bool):
# 		# x is dtype int to interface with the embedding layer
# 		bs,n_tok,inw = x.shape
# 		x = self.in_proj(x)
# 		# x = torch.cat((x, torch.zeros(bs, n_tok, self.d_model - inw, device=x.device)), axis=-1)
# 		for i in range(self.repeat):
# 			for j, layer in enumerate(self.resblocks):
# 				x = layer(x, use_dp)
# 		return self.out_proj(x)
# 
# 	def fixedInit(self):
# 		for layer in self.resblocks:
# 			layer.fixedInit()
# 
# 	def printParamCount(self):
# 		trainable_params = sum(
# 			p.numel() for p in self.parameters() if p.requires_grad
# 		)
# 		print(f"Number of model parameters:{trainable_params}")


if __name__ == '__main__':
	genData(10, 7, True)
