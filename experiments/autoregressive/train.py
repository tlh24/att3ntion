import numpy as np
import math
import argparse
import inspect
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding
import sys
import pdb
from pathlib import Path
current_script_path = Path(__file__).resolve()
script_dir = str(current_script_path.parent)
project_root = str(current_script_path.parent.parent.parent)
for d in [script_dir, project_root]:
    if d not in sys.path:
        sys.path.insert(0, d)

from att3ntion import _HypergraphAttentionNaive, _GraphAttentionNaive, QuickGELU
from gen_data import gen_data, to_numpy, from_numpy

SEQ_L = 1 # length of the output sequence
MASKED = True # masked vs autoregressive training.
		# if False, use autoregressive training

# Try to import CUDA-backed hypergraph attention; fall back to naive if not built.
_cuda_kernels_available = False
try:
	from att3ntion import HypergraphAttention as _HypergraphAttentionCuda
	_cuda_kernels_available = True
except (ImportError, OSError):
	pass

if _cuda_kernels_available:
	class _CudaHypergraphWrapper(_HypergraphAttentionCuda):
		"""HypergraphAttention adapted to the (x, rotary_emb) call convention used here."""
		def forward(self, x, rotary_emb=None):
			return super().forward(x)

		def calcFlops(self, x):
			bs, ntok, d_model = x.shape
			f = 0.0
			f += 3 * bs * ntok * d_model**2 * self.n_heads * d_model
			f += 3 * bs * ntok * d_model**2 * self.n_heads * d_model * 2
			f += bs * self.n_heads * ntok**3 * self.head_dim * 2
			f += bs * self.n_heads * ntok**3 * 2 * 3
			f += bs * self.n_heads * ntok**3 * self.head_dim * 6
			f += bs * self.n_heads * ntok * self.head_dim * (6 + 6)
			f += bs * ntok * d_model**2
			return f

class SwiGLU(nn.Module):
	"""
	Swish Gated Linear Units based Feed-Forward Network.
	"""
	def __init__(self, in_features, hidden_features, out_features):
		super().__init__()
		self.w1 = nn.Linear(in_features, hidden_features)
		self.w2 = nn.Linear(in_features, hidden_features)
		self.w3 = nn.Linear(hidden_features, out_features)

	def forward(self, x):
		return self.w3(F.silu(self.w1(x)) * self.w2(x))

class SimpleCompModel(nn.Module):
	"""Model with hypergraph attention layer."""
	def __init__(self, input_vocab:int, hidden_dim:int, n_heads:int, n_layers:int, attn_impl:str='', n_recurse:int=1, use_cuda_kernels:bool=True):
		super().__init__()
		self.input_vocab = input_vocab
		self.embedding_proj = nn.Embedding(input_vocab, hidden_dim)
		nn.init.normal_(self.embedding_proj.weight, std=0.02)
		self.attn_impl = attn_impl
		self.n_recurse = n_recurse
		self.d_model = hidden_dim
		self.rope = RotaryEmbedding(dim = hidden_dim//n_heads)

		use_cuda = use_cuda_kernels and _cuda_kernels_available and attn_impl == "hypergraph"
		if use_cuda:
			print("Using CUDA hypergraph attention kernels.")
		elif attn_impl == "hypergraph":
			print("Using naive (PyTorch) hypergraph attention.")

		self.repeated_layers = nn.ModuleList()
		for _ in range(n_layers):
			if attn_impl == "hypergraph":
				if use_cuda:
					attention_layer = _CudaHypergraphWrapper(hidden_dim, n_heads)
				else:
					attention_layer = _HypergraphAttentionNaive(hidden_dim, n_heads, head_subspaces=True)
			else:
				attention_layer = _GraphAttentionNaive(hidden_dim, n_heads, head_subspaces=True)

			norm1_layer = nn.RMSNorm(hidden_dim)
			norm2_layer = nn.RMSNorm(hidden_dim)
			# norm1_layer = nn.LayerNorm(hidden_dim)
			# norm2_layer = nn.LayerNorm(hidden_dim)
			if True:
				ffn_layer = nn.Sequential(
					nn.Linear(hidden_dim, 3 * hidden_dim, bias=False),
					nn.ReLU(),
					nn.Linear(3 * hidden_dim, hidden_dim, bias=False)
				)
			else:
				ffn_layer = SwiGLU(hidden_dim, 2*hidden_dim, hidden_dim)

			self.repeated_layers.append(
				nn.ModuleDict({
						'attention': attention_layer,
						'norm1': norm1_layer,
						'ffn': ffn_layer,
						'norm2': norm2_layer,
				})
				)
		self.output_proj = nn.Linear(hidden_dim, input_vocab, bias=False)
		self.gelu = QuickGELU()
		
	def forward(self, x):
		bs, ntok = x.shape
		# mask = torch.tril(torch.ones(ntok, ntok))
		# mask = mask.to(x.device)
		mask = None # non-autoregressive now
		x = self.embedding_proj(x)
		for r in range(self.n_recurse):
			for layer_block in self.repeated_layers:
				xn = layer_block['norm1'](x)
				attn_output = layer_block['attention'](xn, self.rope, mask)
				x = x + attn_output
				xn = layer_block['norm2'](attn_output) # NOTE: could be x
				ffn_output = layer_block['ffn'](xn)
				x = x + ffn_output
		return self.output_proj(x)

	def save_model(self, path: str):
		"""Saves the model's configuration and state dictionary."""
		torch.save(self.state_dict(), path)
		print(f"saved model to {path}")

	def load_model(self, path: str, device):
		"""Loads a model from a file."""
		checkpoint = torch.load(path, map_location=device)
		self.load_state_dict(checkpoint)
		self.to(device)
		self.eval() # Set to evaluation mode by default
		return

	def printParamCount(self):
		trainable_params = sum(
			p.numel() for p in self.parameters() if p.requires_grad
		)
		print(f"SimpleCompModel {self.attn_impl}: number of model parameters:{trainable_params/1e6}M")

	def calcFlops(self, x):
		x = self.embedding_proj(x)
		bs, ntok, _ = x.shape
		f = 0
		for r in range(self.n_recurse):
			for layer_block in self.repeated_layers:
				f += layer_block['attention'].calcFlops(x)
				f += bs * ntok * self.d_model * 10 # layerNorm 1
				f += bs * ntok * self.d_model**2 * 3 * 2 # ffn
				f += bs * ntok * self.d_model * 10 # layerNorm 2
		f += bs * ntok * self.d_model**2 * self.input_vocab # output proj
		return f

USE_ROPE = True # if False, use learned position encoding.

class GrokkingTransformer(nn.Module):
	def __init__(self, p, d, n_heads=1, use_norm=False, mlp_bias=True, mlp_act='relu'):
		super().__init__()
		assert d % n_heads == 0, "d must be divisible by n_heads"
		self.p = p
		self.d = d
		self.n_heads = n_heads
		self.head_dim = d // n_heads

		# Vocab: 0 to p-1 are standard numbers. Token `p` is the special '=' token.
		self.embed = nn.Embedding(p + 1, d)
		nn.init.normal_(self.embed.weight, std=0.02)

		if USE_ROPE:
			self.rope = RotaryEmbedding(dim=self.head_dim)
		else:
			self.pos_emb = nn.Embedding(3, d)
			nn.init.normal_(self.pos_emb.weight, std=0.02)

		# Multi-head attention projections
		self.W_q = nn.Linear(d, d, bias=False)
		self.W_k = nn.Linear(d, d, bias=False)
		self.W_v = nn.Linear(d, d, bias=False)
		self.W_o = nn.Linear(d, d, bias=False)

		self.ln1 = nn.LayerNorm(d) if use_norm else nn.Identity()
		self.ln2 = nn.LayerNorm(d) if use_norm else nn.Identity()

		self.mlp_w1 = nn.Linear(d, 4 * d, bias=mlp_bias)
		self.mlp_w2 = nn.Linear(4 * d, d, bias=mlp_bias)
		self.mlp_act = mlp_act

		self.W_out = nn.Linear(d, p, bias=True)
		nn.init.normal_(self.W_out.weight, std=1.0 / np.sqrt(d))

	def forward(self, x):
		# x shape: (B, 3)
		e = self.embed(x)
		return self.forward_from_embeddings(e)

	def forward_from_embeddings(self, e):
		# Unroll logic allows torch.func.vmap to run over the sequence
		# pdb.set_trace()
		is_batched = e.dim() == 3
		if not is_batched: e = e.unsqueeze(0)

		B, seq_len, d = e.shape

		if not USE_ROPE:
			positions = torch.arange(seq_len, device=e.device)
			# positions = torch.tensor([0, 0, 2], device=e.device)
			# force commutativity.  This is worse!
			e = e + self.pos_emb(positions)

		x_norm = self.ln1(e)
		q, k, v = self.W_q(x_norm), self.W_k(x_norm), self.W_v(x_norm)

		# Split into heads: (B, seq, d) -> (B*n_heads, seq, head_dim)
		def split_heads(t):
			return t.view(B, seq_len, self.n_heads, self.head_dim).transpose(1, 2).reshape(B * self.n_heads, seq_len, self.head_dim)

		q, k, v = split_heads(q), split_heads(k), split_heads(v)

		if USE_ROPE:
			q = self.rope.rotate_queries_or_keys(q.view(B, self.n_heads, seq_len, self.head_dim)).reshape(B * self.n_heads, seq_len, self.head_dim)
			k = self.rope.rotate_queries_or_keys(k.view(B, self.n_heads, seq_len, self.head_dim)).reshape(B * self.n_heads, seq_len, self.head_dim)

		scores = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(self.head_dim)

		# Standard causal mask for autoregressive emulation
		# mask not strictly required..
		mask = torch.tril(torch.ones(seq_len, seq_len, device=e.device)).unsqueeze(0)
		scores = scores.masked_fill(mask == 0, float('-inf'))
		attn = torch.softmax(scores, dim=-1)

		# Merge heads back: (B*n_heads, seq, head_dim) -> (B, seq, d)
		context = torch.bmm(attn, v).view(B, self.n_heads, seq_len, self.head_dim).transpose(1, 2).reshape(B, seq_len, d)
		h = e + context

		# MLP block
		h_norm = self.ln2(context)
		h_mid = self.mlp_w1(h_norm)
		h_mid = h_mid ** 2 if self.mlp_act == 'quadratic' else nn.functional.relu(h_mid)
		mlp_out = self.mlp_w2(h_mid)
		h = h + mlp_out

		return h

def calcLoss(pred, targets):
	n_correct = 0
	n_possible = 0
	bs = pred.shape[0]
	ntok = pred.shape[1]
	seq_pred = pred[:, -SEQ_L:, :].reshape(-1, pred.shape[-1])
	seq_targets = targets[:, -SEQ_L:].long().reshape(-1)
	loss = F.cross_entropy(seq_pred, seq_targets)
	with torch.no_grad():
		n_correct = torch.sum(torch.argmax(seq_pred, dim=-1) == seq_targets).item()
		n_possible = seq_targets.shape[0]
	return loss, n_correct, n_possible

def trainModel(num_epochs, batch_size, hidden_dim, n_heads, device, task, attn_impl="", log_name="", save_model=False, replicate=1, no_amp=False, use_cuda_kernels=True):
	
	if device == 'auto':
		if torch.cuda.is_available():
			device = torch.device('cuda')
		else:
			device = torch.device('cpu')
	else:
		device = torch.device(device)
	
	print(f"Using device: {device}")
	
	if task == 1:
		# for grokking, need to generate the (nearly) full table
		train_s, test_s = gen_data('grok', max_d=1, V=2, C=0, P=113, L=SEQ_L, exact_v=True, data_size=120**2)
	if task == 2:
		# copy task
		train_s, test_s = gen_data('grok', max_d=0, V=3, C=0, P=113, L=SEQ_L, exact_v=False, data_size=120**2)
	if task == 3:
		train_s, test_s = gen_data('grok', max_d=1, V=3, C=0, P=113, L=SEQ_L, exact_v=False, data_size=100_000)
	for i in range(10): print(f"Train: {train_s[i]}")
	for i in range(5):  print(f"Test:  {test_s[i]}")
	print(f"Train size {len(train_s)} test size {len(test_s)}")

	max_exprlen, train_np, test_np = to_numpy(train_s, test_s, 113)
	x = train_np.copy() + 1
	y = train_np + 1 # to_numpy uses -1 for the null tok
	x_v = test_np.copy() + 1
	y_v = test_np + 1
	if MASKED:
		x[:,-SEQ_L:] = 0 # mask off sequence tokens
		x_v[:,-SEQ_L:] = 0
	else:
		x = x[:,:-1] # shift for autoregression
		y = y[:,1:] # this doesn't work well for graph attn
		x_v = x_v[:,:-1]
		y_v = y_v[:,1:]

	dataset = TensorDataset(torch.tensor(x), torch.tensor(y))
	train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

	dataset_v = TensorDataset(torch.tensor(x_v), torch.tensor(y_v))
	loader_v = DataLoader(dataset_v, batch_size=batch_size, shuffle=True)

	vocab_size = np.max(x) + 1

	is_hg = attn_impl == "hypergraph"
	if task == 1 or task == 2:
		n_layers = 1
		if attn_impl == "graph":
			n_layers = 2
		n_recurse = 1
		n_heads = 1
	if task == 3:
		n_layers = 2
		if attn_impl == "graph":
			n_layers = 4
		n_recurse = 2 # depends on the formula depth
		# n_heads = 6 # use the command line arg

	dtype = torch.float32
	model = SimpleCompModel(vocab_size, hidden_dim, n_heads, n_layers=n_layers, attn_impl=attn_impl, n_recurse=n_recurse, use_cuda_kernels=use_cuda_kernels).to(device=device, dtype=dtype)
	# model = GrokkingTransformer(vocab_size, hidden_dim, 1, use_norm=True).to(device=device)
	if save_model:
		try:
			model.load_model(f"comp_model_{attn_impl}_r{replicate}.pt", device)
		except:
			print("train_model: could not load the saved model weights")

	weight_decay = 1e-2 # karpathy 1e-1, default 1e-2
	betas = (0.9, 0.95)
	optimizer = torch.optim.AdamW(model.parameters(),
										lr=0.001,
										weight_decay=weight_decay,
										betas=betas,
										amsgrad=True, eps=1e-5)
	try:
		model.printParamCount()
	except:
		print("\\ grokking transformer does not calculate number of parameters")
	model = torch.compile(model) # mode="max-autotune" or backend="eager"

	if no_amp:
		print("--- Running in full fp32 ---")
	else:
		print("--- Running with Torch automatic mixed precision (fp32 weights) ---")

	nam = {"hypergraph":"hg","graph":"g"}.get(attn_impl)
	fd_losslog = open(f'losslog_{nam}_{log_name}_s{SEQ_L}_r{replicate}.txt', 'w')

	print("\ntrain_model1 started...")
	uu = 0
	rng = np.random.default_rng()
	validation_loss = 1.0
	validation_top1_err = 1.0

	for epoch in range(num_epochs):
		total_loss = 0
		correct_vals = 0
		total = 0

		start_event = torch.cuda.Event(enable_timing=True)
		end_event = torch.cuda.Event(enable_timing=True)
		model.train()
		n_train = x.shape[0]
		for batch_indx in range(n_train // batch_size):
			mb_idx = rng.permutation(n_train)[:batch_size]
			inputs = torch.tensor(x[mb_idx,:]).to(device=device)
			targets = torch.tensor(y[mb_idx,:]).to(device=device)

			if batch_indx == 0:
				d = inputs[0:5, :].detach().cpu().numpy()
				print(d-1) # null token mapped to -1
				print(from_numpy(d-1, 113))
				print("targets:")
				d = targets[0:5, :].detach().cpu().numpy()
				print(from_numpy(d-1, 113))

			if batch_indx % 100 == 0:
				start_event.record()
			optimizer.zero_grad()

			if no_amp:
				pred = model(inputs, batch_indx)
				loss, n_correct, n_possible = calcLoss(pred, targets)
			else:
				with autocast('cuda'):
					pred = model(inputs)
					loss, n_correct, n_possible = calcLoss(pred, targets)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optimizer.step()

			correct_vals += n_correct
			total += n_possible
			lloss = loss.detach().cpu().item()
			total_loss += lloss
			lloss = lloss / inputs.shape[0] # normalize by batch size
			top1_err = 1 - (n_correct / n_possible)
			fd_losslog.write(f"{uu}\t{lloss}\t{top1_err*math.exp(1)}\t{validation_loss}\t{validation_top1_err*math.exp(1)}\n")
			uu += 1
			
			if batch_indx % 500 == 0:
				end_event.record()
				torch.cuda.synchronize()
				amp_time = start_event.elapsed_time(end_event)
				print("batch time:", amp_time, "ms")
				fd_losslog.flush()
		
		avg_loss = total_loss / len(train_loader)
		train_accuracy = 100 * correct_vals / total
		print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%')

		# validation!
		total_loss = 0
		correct_vals = 0
		total = 0
		with torch.no_grad():
			for batch_indx, (inputs_np,outputs_np) in enumerate(loader_v):
				inputs = inputs_np.to(device=device)
				targets = outputs_np.to(device=device)

				if no_amp:
					pred = model(inputs, 0)
					loss, n_correct, n_possible = calcLoss(pred, targets)
				else:
					with autocast('cuda'):
						pred = model(inputs)
						loss, n_correct, n_possible = calcLoss(pred, targets)
				correct_vals += n_correct
				total += n_possible
				lloss = loss.detach().cpu().item()
				lloss = lloss / inputs.shape[0]
				top1_err = 1 - (correct_vals / n_possible)
				# if epoch == num_epochs-1:
				# 	fd_losslog.write(f"{uu}\t{-1.0}\t{-1.0}\t{lloss}\t{top1_err*math.exp(1)}\n")
				# 	uu += 1
				total_loss += lloss

			avg_loss = total_loss / len(loader_v)
			val_accuracy = 100 * correct_vals / total
			print(f'Validation Loss: {avg_loss:.4f}, accuracy {val_accuracy}')
			validation_loss = avg_loss
			validation_top1_err = 1 - correct_vals / total

	fd_losslog.flush()
	fd_losslog.close()
	return model

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train aritmetic model')
	parser.add_argument('--device', type=str, default='auto',
						help='Device to use (cpu, cuda, auto)')
	parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
	parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
	parser.add_argument('--hidden', type=int, default=256, help='Hidden dimension size')
	parser.add_argument('--heads', type=int, default=4, help='Number of attention heads')
	parser.add_argument('--attn', type=str, default='hypergraph', choices=['hypergraph', 'graph'],
						help='Attention implementation to use')
	parser.add_argument('--log-name', type=str,
						help='postfix logname')
	parser.add_argument('--save', action='store_true',
        help='Load and save model parameters.')
	parser.add_argument('--seq-l', type=int, default=1, help="sequence length")
	parser.add_argument('--task', type=int, default=1, help="what task to run. 1 = mod arith; 2 = copy task; 3 = formula generalization")
	parser.add_argument('--repl', type=int, default=1, help="what replicate this is",)
	parser.add_argument('--no-amp', action='store_true', help='Disable Torch automatic mixed precision')
	parser.add_argument('--use-cuda-kernels', action='store_true',
						help='Use CUDA kernels')
	args = parser.parse_args()

	SEQ_L = args.seq_l # sorry about the global
	
	model = trainModel(
		num_epochs=args.epochs,
		device=args.device,
		hidden_dim=args.hidden,
		n_heads=args.heads,
		task = args.task,
		attn_impl=args.attn,
		batch_size=args.batch_size,
		log_name=args.log_name,
		save_model=args.save,
		replicate = args.repl,
		no_amp = args.no_amp,
		use_cuda_kernels = args.use_cuda_kernels,
	)

