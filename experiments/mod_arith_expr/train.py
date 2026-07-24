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
# current_script_path = Path(__file__).resolve()
# script_dir = str(current_script_path.parent)
# project_root = str(current_script_path.parent.parent.parent)
# for d in [script_dir, project_root]:
#     if d not in sys.path:
#         sys.path.insert(0, d)

from _naive import _HypergraphAttentionNaive, _GraphAttentionNaive, QuickGELU
from gen_data import gen_data, to_numpy, from_numpy

SEQ_L = 1 # length of the output sequence
MASKED = True # masked vs autoregressive training.
		# if False, use autoregressive training


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
	def __init__(self, input_vocab:int, hidden_dim:int, n_heads:int, n_layers:int, attn_impl:str='', n_recurse:int=1):
		super().__init__()
		self.input_vocab = input_vocab
		self.embedding_proj = nn.Embedding(input_vocab, hidden_dim)
		nn.init.normal_(self.embedding_proj.weight, std=0.02)
		self.attn_impl = attn_impl
		self.n_recurse = n_recurse
		self.d_model = hidden_dim
		self.rope = RotaryEmbedding(dim = hidden_dim//n_heads)

		self.repeated_layers = nn.ModuleList()
		for _ in range(n_layers):
			if attn_impl == "hypergraph":
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
		
	def forward(self, x, An):
		bs, ntok = x.shape
		mask = torch.tril(torch.ones(ntok, ntok, device=x.device))
		# mask = None # non-autoregressive now
		x = self.embedding_proj(x)
		An_list = []
		k = 0
		for r in range(self.n_recurse):
			for layer_block in self.repeated_layers:
				xn = layer_block['norm1'](x)
				if An is not None:
					attn_output, Aout = layer_block['attention'](xn, self.rope, mask, a_noise=An[k,...].squeeze())
				else:
					attn_output, Aout = layer_block['attention'](xn, self.rope, mask, a_noise=None)
				An_list.append(Aout)
				x = x + attn_output
				xn = layer_block['norm2'](x) # NOTE: could be attn_output
				ffn_output = layer_block['ffn'](xn)
				x = x + ffn_output
				k += 1
		return self.output_proj(x), torch.stack(An_list) if An is not None else None

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


def calcLoss(pred, targets):
	n_correct = 0
	n_possible = 0
	bs = pred.shape[0]
	ntok = pred.shape[1]
	seq_pred = pred[:, -SEQ_L:, :].reshape(-1, pred.shape[-1])
	seq_targets = targets[:, -SEQ_L:].long().reshape(-1)
	loss = F.cross_entropy(seq_pred, seq_targets, reduction='none')
	with torch.no_grad():
		n_correct = torch.sum(torch.argmax(seq_pred, dim=-1) == seq_targets).item()
		n_possible = seq_targets.shape[0]
	return loss, n_correct, n_possible

def trainModel(num_epochs, batch_size, hidden_dim, n_heads, device, task, attn_impl="", log_name="", save_model=False, replicate=1, no_amp=False, nsamp=0, sgd_steps=0, sgd_lr=0.1):
	
	if device == 'auto':
		if torch.cuda.is_available():
			device = torch.device('cuda')
		else:
			device = torch.device('cpu')
	else:
		device = torch.device(device)
	
	print(f"Using device: {device}")
	torch.set_float32_matmul_precision('high')
	
	if task == 1:
		# for grokking, need to generate the (nearly) full table
		train_s, test_s = gen_data('grok', max_d=1, V=2, C=0, P=113, L=SEQ_L, exact_v=True, data_size=120**2)
	if task == 2:
		# copy task
		train_s, test_s = gen_data('grok', max_d=0, V=3, C=0, P=113, L=SEQ_L, exact_v=False, data_size=120**2)
	if task == 3:
		train_s, test_s = gen_data('grok', max_d=1, V=3, C=2, P=113, L=SEQ_L, exact_v=False, data_size=100_000)

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
		n_layers = 3
		if attn_impl == "graph":
			n_layers = 6
		n_recurse = 1 # depends on the formula depth
		# n_heads = 6 # use the command line arg

	dtype = torch.float32
	model = SimpleCompModel(vocab_size, hidden_dim, n_heads, n_layers=n_layers, attn_impl=attn_impl, n_recurse=n_recurse).to(device=device, dtype=dtype)

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
	fd_losslog = open(f'losslog_{nam}_{log_name}_r{replicate}.txt', 'w')

	print("\ntrain_model1 started...")
	uu = 0
	rng = np.random.default_rng()
	validation_loss = 1.0
	validation_top1_err = 1.0
	t_loss_ema = 10.0
	a_loss_ema = 1.0

	for epoch in range(num_epochs):
		total_loss = 0
		correct_vals = 0
		total = 0

		start_event = torch.cuda.Event(enable_timing=True)
		end_event = torch.cuda.Event(enable_timing=True)
		model.train()
		n_train = x.shape[0]
		n_tok = x.shape[-1]
		# if epoch > 0:
		# 	torch.autograd.set_detect_anomaly(True)

		for batch_indx in range(n_train // batch_size):
			mb_idx = rng.permutation(n_train)[:batch_size]
			inputs = torch.tensor(x[mb_idx,:]).to(device=device)
			targets = torch.tensor(y[mb_idx,:]).to(device=device)

			if batch_indx == 0:
				d = inputs[0:5, :].detach().cpu().numpy()
				# print(d-1) # null token mapped to -1
				print(from_numpy(d-1, 113))
				print("targets:")
				d = targets[0:5, :].detach().cpu().numpy()
				print(from_numpy(d-1, 113))

			if batch_indx % ((32*512) // batch_size) == 0:
				start_event.record()
			optimizer.zero_grad()

			# with autocast('cuda'):
			if nsamp > 0:
				# run the sampling step - all nsamp candidates in one forward pass
				layers_repeat = n_layers * n_recurse
				inputs_tiled = inputs.repeat(nsamp, 1)
				targets_tiled = targets.repeat(nsamp, 1)
				# An = torch.randn((layers_repeat, nsamp * batch_size, n_heads, n_tok, n_tok, n_tok), device=device) * 0.25
				An = torch.poisson( torch.ones((layers_repeat, nsamp*batch_size, n_heads, n_tok, n_tok, n_tok), device=device)*0.05)
				
				if sgd_steps > 0:
					An.requires_grad = True
					sgd_l1 = 1/100.0  # static L1 coefficient to prevent overfitting
					for sgd_step in range(sgd_steps):
						pred, _ = model(inputs_tiled, An)
						loss, _, _ = calcLoss(pred, targets_tiled)
						total_loss = loss.sum() + sgd_l1 * An.abs().sum()
						grad_An, = torch.autograd.grad(total_loss, An)
						with torch.no_grad():
							An -= sgd_lr * grad_An
				
				with torch.no_grad():
					pred, _ = model(inputs_tiled, An)
					loss, _, _ = calcLoss(pred, targets_tiled)
					loss_per_element = loss.view(nsamp * batch_size, -1).mean(dim=1)
					loss_per_samp = loss_per_element.view(nsamp, batch_size)
					best_indices = torch.argmin(loss_per_samp, dim=0)  # (batch_size,)
					An_reshaped = An.view(layers_repeat, nsamp, batch_size, *An.shape[2:])
					idx = best_indices.view(1, 1, batch_size, *([1] * (An_reshaped.ndim - 3)))
					idx = idx.expand(layers_repeat, 1, batch_size, *An_reshaped.shape[3:])
					best_Ans = torch.gather(An_reshaped, 1, idx).squeeze(1).detach()
			else:
				best_Ans = None

			# now run the normal forward pass,
			# with attention noise added.
			pred, A = model(inputs, best_Ans) # returns the pre-noise attn
			t_loss, n_correct, n_possible = calcLoss(pred, targets)
			t_loss = t_loss.sum()

			if nsamp > 0:
				# need to center per sample
				with torch.no_grad():
					Ans_mean = torch.mean(best_Ans, dim=[3,4,5], keepdim=True)
					best_Ans = best_Ans - Ans_mean + A
					# An is added to atten, when calculating values.
				# only 'pull up' on attention; otherwise it collapses.
				# a_loss = torch.sum(((A - best_Ans) * (best_Ans > 0))**2)
				a_loss = F.mse_loss(A, best_Ans)
				t_loss_ema = t_loss.detach().cpu().item() * 0.01 + 0.99 * t_loss_ema
				a_loss_ema = a_loss.detach().cpu().item() * 0.01 + 0.99 * a_loss_ema
				scl = (t_loss_ema / (25*(a_loss_ema + 0.001))) # normalize the effect of a_loss
				scl = np.clip(scl, 0, 400)
				loss = t_loss + scl*a_loss
			else:
				loss = t_loss

			if torch.isnan(t_loss) or torch.isinf(t_loss):
				print(f"NaN/Inf in t_loss at batch {batch_indx}")
				pdb.set_trace()
			if nsamp > 0:
				if torch.isnan(a_loss) or a_loss < 1e-8:
					print(f"a_loss={a_loss.item():.2e}, scl={scl.item():.2e} at batch {batch_indx}")
					pdb.set_trace()

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
			
			if batch_indx % ((32*512) // batch_size) == 0:
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
						pred,_ = model(inputs, None)
						loss, n_correct, n_possible = calcLoss(pred, targets)
				correct_vals += n_correct
				total += n_possible
				lloss = loss.sum().detach().cpu().item()
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
	parser.add_argument('--task', type=int, default=3, help="what task to run. 1 = mod arith; 2 = copy task; 3 = formula generalization")
	parser.add_argument('--repl', type=int, default=1, help="what replicate this is",)
	parser.add_argument('--no-amp', action='store_true', help='Disable Torch automatic mixed precision')
	parser.add_argument('--nsamp', type=int, default=0,
						help='number of attention noise samlples')
	parser.add_argument('--sgd-steps', type=int, default=0,
						help='number of inner SGD steps to optimize attention noise')
	parser.add_argument('--sgd-lr', type=float, default=0.01,
						help='learning rate for inner SGD steps')
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
		nsamp = args.nsamp,
		sgd_steps=args.sgd_steps,
		sgd_lr=args.sgd_lr,
	)

