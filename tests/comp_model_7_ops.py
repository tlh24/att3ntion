import numpy as np
import argparse
import inspect
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding
import matplotlib.pyplot as plt

import sys
from pathlib import Path
current_script_path = Path(__file__).resolve()
parent_dir = current_script_path.parent.parent
parent_dir_str = str(parent_dir)
if parent_dir_str not in sys.path:
    sys.path.insert(0, parent_dir_str)

from pure_pytorch_reference import HypergraphAttention_Naive, GraphAttention_Naive, QuickGELU
# from hypergraph_attention import HypergraphAttentionCPP
from gen_data_comp import genData3, genData4, genData7
import pdb

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
	def __init__(self, input_dim:int, hidden_dim:int, num_heads:int, n_layers:int, attn_impl:str='', n_recurse:int=1):
		super().__init__()
		self.input_dim = input_dim
		self.embedding_proj = nn.Linear(self.input_dim, hidden_dim)
		self.rotary_emb = RotaryEmbedding(dim = hidden_dim)
		self.attn_impl = attn_impl
		self.n_recurse = n_recurse
		self.d_model = hidden_dim
		
		self.repeated_layers = nn.ModuleList()
		for _ in range(n_layers):
			if attn_impl == "hypergraph":
				attention_layer = HypergraphAttention_Naive(hidden_dim, num_heads, head_subspaces=True)
			else:
				attention_layer = GraphAttention_Naive(hidden_dim, num_heads, head_subspaces=True)

			norm1_layer = nn.RMSNorm(hidden_dim) # was LayerNorm
			norm2_layer = nn.RMSNorm(hidden_dim)
			# norm1_layer = nn.LayerNorm(hidden_dim)
			# norm2_layer = nn.LayerNorm(hidden_dim)
			if True:
				ffn_layer = nn.Sequential(
					nn.Linear(hidden_dim, 3 * hidden_dim),
					nn.ReLU(),
					nn.Linear(3 * hidden_dim, hidden_dim)
				)
			else:
				ffn_layer = SwiGLU(hidden_dim, 2*hidden_dim, hidden_dim)
				# keep the same number of parameters.

			self.repeated_layers.append(
				nn.ModuleDict({
						'attention': attention_layer,
						'norm1': norm1_layer,
						'ffn': ffn_layer,
						'norm2': norm2_layer,
				})
				)
		self.output_proj = nn.Linear(hidden_dim, self.input_dim)
		self.gelu = QuickGELU()
		
	def forward(self, x, b):
		# skip = b % (self.n_recurse)
		skip = 0
		if skip > 0:
			with torch.no_grad():
				x = self.embedding_proj(x)
		else:
			x = self.embedding_proj(x)
		for r in range(self.n_recurse):
			if r < skip:
				with torch.no_grad():
					for layer_block in self.repeated_layers:
						# attn_output = layer_block['attention'](x, self.rotary_emb)
						xn = layer_block['norm1'](x) # PreNorm
						attn_output = layer_block['attention'](xn, None)
						x = x + attn_output
						xn = layer_block['norm2'](x)
						ffn_output = layer_block['ffn'](xn)
						x = x + ffn_output
			else:
				for layer_block in self.repeated_layers:
					# attn_output = layer_block['attention'](x, self.rotary_emb)
					xn = layer_block['norm1'](x)
					attn_output = layer_block['attention'](xn, None)
					x = x + attn_output
					xn = layer_block['norm2'](x)
					ffn_output = layer_block['ffn'](xn)
					x = x + ffn_output
					# attn_output = layer_block['attention'](x, None)
					# x = layer_block['norm1'](x + attn_output)
					# ffn_output = layer_block['ffn'](x)
					# x = layer_block['norm2'](x + ffn_output)
		
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
		bs, ntok, d_model = x.shape
		f = 0
		for r in range(self.n_recurse):
			for layer_block in self.repeated_layers:
				f += layer_block['attention'].calcFlops(x)
				f += bs * ntok * self.d_model * 10 # layerNorm 1
				f += bs * ntok * self.d_model**2 * 3 * 2 # ffn
				f += bs * ntok * self.d_model * 10 # layerNorm 2
		f += bs * ntok * self.d_model**2 * self.input_dim # output proj
		return f

	def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
		# this is from nanoGPT!
		# start with all of the candidate parameters
		param_dict = {pn: p for pn, p in self.named_parameters()}
		# filter out those that do not require grad
		param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
		# create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
		# i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
		decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
		nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
		optim_groups = [
			{'params': decay_params, 'weight_decay': weight_decay},
			{'params': nodecay_params, 'weight_decay': 0.0}
		]
		num_decay_params = sum(p.numel() for p in decay_params)
		num_nodecay_params = sum(p.numel() for p in nodecay_params)
		print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
		print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
		# Create AdamW optimizer and use the fused version if it is available
		fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
		use_fused = fused_available and device_type == 'cuda'
		extra_args = dict(fused=True) if use_fused else dict()
		extra_args = {**extra_args, 'amsgrad': False}
		optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
		# optimizer = torch.optim.NAdam(optim_groups, lr=learning_rate)
		print(f"using fused AdamW: {use_fused}")

		return optimizer


def calcLoss(task, pred, targets):
	n_correct = 0
	n_possible = 0
	bs = pred.shape[0]
	ntok = pred.shape[1]
	if task == 3:
		# mod arithmetic with pointer args
		value_targets = torch.argmax(targets[:,-1,5:40], axis=-1)
		loss = F.cross_entropy( pred[:,-1,5:40], value_targets)
		with torch.no_grad():
			n_correct = torch.sum(torch.argmax(pred[:,-1,5:40], axis=-1) == value_targets).item()
			n_possible = pred.shape[0]
	if task == 4:
		# can it calculate the parse-tree pointers?
		loss = F.mse_loss(pred[:,:,-16:], targets[:,:,-16:])
		value_targets = torch.argmax(targets[:,:,5:40], axis=-1)
		loss += 0.01* F.cross_entropy( \
			pred[:,:,5:40].permute(0,2,1), value_targets)
		with torch.no_grad():
			n_correct = torch.sum(torch.argmax(pred[:,:,5:40], axis=-1) == value_targets).item()
			n_possible = pred.shape[0] * pred.shape[1]
	if task == 7:
		value_targets = torch.argmax(targets[:,:,1:32], axis=2)
		# posenc & pointer loss
		loss = torch.sum(F.mse_loss(pred[:,:,32:], targets[:,:,32:], reduction='none') * targets[:,:,0].unsqueeze(-1))
		# value and op loss
		celoss = F.cross_entropy( \
			pred[:,:,1:32].permute(0,2,1), value_targets, reduction='none')
		loss += torch.sum(celoss * targets[:,:,0])
		# permute is to work with pytorch's cross-entropy calc
		# for both, mask off unused tokens
		loss += F.mse_loss(pred[:,:,0], targets[:,:,0]) # occupied flag
		with torch.no_grad():
			n_correct = torch.sum( \
				(torch.argmax(pred[:,:,1:32], axis=2) == value_targets) \
					* targets[:,:,0] ).item()
			n_possible = torch.sum( targets[:,:,0] ).item()
	return loss, n_correct, n_possible

def trainModel(num_epochs, batch_size, hidden_dim, num_heads, device, attn_impl="", log_name="", save_model=False, task=3, replicate=1):
	
	if device == 'auto':
		if torch.cuda.is_available():
			device = torch.device('cuda')
		else:
			device = torch.device('cpu')
	else:
		device = torch.device(device)
	
	print(f"Using device: {device}")
	
	if task == 3:
		gen_func = genData3
		nsamples = 1500
	if task == 4:
		gen_func = genData4
		nsamples = 1000
	if task == 7:
		gen_func = genData7
		nsamples = 6000 # longer to allow graph attention to converge.

	x, y = gen_func(batch_size * nsamples, do_print=False)
	x_v, y_v = gen_func(batch_size * nsamples, do_print=False, validation=True)

	dataset = TensorDataset(torch.tensor(x), torch.tensor(y))
	train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

	dataset_v = TensorDataset(torch.tensor(x_v), torch.tensor(y_v))
	loader_v = DataLoader(dataset_v, batch_size=batch_size, shuffle=True)

	input_dim = x.shape[2]

	if attn_impl == "hypergraph":
		n_layers = 3
	else:
		n_layers = 6 # match the number of parameters and (approx) model complexity.  (But not flops, of course!)
	n_recurse = 1

	if task == 4:
		n_recurse = 5
		if attn_impl == "hypergraph":
			n_layers = 2
		else:
			n_layers = 4 # Positive control: these converge at the same rate.

	model = SimpleCompModel(input_dim, hidden_dim, num_heads, n_layers=n_layers, attn_impl=attn_impl, n_recurse=n_recurse).to(device)
	if save_model:
		try:
			model.load_model(f"comp_model_{attn_impl}_r{replicate}.pt", device)
		except:
			print("train_model1: could not load the saved model weights")
	# from nanogpt:
	learning_rate = 6e-4 # max learning rate
	weight_decay = 1e-2 # karpathy 1e-1, default 1e-2
	beta1 = 0.9 # default 0.9, both.
	beta2 = 0.95 # karpathy 0.95, default 0.999
	# optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), 'cuda')
	optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, amsgrad=True)
	model.printParamCount()
	model = torch.compile(model) # mode="max-autotune"

	bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
	print(f"Bfloat16 supported: {bf16_supported}")
	print("--- Running with Automatic Mixed Precision ---")

	nam = {"hypergraph":"hg","graph":"g"}.get(attn_impl)
	fd_losslog = open(f'losslog_{nam}_t{task}_{log_name}_r{replicate}.txt', 'w')

	print("\ntrain_model1 started...")
	uu = 0
	for epoch in range(num_epochs):
		model.train()
		total_loss = 0
		correct_ops = 0
		correct_vals = 0
		total = 0

		start_event = torch.cuda.Event(enable_timing=True)
		end_event = torch.cuda.Event(enable_timing=True)
		
		for batch_indx, (inputs_np,outputs_np) in enumerate(train_loader):
			inputs = torch.FloatTensor(inputs_np).to(device)
			targets = torch.FloatTensor(outputs_np).to(device)

			if batch_indx % 100 == 0:
				start_event.record()
			optimizer.zero_grad()

			with autocast('cuda', dtype=torch.bfloat16):
				pred = model(inputs, batch_indx)
				loss, n_correct, n_possible = calcLoss(task, pred, targets)
				correct_vals += n_correct

			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optimizer.step()

			lloss = loss.detach().cpu().item()
			fd_losslog.write(f"{uu}\t{lloss}\t0.0\n")
			uu += 1

			total_loss += loss.item()
			total += n_possible
			
			if batch_indx % 200 == 0:
				end_event.record()
				torch.cuda.synchronize()
				amp_time = start_event.elapsed_time(end_event)
				print("batch time:", amp_time, "ms")
				fd_losslog.flush()

		# plot the inputs / outputs
		if epoch == (num_epochs-1) and False:
			fig,axs = plt.subplots(2, 4, figsize=(13,10))
			def mangle(t):
				return t.detach().cpu().squeeze().float().numpy()
			def plot(r, c, t, blank=False):
				g = mangle(t)
				if task == 4 and blank: # mask areas with no loss
					g[:,:5] = 0
					mx = np.max(g[:,5:40], axis=-1)
					mx = np.expand_dims(mx, -1)
					g[:,5:40] = g[:,5:40] / (mx+1)
				im = axs[r,c].imshow( g.T )
				plt.colorbar(im, ax=axs[r,c])

			for j in range(2):
				plot(j,0, inputs[j,...])
				plot(j,1, targets[j,...])
				plot(j,2, pred[j,...], True)
				plot(j,3, pred[j,...] - targets[j,...], True)

				axs[j,0].set_title('input')
				axs[j,1].set_title('target')
				axs[j,2].set_title('pred')
				axs[j,3].set_title('err')
			plt.show()
		
		avg_loss = total_loss / len(train_loader)
		train_accuracy = 100 * correct_vals / total
		print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Result Acc: {train_accuracy:.2f}%')

		if save_model:
			# save after each epoch
			model.save_model(f"comp_model_{args.attn}_r{replicate}.pt")

	fd_losslog.flush()
	# validation!
	total_loss = 0
	correct_vals = 0
	total = 0
	with torch.no_grad():
		for batch_indx, (inputs_np,outputs_np) in enumerate(loader_v):
			inputs = torch.FloatTensor(inputs_np).to(device)
			targets = torch.FloatTensor(outputs_np).to(device)
			value_targets = torch.FloatTensor(outputs_np[:,:, 1:32]).to(device)
			value_targets = value_targets.permute(0, 2, 1) # for cross eentropy loss - it measures CE over axis 1

			if batch_indx % 100 == 0:
				start_event.record()

			with autocast('cuda', dtype=torch.bfloat16):
				pred = model(inputs, 0)
				loss, n_correct, n_possible = calcLoss(task, pred, targets)
			correct_vals += n_correct

			total += n_possible

			if batch_indx % 200 == 0:
				end_event.record()
				torch.cuda.synchronize()
				amp_time = start_event.elapsed_time(end_event)
				f = model.calcFlops(inputs)
				print("batch time:", amp_time, "ms")
				print(f"{(f/1e9) / (amp_time / 1000.0)} GFlops approx")
				fd_losslog.flush()

			lloss = loss.detach().cpu().item()
			fd_losslog.write(f"{uu}\t{lloss}\t0.0\n")
			uu += 1

			total_loss += loss.item()

		avg_loss = total_loss / len(train_loader)
		val_accuracy = 100 * correct_vals / total
		print(f'Validation Loss: {avg_loss:.4f}, accuracy {val_accuracy}')

	fd_losslog.flush()
	fd_losslog.close()
	return model

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train analogy model')
	parser.add_argument('--device', type=str, default='auto',
						help='Device to use (cpu, cuda, auto)')
	parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
	parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
	parser.add_argument('--hidden', type=int, default=256, help='Hidden dimension size')
	parser.add_argument('--heads', type=int, default=8, help='Number of attention heads')
	parser.add_argument('--attn', type=str, default='hypergraph', choices=['hypergraph', 'graph'],
						help='Attention implementation to use')
	parser.add_argument('--log-name', type=str,
						help='postfix logname')
	parser.add_argument('--save', action='store_true',
        help='Load and save model parameters.')
	parser.add_argument('--task', type=int, help="what task to run the model on", required=True)
	parser.add_argument('--repl', type=int, default=1, help="what replicate this is",)
	args = parser.parse_args()
	
	model = trainModel(
		num_epochs=args.epochs,
		device=args.device,
		hidden_dim=args.hidden,
		num_heads=args.heads,
		attn_impl=args.attn,
		batch_size=args.batch_size,
		log_name=args.log_name,
		save_model=args.save,
		task = args.task,
		replicate = args.repl
	)

