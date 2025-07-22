import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast
from rotary_embedding_torch import RotaryEmbedding
import matplotlib.pyplot as plt

import sys
from pathlib import Path
current_script_path = Path(__file__).resolve()
parent_dir = current_script_path.parent.parent
parent_dir_str = str(parent_dir)
if parent_dir_str not in sys.path:
    sys.path.insert(0, parent_dir_str)

from hyper_attn_pytorch import HypergraphAttention_Naive, GraphAttention_Naive, QuickGELU
# from hyper_attn_cpp_wrapper import HypergraphAttentionCPP
from gen_data_comp import genData7
import pdb

class SimpleCompModel(nn.Module):
	"""Model with hypergraph attention layer."""
	def __init__(self, hidden_dim:int, num_heads:int, n_layers:int, attn_impl:str='', n_recurse:int=1, modulo:int=11):
		super().__init__()
		self.input_dim = 64
		self.embedding_proj = nn.Linear(self.input_dim, hidden_dim)
		self.rotary_emb = RotaryEmbedding(dim = hidden_dim)
		self.attn_impl = attn_impl
		self.n_recurse = n_recurse
		self.d_model = hidden_dim
		
		self.repeated_layers = nn.ModuleList()
		for _ in range(n_layers):
			if attn_impl == "hypergraph":
				attention_layer = HypergraphAttention_Naive(hidden_dim, num_heads)
			else:
				attention_layer = GraphAttention_Naive(hidden_dim, num_heads)

			norm1_layer = nn.LayerNorm(hidden_dim)
			ffn_layer = nn.Sequential(
				nn.Linear(hidden_dim, 4 * hidden_dim),
				nn.ReLU(),
				nn.Linear(4 * hidden_dim, hidden_dim)
			)
			norm2_layer = nn.LayerNorm(hidden_dim)

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
		
	def forward(self, x):
		x = self.embedding_proj(x)

		for r in range(self.n_recurse):
			'''allocation module:
			each token has access to one extra token per
			full pass through the network.
			'''
			with torch.no_grad():
				decode = self.output_proj(x)
				# decode the least active tokens from the first index
				# of the decoded latents.
				indx = torch.sort(decode[:,:,0].squeeze(), dim=-1, descending=False)
				indx = indx
			for layer_block in self.repeated_layers:
				# attn_output = layer_block['attention'](x, self.rotary_emb)
				attn_output = layer_block['attention'](x, None)
				x = layer_block['norm1'](x + attn_output)
				ffn_output = layer_block['ffn'](x)
				x = layer_block['norm2'](x + ffn_output)
		
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
				f += bs * ntok * self.d_model**2 * 4 * 2 # ffn
				f += bs * ntok * self.d_model * 10 # layerNorm 2
		f += bs * ntok * self.d_model**2 * self.input_dim # output proj
		return f

def train_model1(num_epochs, batch_size, hidden_dim, num_heads, device, modulo, attn_impl=""):
	
	if device == 'auto':
		if torch.cuda.is_available():
			device = torch.device('cuda')
		else:
			device = torch.device('cpu')
	else:
		device = torch.device(device)
	
	print(f"Using device: {device}")
	
	x, y = genData7(batch_size * 1000, do_print=False)
	dataset = TensorDataset(torch.tensor(x), torch.tensor(y))
	train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

	x_v, y_v = genData7(batch_size * 1000, do_print=False)
	dataset_v = TensorDataset(torch.tensor(x_v), torch.tensor(y_v))
	loader_v = DataLoader(dataset_v, batch_size=batch_size, shuffle=True)

	if attn_impl == "hypergraph":
		n_layers = 4
	else:
		n_layers = 8

	model = SimpleCompModel(hidden_dim, num_heads, n_layers=n_layers, attn_impl=attn_impl, n_recurse=1, modulo=modulo).to(device)
	try:
		model.load_model(f"comp_model_{attn_impl}.pt", device)
	except:
		print("train_model1: could not load the saved model weights")
	optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, amsgrad=True)
	criterion_ce = nn.CrossEntropyLoss(reduction='none')
	criterion_mse = nn.MSELoss()
	model.printParamCount()
	model = torch.compile(model) # mode="max-autotune"

	bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
	print(f"Bfloat16 supported: {bf16_supported}")
	print("--- Running with Automatic Mixed Precision ---")


	fd_losslog = open(f'losslog_{attn_impl}.txt', 'w')

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
			value_targets = torch.FloatTensor(outputs_np[:,:, 1:32]).to(device)
			value_targets = value_targets.permute(0, 2, 1) # for cross eentropy loss - it measures CE over axis 1

			if batch_indx % 100 == 0:
				start_event.record()
			optimizer.zero_grad()

			with autocast('cuda', dtype=torch.bfloat16):
				pred = model(inputs)
				# posenc loss
				loss = torch.mean( criterion_mse(pred[:,:,-32:], targets[:,:,-32:]) * targets[:,:,0] )
				# value and op loss
				loss += torch.mean(criterion_ce(pred[:,:,1:32].permute(0,2,1), value_targets) * targets[:,:,0])
				# permute is to work with pytorch's cross-entropy calc
				# for both, mask off unused tokens

			loss.backward()
			optimizer.step()

			lloss = loss.detach().cpu().item()
			fd_losslog.write(f"{uu}\t{lloss}\t0.0\n")
			fd_losslog.flush()
			uu += 1

			total_loss += loss.item()
			total += inputs.size(0)
			
			if batch_indx % 100 == 0:
				end_event.record()
				torch.cuda.synchronize()
				amp_time = start_event.elapsed_time(end_event)
				print("batch time:", amp_time, "ms")
		
		avg_loss = total_loss / len(train_loader)
		val_accuracy = 100 * correct_vals / total
		print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Result Acc: {val_accuracy:.2f}%')

		# plot the inputs / outputs
		if False:
			fig,axs = plt.subplots(2, 4, figsize=(13,10))
			def mangle(t):
				return t.detach().cpu().squeeze().float().numpy()
			def plot(r, c, t):
				g = mangle(t)
				im = axs[r,c].imshow( g )
				plt.colorbar(im, ax=axs[r,c])

			for j in range(2):
				plot(j,0, inputs[j,...])
				plot(j,1, targets[j,...])
				plot(j,2, pred[j,...])
				plot(j,3, pred[j,...] - targets[j,...])

				axs[j,0].set_title('input')
				axs[j,1].set_title('target')
				axs[j,2].set_title('pred')
				axs[j,3].set_title('err')
			plt.show()

		# save after each epoch
		model.save_model(f"comp_model_{args.attn}.pt")

	# validation!
	total_loss = 0
	with torch.no_grad():
		for batch_indx, (inputs_np,outputs_np) in enumerate(loader_v):
			inputs = torch.FloatTensor(inputs_np).to(device)
			targets = torch.FloatTensor(outputs_np).to(device)
			value_targets = torch.FloatTensor(outputs_np[:,:, 1:32]).to(device)
			value_targets = value_targets.permute(0, 2, 1) # for cross eentropy loss - it measures CE over axis 1

			if batch_indx % 100 == 0:
				start_event.record()

			with autocast('cuda', dtype=torch.bfloat16):
				pred = model(inputs)
				# posenc loss
				loss = torch.mean( criterion_mse(pred[:,:,-32:], targets[:,:,-32:]) * targets[:,:,0] )
				# value and op loss
				loss += torch.mean(criterion_ce(pred[:,:,1:32].permute(0,2,1), value_targets) * targets[:,:,0])
				# permute is to work with pytorch's cross-entropy calc
				# for both, mask off unused tokens

			if batch_indx % 100 == 0:
				end_event.record()
				torch.cuda.synchronize()
				amp_time = start_event.elapsed_time(end_event)
				f = model.calcFlops(inputs)
				print("batch time:", amp_time, "ms")
				print(f"{(f/1e9) / (amp_time / 1000.0)} GFlops approx")

			lloss = loss.detach().cpu().item()
			fd_losslog.write(f"{uu}\t{lloss}\t0.0\n")
			fd_losslog.flush()
			uu += 1

			total_loss += loss.item()

		avg_loss = total_loss / len(train_loader)
		print(f'Validation Loss: {avg_loss:.4f},')

	fd_losslog.close()
	return model

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train analogy model')
	parser.add_argument('--device', type=str, default='auto',
						help='Device to use (cpu, cuda, auto)')
	parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
	parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
	parser.add_argument('--modulo', type=int, default=16, help='Modulo for arithmetic operations')
	parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension size')
	parser.add_argument('--heads', type=int, default=4, help='Number of attention heads')
	parser.add_argument('--attn', type=str, default='hypergraph', choices=['hypergraph', 'graph'],
						help='Attention implementation to use')
	args = parser.parse_args()

	print("This script tests the graph and hypergraph transformer on a one-digit multiply task, multi-digit add, and shift tasks.")
	
	model = train_model1(
		num_epochs=args.epochs,
		device=args.device,
		modulo=args.modulo,
		hidden_dim=args.hidden_dim,
		num_heads=args.heads,
		attn_impl=args.attn,
		batch_size=args.batch_size
	)

