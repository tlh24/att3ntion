import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse

import sys
from pathlib import Path
current_script_path = Path(__file__).resolve()
script_dir = str(current_script_path.parent)
project_root = str(current_script_path.parent.parent.parent)
for d in [script_dir, project_root]:
    if d not in sys.path:
        sys.path.insert(0, d)

from att3ntion import HypergraphAttention, _HypergraphAttentionTorch, _HypergraphAttentionNaive
from gen_data import genData


def _ensure_all_finite(named_tensors, epoch, batch_idx, stage):
	"""
	Utility helper that scans tensors for NaN/Inf so we can stop at the first bad batch.
	"""
	for name, tensor in named_tensors:
		if tensor is None or not torch.is_tensor(tensor):
			continue
		if not tensor.dtype.is_floating_point:
			continue
		if not torch.isfinite(tensor).all():
			raise FloatingPointError(
				f"Detected non-finite values in '{name}' during {stage} "
				f"(epoch {epoch+1}, batch {batch_idx+1})."
			)


class SimpleAnalogyModel(nn.Module):
	"""Simpler model with hypergraph attention layer."""
	def __init__(self, hidden_dim:int, num_heads:int, n_layers:int, attn_impl:str='pytorch'):
		super().__init__()
		input_dim = 32
		self.embedding_proj = nn.Linear(input_dim, hidden_dim)
		self.enable_forward_checks = False
		
		self.repeated_layers = nn.ModuleList()
		for _ in range(n_layers):
			# Select attention implementation for this layer
			if attn_impl == 'torch':
				attention_layer = _HypergraphAttentionNaive(hidden_dim, num_heads, head_subspaces=True)
			elif attn_impl == 'cuda':
				attention_layer = HypergraphAttention(hidden_dim, num_heads)
			elif attn_impl == 'torch_cpp':
				attention_layer = _HypergraphAttentionTorch(hidden_dim, num_heads)
			else:
				raise ValueError(f"Unknown attention implementation: {attn_impl}")

			norm1_layer = nn.LayerNorm(hidden_dim)
			ffn_layer = nn.Sequential(
				nn.Linear(hidden_dim, 3 * hidden_dim),
				nn.ReLU(),
				nn.Linear(3 * hidden_dim, hidden_dim)
			)
			norm2_layer = nn.LayerNorm(hidden_dim)

			# Store the components for this specific repeated block
			self.repeated_layers.append(
				nn.ModuleDict({
						'attention': attention_layer,
						'norm1': norm1_layer,
						'ffn': ffn_layer,
						'norm2': norm2_layer,
				})
				)
		
		self.op_classifier = nn.Linear(hidden_dim, 4)
		self.value_classifier = nn.Linear(hidden_dim, 28)
		
	def forward(self, x):
		x = self.embedding_proj(x)

		for layer_block in self.repeated_layers:
			attention_module = layer_block['attention']
			if isinstance(attention_module, _HypergraphAttentionNaive):
				attn_output = attention_module(x, None)
			else:
				attn_output = attention_module(x)
			res1 = x + attn_output
			if getattr(self, "enable_forward_checks", False):
				_ensure_all_finite(
					[("residual_after_attention", res1)],
					-1, -1, "forward residual 1"
				)
			x = layer_block['norm1'](res1)
			ffn_output = layer_block['ffn'](x)
			res2 = x + ffn_output
			if getattr(self, "enable_forward_checks", False):
				_ensure_all_finite(
					[("residual_after_ffn", res2)],
					-1, -1, "forward residual 2"
				)
			x = layer_block['norm2'](res2)
		
		# Predict operators at positions 1 and 5, and value at position 7
		op_pred1 = self.op_classifier(x[:, 1])
		op_pred5 = self.op_classifier(x[:, 5])
		value_pred = self.value_classifier(x[:, 7])
		return op_pred1, op_pred5, value_pred

def prepare_data(data_tensor, device):
	inputs = data_tensor.copy()
	# Mask the operators and final value that need to be predicted
	inputs[:, 1, :4] = 0  
	inputs[:, 5, :4] = 0  
	inputs[:, 7] = 0     
	
	# Extract targets
	op_targets = np.zeros((data_tensor.shape[0], 2), dtype=np.int64)
	op_targets[:, 0] = np.argmax(data_tensor[:, 1, :4], axis=1)
	op_targets[:, 1] = np.argmax(data_tensor[:, 5, :4], axis=1)
	value_targets = np.argmax(data_tensor[:, 7, 4:], axis=1)
	
	return (torch.FloatTensor(inputs).to(device), 
			torch.LongTensor(op_targets).to(device),
			torch.LongTensor(value_targets).to(device))

def train_model(num_epochs=100, batch_size=128, hidden_dim=128, num_heads=4,
				device='cpu', modulo=19, attn_impl='pytorch',
				detect_anomaly=False, check_finite=False, max_batches=None):
	
	if device == 'auto':
		if torch.cuda.is_available():
			device = torch.device('cuda')
		# elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
		# 	device = torch.device('mps')
		else:
			device = torch.device('cpu')
	else:
		device = torch.device(device)
	
	print(f"Using device: {device}")
	
	if detect_anomaly:
		torch.autograd.set_detect_anomaly(True)
		print("Autograd anomaly detection ENABLED – expect slower training.")
	
	data = genData(batch_size * 1000, modulo)
	dataset = TensorDataset(torch.tensor(data))
	train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

	model = SimpleAnalogyModel(hidden_dim, num_heads, n_layers=2, attn_impl=attn_impl).to(device)
	model.enable_forward_checks = check_finite
	optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
	scheduler = None
	if args.lr_schedule:
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
	
	criterion = nn.CrossEntropyLoss()
	
	print("\nTraining started...")
	for epoch in range(num_epochs):
		model.train()
		total_loss = 0
		correct_ops = 0
		correct_vals = 0
		total = 0
		
		for batch_idx, (inputs_np,) in enumerate(train_loader):
			if max_batches is not None and batch_idx >= max_batches:
				break
			inputs, op_targets, value_targets = prepare_data(inputs_np.numpy(), device)
			
			optimizer.zero_grad()
			
			if check_finite:
				_ensure_all_finite(
					[('inputs', inputs)],
					epoch, batch_idx, 'input preprocessing'
				)

			op_pred1, op_pred5, value_pred = model(inputs)
			
			if check_finite:
				_ensure_all_finite(
					[
						('op_pred1', op_pred1),
						('op_pred5', op_pred5),
						('value_pred', value_pred)
					],
					epoch, batch_idx, 'forward pass'
				)

			# Calculate losses
			loss = (criterion(op_pred1, op_targets[:, 0]) + 
				   criterion(op_pred5, op_targets[:, 1]) + 
				   criterion(value_pred, value_targets))

			if check_finite:
				_ensure_all_finite(
					[('loss', loss)],
					epoch, batch_idx, 'loss computation'
				)
			
			loss.backward()
			# Gradient clipping to stabilise training
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			optimizer.step()
			if scheduler is not None:
				scheduler.step()

			if check_finite:
				_ensure_all_finite(
					[(f'{name}.grad', param.grad) for name, param in model.named_parameters()],
					epoch, batch_idx, 'backward pass'
				)

			if check_finite:
				_ensure_all_finite(
					[(name, param) for name, param in model.named_parameters()],
					epoch, batch_idx, 'optimizer step'
				)
			
			total_loss += loss.item()
			total += inputs.size(0)
			
			# Calculate accuracies
			correct_ops += ((torch.argmax(op_pred1, dim=1) == op_targets[:, 0]).sum().item() +
						  (torch.argmax(op_pred5, dim=1) == op_targets[:, 1]).sum().item()) / 2
			correct_vals += (torch.argmax(value_pred, dim=1) == value_targets).sum().item()
		
		if (epoch + 1) % 1 == 0:
			avg_loss = total_loss / len(train_loader)
			op_accuracy = 100 * correct_ops / total
			val_accuracy = 100 * correct_vals / total
			print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Op Acc: {op_accuracy:.2f}%, Result Acc: {val_accuracy:.2f}%')
	
	return model

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train analogy model')
	parser.add_argument('--device', type=str, default='auto', choices=['cpu', 'cuda', 'auto'],
						help='Device to use (cpu, cuda, auto)')
	parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
	parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
	parser.add_argument('--modulo', type=int, default=19, help='Modulo for arithmetic operations')
	parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension size')
	parser.add_argument('--num-heads', type=int, default=4, help='Number of attention heads')
	parser.add_argument('--attn-impl', type=str, default='torch', choices=['torch', 'cuda', 'torch_cpp'],
						help='Attention implementation to use')
	parser.add_argument('--max-batches', type=int, default=None,
						help='Max batches per epoch for quick debug runs')
	parser.add_argument('--lr', type=float, default=3e-4, help='Base learning rate')
	parser.add_argument('--lr-schedule', action='store_true', help='Use cosine annealing LR schedule')
	parser.add_argument('--detect-anomaly', action='store_true',
						help='Enable torch.autograd anomaly detection to pinpoint the CUDA op that produced NaNs/Infs.')
	parser.add_argument('--check-finite', action='store_true',
						help='Scan inputs/outputs/grads every batch and raise as soon as non-finite values appear.')
	args = parser.parse_args()

	if args.detect_anomaly or args.check_finite:
		os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
		print("CUDA_LAUNCH_BLOCKING=1 (synchronous kernel launches for debugging)")
	
	torch.manual_seed(42)
	np.random.seed(42)
	
	# print("Example data points:")
	# print(genData(3, args.modulo, do_print=True))
	
	model = train_model(
		num_epochs=args.epochs,
		device=args.device,
		modulo=args.modulo,
		hidden_dim=args.hidden_dim,
		num_heads=args.num_heads,
		attn_impl=args.attn_impl,
		batch_size=args.batch_size,
		max_batches=args.max_batches,
		detect_anomaly=args.detect_anomaly,
		check_finite=args.check_finite
	)
