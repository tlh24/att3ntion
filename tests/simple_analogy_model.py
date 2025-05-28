import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse

import sys
from pathlib import Path
current_script_path = Path(__file__).resolve()
parent_dir = current_script_path.parent.parent
parent_dir_str = str(parent_dir)
if parent_dir_str not in sys.path:
    sys.path.insert(0, parent_dir_str)

from hyper_attn_full_pytorch import HypergraphAttention
from hyper_attn_cpp_wrapper import HypergraphAttentionCPP
from gen_data import genData
import pdb

class SimpleAnalogyModel(nn.Module):
	"""Simpler model with single hypergraph attention layer."""
	def __init__(self, hidden_dim:int, num_heads:int, attn_impl:str='pytorch'):
		super().__init__()
		input_dim = 32
		self.embedding_proj = nn.Linear(input_dim, hidden_dim)
		
		# Select attention implementation
		if attn_impl == 'pytorch':
			self.attention = HypergraphAttention(hidden_dim, num_heads)
		elif attn_impl == 'cpp':
			self.attention = HypergraphAttentionCPP(hidden_dim, num_heads)
		else:
			raise ValueError(f"Unknown attention implementation: {attn_impl}")
		
		self.norm1 = nn.LayerNorm(hidden_dim)
		self.ffn = nn.Sequential(
			nn.Linear(hidden_dim, 4 * hidden_dim),
			nn.ReLU(),
			nn.Linear(4 * hidden_dim, hidden_dim)
		)
		self.norm2 = nn.LayerNorm(hidden_dim)
		
		self.op_classifier = nn.Linear(hidden_dim, 4)
		self.value_classifier = nn.Linear(hidden_dim, 28)
		
	def forward(self, x):
		x = self.embedding_proj(x)
		
		attn_output = self.attention(x)
		x = self.norm1(x + attn_output)
		
		ffn_output = self.ffn(x)
		x = self.norm2(x + ffn_output)
		
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

def train_model(num_epochs=100, batch_size=128, hidden_dim=128, num_heads=4, device='cpu', modulo=20, attn_impl='pytorch'):
	
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
	
	data = genData(batch_size * 10, modulo)
	dataset = TensorDataset(torch.tensor(data))
	train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

	model = SimpleAnalogyModel(hidden_dim, num_heads, attn_impl=attn_impl).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	criterion = nn.CrossEntropyLoss()
	
	print("\nTraining started...")
	for epoch in range(num_epochs):
		model.train()
		total_loss = 0
		correct_ops = 0
		correct_vals = 0
		total = 0
		
		for batch_idx, (inputs_np,) in enumerate(train_loader):
			inputs, op_targets, value_targets = prepare_data(inputs_np.numpy(), device)
			
			optimizer.zero_grad()
			op_pred1, op_pred5, value_pred = model(inputs)
			
			# Calculate losses
			loss = (criterion(op_pred1, op_targets[:, 0]) + 
				   criterion(op_pred5, op_targets[:, 1]) + 
				   criterion(value_pred, value_targets))
			
			loss.backward()
			optimizer.step()
			
			total_loss += loss.item()
			total += inputs.size(0)
			
			# Calculate accuracies
			correct_ops += ((torch.argmax(op_pred1, dim=1) == op_targets[:, 0]).sum().item() +
						  (torch.argmax(op_pred5, dim=1) == op_targets[:, 1]).sum().item()) / 2
			correct_vals += (torch.argmax(value_pred, dim=1) == value_targets).sum().item()
		
		if (epoch + 1) % 10 == 0:
			avg_loss = total_loss / len(train_loader)
			op_accuracy = 100 * correct_ops / total
			val_accuracy = 100 * correct_vals / total
			print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Op Acc: {op_accuracy:.2f}%, Result Acc: {val_accuracy:.2f}%')
	
	return model

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train analogy model')
	parser.add_argument('--device', type=str, default='auto', choices=['cpu', 'cuda', 'auto'],
						help='Device to use (cpu, cuda, auto)')
	parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
	parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
	parser.add_argument('--modulo', type=int, default=17, help='Modulo for arithmetic operations')
	parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension size')
	parser.add_argument('--num-heads', type=int, default=4, help='Number of attention heads')
	parser.add_argument('--attn-impl', type=str, default='pytorch', choices=['pytorch', 'cpp'],
						help='Attention implementation to use')
	args = parser.parse_args()
	
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
		batch_size=args.batch_size
	)
