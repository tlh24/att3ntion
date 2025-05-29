import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
from rotary_embedding_torch import RotaryEmbedding

import sys
from pathlib import Path
current_script_path = Path(__file__).resolve()
parent_dir = current_script_path.parent.parent
parent_dir_str = str(parent_dir)
if parent_dir_str not in sys.path:
    sys.path.insert(0, parent_dir_str)

from hyper_attn_pytorch import HypergraphAttention_Naive
# from hyper_attn_cpp_wrapper import HypergraphAttentionCPP
from gen_data_comp import genData1, genData2
import pdb

class SimpleCompModel(nn.Module):
	"""Model with hypergraph attention layer."""
	def __init__(self, hidden_dim:int, num_heads:int, n_layers:int, attn_impl:str='pytorch'):
		super().__init__()
		input_dim = 32
		self.embedding_proj = nn.Linear(input_dim, hidden_dim)
		self.rotary_emb = RotaryEmbedding(dim = hidden_dim)
		
		self.repeated_layers = nn.ModuleList()
		for _ in range(n_layers):
			attention_layer = HypergraphAttention_Naive(hidden_dim, num_heads)

			norm1_layer = nn.LayerNorm(hidden_dim)
			ffn_layer = nn.Sequential(
				nn.Linear(hidden_dim, 3 * hidden_dim),
				nn.ReLU(),
				nn.Linear(3 * hidden_dim, hidden_dim)
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
		
		# self.op_classifier = nn.Linear(hidden_dim, 4)
		self.value_classifier = nn.Linear(hidden_dim, 28)
		
	def forward(self, x):
		x = self.embedding_proj(x)

		for layer_block in self.repeated_layers:
			attn_output = layer_block['attention'](x, self.rotary_emb)
			x = layer_block['norm1'](x + attn_output)
			ffn_output = layer_block['ffn'](x)
			x = layer_block['norm2'](x + ffn_output)
		
		value_pred = self.value_classifier(x[:, 3])
		return value_pred

	def save_model(self, path: str):
		"""Saves the model's configuration and state dictionary."""
		torch.save(self.state_dict(), path)
		print(f"saved model to {path}")

	@classmethod
	def load_model(cls, path: str, device):
		"""Loads a model from a file."""
		checkpoint = torch.load(path, map_location=device)
		model.load_state_dict(checkpoint)
		model.to(device)
		model.eval() # Set to evaluation mode by default
		return model

class CompModel(nn.Module):
	"""Model with hypergraph attention layer."""
	def __init__(self, hidden_dim:int, num_heads:int, n_layers:int, attn_impl:str='pytorch'):
		super().__init__()
		input_dim = 32
		self.embedding_proj = nn.Linear(input_dim, hidden_dim)
		self.rotary_emb = RotaryEmbedding(dim = hidden_dim)

		self.repeated_layers = nn.ModuleList()
		for _ in range(n_layers):
			attention_layer = HypergraphAttention_Naive(hidden_dim, num_heads)

			norm1_layer = nn.LayerNorm(hidden_dim)
			ffn_layer = nn.Sequential(
				nn.Linear(hidden_dim, 3 * hidden_dim),
				nn.ReLU(),
				nn.Linear(3 * hidden_dim, hidden_dim)
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

		self.value_classifier = nn.Linear(hidden_dim, 28)

	def forward(self, x):
		x = self.embedding_proj(x)

		for layer_block in self.repeated_layers:
			attn_output = layer_block['attention'](x, self.rotary_emb)
			x = layer_block['norm1'](x + attn_output)
			ffn_output = layer_block['ffn'](x)
			x = layer_block['norm2'](x + ffn_output)

		value_pred = self.value_classifier(x[:, -1])
		return value_pred

	def save_model(self, path: str):
		"""Saves the model's configuration and state dictionary."""
		torch.save(self.state_dict(), path)
		print(f"saved model to {path}")

	def loadSimple(self, path:str, device):
		# init from file
		mdata = torch.load("comp_model.pt")
		self.load_state_dict(mdata, strict=False) # this won't fill everything

		print("Freezing weights for layer1 and layer3")
		for param_name, param in self.named_parameters():
			if param_name.startswith("repeated_layers.0") or param_name.startswith("repeated_layers.2"):
				param.requires_grad = False
				print(f"  Froze: {param_name}")
			else:
				param.requires_grad = True # Ensure other layers (like layer2 and output_layer) are trainable
				print(f"  Trainable: {param_name}")

		self.repeated_layers[2].attention.Wq.weight.copy_( mdata["repeated_layers.0.attention.Wq.weight"])
		self.repeated_layers[2].attention.Wr.weight.copy_( mdata["repeated_layers.0.attention.Wr.weight"])
		self.repeated_layers[2].attention.Ws.weight.copy_( mdata["repeated_layers.0.attention.Ws.weight"])

		self.repeated_layers[2].attention.Wv_q.weight.copy_( mdata["repeated_layers.0.attention.Wv_q.weight"])
		self.repeated_layers[2].attention.Wv_q.bias.copy_( mdata["repeated_layers.0.attention.Wv_q.bias"])
		self.repeated_layers[2].attention.Wv_q.weight.copy_( mdata["repeated_layers.0.attention.Wv_r.weight"])
		self.repeated_layers[2].attention.Wv_q.bias.copy_( mdata["repeated_layers.0.attention.Wv_r.bias"])
		self.repeated_layers[2].attention.Wv_q.weight.copy_( mdata["repeated_layers.0.attention.Wv_s.weight"])
		self.repeated_layers[2].attention.Wv_q.bias.copy_( mdata["repeated_layers.0.attention.Wv_s.bias"])

		self.repeated_layers[2].attention.Wo.weight.copy_( mdata["repeated_layers.0.attention.Wo.weight"])
		self.repeated_layers[2].attention.Wo.bias.copy_( mdata["repeated_layers.0.attention.Wo.bias"])

		self.repeated_layers[2].norm1.weight.copy_( mdata["repeated_layers.0.norm1.weight"])
		self.repeated_layers[2].norm1.bias.copy_( mdata["repeated_layers.0.norm1.bias"])
		self.repeated_layers[2].norm2.weight.copy_( mdata["repeated_layers.0.norm1.weight"])
		self.repeated_layers[2].norm2.bias.copy_( mdata["repeated_layers.0.norm1.bias"])

		self.repeated_layers[2].ffn[0].weight.copy_( mdata["repeated_layers.0.ffn.0.weight"])
		self.repeated_layers[2].ffn[0].bias.copy_( mdata["repeated_layers.0.ffn.0.bias"])
		self.repeated_layers[2].ffn[2].weight.copy_( mdata["repeated_layers.0.ffn.2.weight"])
		self.repeated_layers[2].ffn[2].bias.copy_( mdata["repeated_layers.0.ffn.2.bias"])

		self.to(device)
		self.eval() # Set to evaluation mode by default
		return


	def load_model(cls, path: str, device):
		"""Loads a model from a file."""
		checkpoint = torch.load(path, map_location=device)
		self.load_state_dict(checkpoint)
		self.to(device)
		self.eval() # Set to evaluation mode by default
		return

def prepare_data(data_tensor, device):
	inputs = data_tensor.copy()
	# Mask the value
	inputs[:, -1, :] = 0
	
	# Extract targets
	value_targets = np.argmax(data_tensor[:, -1, 4:], axis=1)
	
	return (torch.FloatTensor(inputs).to(device), 
			torch.LongTensor(value_targets).to(device))

def train_model(num_epochs=100, batch_size=128, hidden_dim=128, num_heads=4, device='cpu', modulo=19, attn_impl='pytorch'):
	
	if device == 'auto':
		if torch.cuda.is_available():
			device = torch.device('cuda')
		else:
			device = torch.device('cpu')
	else:
		device = torch.device(device)
	
	print(f"Using device: {device}")
	
	data = genData1(batch_size * 100, modulo)
	dataset = TensorDataset(torch.tensor(data))
	train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

	model = SimpleCompModel(hidden_dim, num_heads, n_layers=1, attn_impl=attn_impl).to(device)
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
			inputs, value_targets = prepare_data(inputs_np.numpy(), device)
			
			optimizer.zero_grad()
			value_pred = model(inputs)
			
			# Calculate losses
			loss = (criterion(value_pred, value_targets))
			
			loss.backward()
			optimizer.step()
			
			total_loss += loss.item()
			total += inputs.size(0)
			
			# Calculate accuracies
			correct_vals += (torch.argmax(value_pred, dim=1) == value_targets).sum().item()
		
		if (epoch + 1) % 10 == 0:
			avg_loss = total_loss / len(train_loader)
			val_accuracy = 100 * correct_vals / total
			print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Result Acc: {val_accuracy:.2f}%')
	
	return model

def train_model2(num_epochs=100, batch_size=128, hidden_dim=128, num_heads=4, device='cpu', modulo=19, attn_impl='pytorch'):

	if device == 'auto':
		if torch.cuda.is_available():
			device = torch.device('cuda')
		else:
			device = torch.device('cpu')
	else:
		device = torch.device(device)

	print(f"Using device: {device}")

	data = genData2(batch_size * 100, modulo)
	dataset = TensorDataset(torch.tensor(data))
	train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

	model = CompModel(hidden_dim, num_heads, n_layers=3, attn_impl=attn_impl)
	model.loadSimple("comp_model.pt", device)
	trainable_params = filter(lambda p: p.requires_grad, model.parameters())
	optimizer = torch.optim.Adam(trainable_params, lr=0.001)
	criterion = nn.CrossEntropyLoss()

	print("\nTraining started...")
	for epoch in range(num_epochs):
		model.train()
		total_loss = 0
		correct_ops = 0
		correct_vals = 0
		total = 0

		for batch_idx, (inputs_np,) in enumerate(train_loader):
			inputs, value_targets = prepare_data(inputs_np.numpy(), device)

			optimizer.zero_grad()
			value_pred = model(inputs)

			# Calculate losses
			loss = (criterion(value_pred, value_targets))

			loss.backward()
			optimizer.step()

			total_loss += loss.item()
			total += inputs.size(0)

			# Calculate accuracies
			correct_vals += (torch.argmax(value_pred, dim=1) == value_targets).sum().item()

		if (epoch + 1) % 10 == 0:
			avg_loss = total_loss / len(train_loader)
			val_accuracy = 100 * correct_vals / total
			print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Result Acc: {val_accuracy:.2f}%')

	return model

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train analogy model')
	parser.add_argument('--device', type=str, default='auto', choices=['cpu', 'cuda', 'auto'],
						help='Device to use (cpu, cuda, auto)')
	parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
	parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
	parser.add_argument('--modulo', type=int, default=19, help='Modulo for arithmetic operations')
	parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension size')
	parser.add_argument('--num-heads', type=int, default=4, help='Number of attention heads')
	parser.add_argument('--attn-impl', type=str, default='pytorch', choices=['pytorch', 'cpp'],
						help='Attention implementation to use')
	args = parser.parse_args()
	
	torch.manual_seed(42)
	np.random.seed(42)
	
	# print("Example data points:")
	# print(genData(3, args.modulo, do_print=True))
	
	# model = train_model(
	# 	num_epochs=args.epochs,
	# 	device=args.device,
	# 	modulo=args.modulo,
	# 	hidden_dim=args.hidden_dim,
	# 	num_heads=args.num_heads,
	# 	attn_impl=args.attn_impl,
	# 	batch_size=args.batch_size
	# )

	# model.save_model("comp_model.pt")

	model = train_model2(
		num_epochs=args.epochs,
		device=args.device,
		modulo=args.modulo,
		hidden_dim=args.hidden_dim,
		num_heads=args.num_heads,
		attn_impl=args.attn_impl,
		batch_size=args.batch_size
	)

