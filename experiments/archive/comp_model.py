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
script_dir = str(current_script_path.parent)
compositional_dir = str(current_script_path.parent.parent / 'compositional')
project_root = str(current_script_path.parent.parent.parent)
for d in [script_dir, compositional_dir, project_root]:
    if d not in sys.path:
        sys.path.insert(0, d)

from att3ntion import _HypergraphAttentionNaive, _GraphAttentionNaive, QuickGELU
from gen_data_comp import genData1, genData2, genData3, genData5

class SimpleCompModel(nn.Module):
	"""Model with hypergraph attention layer."""
	def __init__(self, hidden_dim:int, num_heads:int, n_layers:int, attn_impl:str='', n_recurse:int=1, modulo:int=11):
		super().__init__()
		input_dim = 40
		self.embedding_proj = nn.Linear(input_dim, hidden_dim)
		self.rotary_emb = RotaryEmbedding(dim = hidden_dim)
		self.attn_impl = attn_impl
		self.n_recurse = n_recurse
		
		self.repeated_layers = nn.ModuleList()
		for _ in range(n_layers):
			if attn_impl == "hypergraph":
				attention_layer = _HypergraphAttentionNaive(hidden_dim, num_heads)
			else:
				attention_layer = _GraphAttentionNaive(hidden_dim, num_heads)

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
		
		# self.op_classifier = nn.Linear(hidden_dim, 4)
		self.value_classifier = nn.Linear(hidden_dim, modulo)
		self.posenc_proj = nn.Linear(hidden_dim, 16)
		self.gelu = QuickGELU()
		
	def forward(self, x):
		x = self.embedding_proj(x)

		for r in range(self.n_recurse):
			for layer_block in self.repeated_layers:
				# attn_output = layer_block['attention'](x, self.rotary_emb)
				attn_output = layer_block['attention'](x, None)
				x = layer_block['norm1'](x + attn_output)
				ffn_output = layer_block['ffn'](x)
				x = layer_block['norm2'](x + ffn_output)
		
		value_pred = (self.value_classifier(x[:, -1]))
		posenc_pred = self.posenc_proj(x)
		return x, posenc_pred, value_pred

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

class CompModel(nn.Module):
	"""Model with hypergraph attention layer."""
	def __init__(self, hidden_dim:int, num_heads:int, n_layers:int, attn_impl:str=''):
		super().__init__()
		input_dim = 32
		self.embedding_proj = nn.Linear(input_dim, hidden_dim)
		self.rotary_emb = RotaryEmbedding(dim = hidden_dim)
		self.attn_impl = attn_impl

		self.repeated_layers = nn.ModuleList()
		for _ in range(n_layers):
			if attn_impl == "hypergraph":
				attention_layer = _HypergraphAttentionNaive(hidden_dim, num_heads)
			else:
				attention_layer = _GraphAttentionNaive(hidden_dim, num_heads)

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
		self.gelu = QuickGELU()

	def forward(self, x):
		x = self.embedding_proj(x)

		for layer_block in self.repeated_layers:
			attn_output = layer_block['attention'](x, self.rotary_emb)
			x = layer_block['norm1'](x + attn_output)
			ffn_output = layer_block['ffn'](x)
			x = layer_block['norm2'](x + ffn_output)

		value_pred = ( self.value_classifier(x[:, -1]) ) # FIXME replaces the op
		return value_pred

	def save_model(self, path: str):
		"""Saves the model's configuration and state dictionary."""
		torch.save(self.state_dict(), path)
		print(f"saved model to {path}")

	def loadSimple(self, path:str, device):
		# init from file
		mdata = torch.load(path)
		self.load_state_dict(mdata, strict=False) # this won't fill everything

		if self.attn_impl == "hypergraph":

			print("Freezing weights for layer0 and layer2")
			for param_name, param in self.named_parameters():
				if param_name.startswith("repeated_layers.0") or param_name.startswith("repeated_layers.2"):
					param.requires_grad = False
					print(f"  Froze: {param_name}")
				else:
					param.requires_grad = True # Ensure other layers (like layer2 and output_layer) are trainable
					print(f"  Trainable: {param_name}")
			with torch.no_grad():
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

		else:

			print("Freezing weights for layers 0, 1, 4, 5")
			for param_name, param in self.named_parameters():
				if (param_name.startswith("repeated_layers.0") or param_name.startswith("repeated_layers.1")) or ( param_name.startswith("repeated_layers.4") or
				param_name.startswith("repeated_layers.5")):
					param.requires_grad = False
					print(f"  Froze: {param_name}")
				else:
					param.requires_grad = True # Ensure other layers (like layer2 and output_layer) are trainable
					print(f"  Trainable: {param_name}")

			with torch.no_grad():
				for i in range(2):
					self.repeated_layers[4+i].attention.Wq.weight.copy_( mdata[f"repeated_layers.{i}.attention.Wq.weight"])
					self.repeated_layers[4+i].attention.Wk.weight.copy_( mdata[f"repeated_layers.{i}.attention.Wk.weight"])

					self.repeated_layers[4+i].attention.Wv.weight.copy_( mdata[f"repeated_layers.{i}.attention.Wv.weight"])
					self.repeated_layers[4+i].attention.Wv.bias.copy_( mdata[f"repeated_layers.{i}.attention.Wv.bias"])

					self.repeated_layers[4+i].attention.Wo.weight.copy_( mdata[f"repeated_layers.0.attention.Wo.weight"])
					self.repeated_layers[4+i].attention.Wo.bias.copy_( mdata[f"repeated_layers.0.attention.Wo.bias"])

					self.repeated_layers[4+i].norm1.weight.copy_( mdata[f"repeated_layers.{i}.norm1.weight"])
					self.repeated_layers[4+i].norm1.bias.copy_( mdata[f"repeated_layers.{i}.norm1.bias"])
					self.repeated_layers[4+i].norm2.weight.copy_( mdata[f"repeated_layers.{i}.norm1.weight"])
					self.repeated_layers[4+i].norm2.bias.copy_( mdata[f"repeated_layers.{i}.norm1.bias"])

					self.repeated_layers[4+i].ffn[0].weight.copy_( mdata[f"repeated_layers.{i}.ffn.0.weight"])
					self.repeated_layers[4+i].ffn[0].bias.copy_( mdata[f"repeated_layers.{i}.ffn.0.bias"])
					self.repeated_layers[4+i].ffn[2].weight.copy_( mdata[f"repeated_layers.{i}.ffn.2.weight"])
					self.repeated_layers[4+i].ffn[2].bias.copy_( mdata[f"repeated_layers.{i}.ffn.2.bias"])

		self.to(device)
		self.eval() # Set to evaluation mode by default
		return

	def load_model(path: str, device):
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
		print(f"CompModel {self.attn_impl}: number of model parameters:{trainable_params/1e6}M")

def prepare_data(data_tensor, device, modulo):
	inputs = data_tensor.copy()
	# Mask the value
	inputs[:, -1, :] = 0
	
	value_targets = np.argmax(data_tensor[:, -1, 5:5+modulo], axis=1)
	
	return (torch.FloatTensor(inputs).to(device), 
			torch.LongTensor(value_targets).to(device))

def prepare_data_posenc(data_tensor, device, modulo):
	inputs = data_tensor.copy()
	# Mask the parse posenc
	inputs[:, :, -16:] = 0
	# mask the last token value (but not position)
	inputs[:, -1, 5:modulo+5] = 0

	# Extract targets
	value_targets = np.argmax(data_tensor[:, -1, 5:5+modulo], axis=-1)
	pos_targets = data_tensor[:, :, -16:]

	return (torch.FloatTensor(inputs).to(device), \
			torch.FloatTensor(pos_targets).to(device), \
			torch.LongTensor(value_targets).to(device))

def train_model1(num_epochs, batch_size, hidden_dim, num_heads, device, modulo, attn_impl=""):
	
	if device == 'auto':
		if torch.cuda.is_available():
			device = torch.device('cuda')
		else:
			device = torch.device('cpu')
	else:
		device = torch.device(device)
	
	print(f"Using device: {device}")
	data = genData1(batch_size * 100, modulo, do_print=False)
	dataset = TensorDataset(torch.tensor(data))
	train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

	if attn_impl == "hypergraph":
		n_layers = 2
	else:
		n_layers = 4

	model = SimpleCompModel(hidden_dim, num_heads, n_layers=n_layers, attn_impl=attn_impl, n_recurse=3, modulo=modulo).to(device)
	try:
		model.load_model(f"comp_model_{attn_impl}.pt", device)
	except:
		print("train_model1: could not load the saved model weights")
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
	criterion_ce = nn.CrossEntropyLoss() # NOTE
	criterion_mse = nn.MSELoss()
	model.printParamCount()

	bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
	print(f"Bfloat16 supported: {bf16_supported}")
	print("\n--- Running with Automatic Mixed Precision ---")


	fd_losslog = open(f'losslog_trainModel1_{attn_impl}.txt', 'w')

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

		
		for batch_indx, (inputs_np,) in enumerate(train_loader):
			inputs, pos_targets, value_targets = prepare_data_posenc(inputs_np.numpy(), device, modulo)

			if batch_indx % 100 == 0:
				start_event.record()
			optimizer.zero_grad()

			with autocast('cuda', dtype=torch.bfloat16):
				outputs, pos_pred, value_pred = model(inputs)
				loss = 10*criterion_mse(pos_pred, pos_targets) + \
					0.2*criterion_ce(value_pred, value_targets)
				# loss = criterion_ce(value_pred, value_targets)
			# value_pred = model(inputs)
			# loss = (criterion(value_pred, targets))
			
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

			# Calculate accuracies
			correct_vals += (torch.argmax(value_pred, dim=1) == value_targets).sum().item()
		
		avg_loss = total_loss / len(train_loader)
		val_accuracy = 100 * correct_vals / total
		print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Result Acc: {val_accuracy:.2f}%')

		# save after each epoch
		model.save_model(f"comp_model_{args.attn_impl}.pt")

		# visualize it
		if False:
			fig,ax = plt.subplots(2,2, figsize=(12,9))
			ax[0,0].imshow(pos_pred.detach().float().cpu().numpy()[0,:,:])
			ax[0,0].set_title("pos_pred")
			ax[0,1].imshow(outputs.detach().float().cpu().numpy()[0,:,:])
			ax[0,1].set_title("outputs")
			ax[1,0].imshow(pos_targets.detach().float().cpu().numpy()[0,:,:])
			ax[1,0].set_title("pos_targets")
			ax[1,1].plot(value_pred.detach().float().cpu().numpy()[0,:], 'r')
			ax[1,1].plot(value_targets.detach().float().cpu().numpy()[0],1, 'ko')
			ax[1,1].set_title("value pred and target")
			plt.show()

	fd_losslog.close()
	return model

def train_model2(num_epochs, batch_size, hidden_dim, num_heads, device='cpu', modulo=23, attn_impl=""):

	if device == 'auto':
		if torch.cuda.is_available():
			device = torch.device('cuda')
		else:
			device = torch.device('cpu')
	else:
		device = torch.device(device)

	print(f"Using device: {device}")

	data = genData2(batch_size * 1000, modulo)
	dataset = TensorDataset(torch.tensor(data))
	train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

	if attn_impl == "hypergraph":
		n_layers = 3
	else:
		n_layers = 6

	model = CompModel(hidden_dim, num_heads, n_layers=n_layers, attn_impl=attn_impl)
	model.loadSimple(f"comp_model_{attn_impl}.pt", device)
	model.to(device)
	trainable_params = filter(lambda p: p.requires_grad, model.parameters())
	optimizer = torch.optim.Adam(trainable_params, lr=0.001)
	criterion = nn.CrossEntropyLoss()
	model.printParamCount()
	model = torch.compile(model) # mode="max-autotune"

	fd_losslog = open(f'losslog_trainModel2_{attn_impl}.txt', 'w')

	print("\ntrain_model2 started...")
	uu = 0
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

			lloss = loss.detach().cpu().item()
			fd_losslog.write(f"{uu}\t{lloss}\t0.0\n")
			fd_losslog.flush()
			uu += 1

			total_loss += loss.item()
			total += inputs.size(0)

			# Calculate accuracies
			correct_vals += (torch.argmax(value_pred, dim=1) == value_targets).sum().item()

		avg_loss = total_loss / len(train_loader)
		val_accuracy = 100 * correct_vals / total
		print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Result Acc: {val_accuracy:.2f}%')

	fd_losslog.close()
	return model

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train analogy model')
	parser.add_argument('--device', type=str, default='auto',
						help='Device to use (cpu, cuda, auto)')
	parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
	parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
	parser.add_argument('--modulo', type=int, default=11, help='Modulo for arithmetic operations')
	parser.add_argument('--hidden-dim', type=int, default=96, help='Hidden dimension size')
	parser.add_argument('--num-heads', type=int, default=4, help='Number of attention heads')
	parser.add_argument('--attn-impl', type=str, default='hypergraph', choices=['hypergraph', 'graph'],
						help='Attention implementation to use')
	args = parser.parse_args()
	
	torch.manual_seed(42)
	np.random.seed(42)
	
	# print("Example data points:")
	# print(genData(3, args.modulo, do_print=True))
	
	model = train_model1(
		num_epochs=args.epochs,
		device=args.device,
		modulo=args.modulo,
		hidden_dim=args.hidden_dim,
		num_heads=args.num_heads,
		attn_impl=args.attn_impl,
		batch_size=args.batch_size
	)
	model.save_model(f"comp_model_{args.attn_impl}.pt")
 #
	# model = train_model2(
	# 	num_epochs=10,
	# 	device=args.device,
	# 	modulo=args.modulo,
	# 	hidden_dim=args.hidden_dim,
	# 	num_heads=args.num_heads,
	# 	attn_impl=args.attn_impl,
	# 	batch_size=args.batch_size
	# )

