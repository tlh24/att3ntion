import argparse
from termcolor import colored
import torch
from torch import nn, optim
import numpy as np
import psgd
import utils
import pdb
import threading
import multiprocessing
from hyper_attn_pytorch import HypergraphAttention
from test import genData

gendata_dim = 32

class Transformer(nn.Module):
	def __init__(self, d_model:int, layers:int, repeat:int, n_head:int):
		super().__init__()
		self.d_model = d_model
		self.n_head = n_head
		self.layers = layers
		self.repeat = repeat
		self.resblocks = nn.ModuleList(\
			[HypergraphAttention(d_model, n_head) \
				for _ in range(layers)])
		self.in_proj = nn.Linear(gendata_dim, d_model, bias=True)
		self.out_proj = nn.Linear(d_model, gendata_dim, bias=True)

	def forward(self, x:torch.Tensor):
		# x is dtype int to interface with the embedding layer
		bs,ntok,inw = x.shape
		x = self.in_proj(x)
		# x = torch.cat((x, torch.zeros(bs, ntok, self.d_model - inw, device=x.device)), axis=-1)
		for i in range(self.repeat):
			for j, layer in enumerate(self.resblocks):
				y = layer(x)
				x = x + y
		return self.out_proj(x)

	def fixedInit(self):
		for layer in self.resblocks:
			layer.fixedInit()

	def printParamCount(self):
		trainable_params = sum(
			p.numel() for p in self.parameters() if p.requires_grad
		)
		print(f"Number of model parameters:{trainable_params}")

class SimpleModel(nn.Module):
	def __init__(self):
		super(SimpleModel, self).__init__()
		self.hyper_attn = HypergraphAttention(embedding_dim=32, num_heads=2)
		self.output = nn.Linear(32, 32)
		
	def forward(self, x):
		attn_output = self.hyper_attn(x)
		self.attn_output_shape = attn_output.shape
		return self.output(attn_output[:, 5]), self.output(attn_output[:, 7])

if __name__ == '__main__':
	print("Demonstrating Hypergraph Attention Module")
	print("----------------------------------------")
	
	parser = argparse.ArgumentParser()
	parser.add_argument('-t', action='store_true', help='make test data and print it')
	parser.add_argument('-b', type=int, default=64, help='batch size')
	parser.add_argument('-c', action='store_true', help='start fresh, dont load a model')
	parser.add_argument('-a', action='store_true', help='use AdamW')
	parser.add_argument('-v', action='store_true', help='validate only')
	cmd_args = parser.parse_args()
	
	if cmd_args.t: 
		batch_size = 10
		x, y = genData(batch_size, 7, do_print=True)
		exit()
		
	fd_losslog = open('losslog.txt', 'w')
	
	# this messes with pdb, but allows you to press enter to switch from training to validation.
	input_thread = threading.Thread(target=utils.monitorInput, daemon=True)
	input_thread.start()
	
	batch_size = cmd_args.b
	modulo = 7
	
	model = Transformer(d_model=64, layers=2, repeat=1, n_head=4)
	
	if cmd_args.c: 
		print(colored("not loading any model weights.", "blue"))
	else: 
		try: 
			model.load_state_dict(\
				torch.load('demo.pt',weights_only=True,map_location='cpu'))
			print(colored("loaded model.", "green"))
		except Exception as error:
			print(error)
			
	model = model.cuda()
	
	if cmd_args.a: 
		optimizer = optim.AdamW(model.parameters(), lr=2.5e-4, amsgrad=True)
	else: 
		optimizer = psgd.LRA(model.parameters(),\
			lr_params=0.01,lr_preconditioner= 0.01, momentum=0.9,\
			preconditioner_update_probability=0.25, \
			exact_hessian_vector_product=False, \
			rank_of_approximation=20, grad_clip_max_norm=5.0)
	
	def train(uu):
		sample_data = genData(8*2048, modulo, do_print=False)
		x = torch.tensor(sample_data, dtype=torch.float32)
		x = x.cuda()
		y = x.clone()
		# mask the op and variable F in the expressions: 
		#      0  1  2   3        4  5  6    7
		# {if} A `op B = C {then} D `op E = `F
		x[:,1,:] = 0
		x[:,5,:] = 0
		x[:,7,:] = 0
		
		for i in range(16*2000): # num iters
			indx = torch.randperm(x.shape[0])
			indx = indx[:batch_size]
			xx = x[indx,:,:]
			target = y[indx]
			
			if cmd_args.a: 
				optimizer.zero_grad()
				pred = model(xx)
				loss = torch.sum( (pred[:,:4,:] - target[:,:4,:])**2 )
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
				loss.backward()
				optimizer.step()
			else: 
				def closure():
					pred = model(xx)
					# only look at the last token
					loss = torch.sum( (pred[:,:4,:] - target[:,:4,:])**2 ) + \
						sum( \
							[torch.sum(5e-4 * torch.rand_like(param) * torch.abs(param) ) \
						for param in model.parameters()])
					return loss
				loss = optimizer.step(closure)
			lloss = loss.detach().cpu().item()
			if i % 10 == 0:
				print(lloss)
				fd_losslog.write(f'{uu}\t{lloss}\n')
				fd_losslog.flush()
			uu += 1
			if uu % 1000 == 0: 
				torch.save(model.state_dict(), 'demo.pt')
				print(colored('saved model', 'blue'))
			if utils.switch_to_validation:
				break
		return uu
		
	def test(uu): 
		sample_data = genData(4*2048, modulo)
		x = torch.tensor(sample_data, dtype=torch.float32)
		x = x.cuda()
		y = x.clone()
		# mask the op and variable F in the expressions: 
		#      0  1  2   3        4  5  6    7
		# {if} A `op B = C {then} D `op E = `F
		x[:,1,:] = 0
		x[:,5,:] = 0
		x[:,7,:] = 0
		
		for i in range(4*2048 // batch_size):
			indx = torch.arange(i*batch_size, (i+1)*batch_size)
			xx = x[indx,:,:]
			target = y[indx]
			pred = model(xx)
			loss = torch.sum( (pred[:,:4,:] - target[:,:4,:])**2 )
			lloss = loss.detach().cpu().item()
			print('v',lloss)
			fd_losslog.write(f'{uu}\t{lloss}\n')
			fd_losslog.flush()
			uu += 1

	uu = 0
	if not cmd_args.v: 
		uu = train(uu)
	test(uu)
	
	fd_losslog.close()
