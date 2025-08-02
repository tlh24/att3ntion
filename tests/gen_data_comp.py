import numpy as np
import matplotlib.pyplot as plt
import random
import math
import argparse
import pdb

def randint(k):
	return np.random.randint(k)

def modOp(va, vb, op, md):
	vc0 = (va + vb) % md
	vc1 = (va - vb) % md
	vc2 = (va * vb) % md
	if vb == 0:
		vc3 = 0 # can't divide by zero!
	else:
		vc3 = (va // vb) % md

	match op:
		case 0:
			vc = vc0
			ops = '+'
		case 1:
			vc = vc1
			ops = '-'
		case 2:
			vc = vc2
			ops = '*'
		case 3:
			vc = vc3
			ops = '/'
	return vc, ops

def clipOp(va, vb, op, md):
	def clip(x):
		if x > md-1:
			return md-1
		if x < 0:
			return 0
		return x

	vc0 = clip(va + vb)
	vc1 = clip(va - vb)
	vc2 = clip(va * vb)
	if vb == 0:
		vc3 = 0 # can't divide by zero!
	else:
		vc3 = clip(va // vb)

	match op:
		case 0:
			vc = vc0
			ops = '+'
		case 1:
			vc = vc1
			ops = '-'
		case 2:
			vc = vc2
			ops = '*'
		case 3:
			vc = vc3
			ops = '/'
	return vc, ops

def genData1(bs, md, do_print=False):
	'''
	Task 1: learn the rules of modulo arithmetic.
	'''
	assert(md < 32-4)
	x = np.zeros((bs, 4, 32), dtype=int)

	for b in range(bs):
		va = randint(md)
		vb = randint(md)
		op = np.random.randint(4)
		vc, ops = modOp(va, vb, op, md)

		# encode.
		x[b,0,va+4] = 1
		x[b,1,op  ] = 1
		x[b,2,vb+4] = 1
		x[b,3,vc+4] = 1 # must be masked

		s = f"{va} {ops} {vb} = {vc}"
		if do_print:
			print(s)
	# endfor b
	return x

def genData2(bs, md, do_print=False):
	'''
	Task 2: learn to compose arithmetic.
	of the form out = (a op b) op (c op d)
	all ops +-*/ are modulo md.
	'''

	assert(md < 32-4)
	x = np.zeros((bs, 12, 32), dtype=int)

	for b in range(bs):
		# probably should use recursive generation here...
		va = np.zeros((2,), dtype=int)
		vb = np.zeros((2,), dtype=int)
		op = np.zeros((3,), dtype=int)
		for i in range(2):
			va[i] = randint(md)
			vb[i] = randint(md-1) + 1 # avoid divide by zero and noops
		for i in range(3):
			op[i] = np.random.randint(4)

		vc1, ops1 = modOp(va[0], vb[0], op[0], md)
		vc2, ops2 = modOp(va[1], vb[1], op[1], md)
		vc3, ops3 = modOp(vc1, vc2, op[2], md)

		# encode.
		x[b,0 ,  -1   ] = 1 # (
		x[b,1 ,va[0]+4] = 1
		x[b,2 ,op[0]  ] = 1
		x[b,3 ,vb[0]+4] = 1
		x[b,4 ,  -2   ] = 1 # )
		x[b,5 ,op[2]+4] = 1
		x[b,6 ,  -1   ] = 1 # (
		x[b,7 ,va[1]+4] = 1
		x[b,8 ,op[1]  ] = 1
		x[b,9 ,vb[1]+4] = 1
		x[b,10,  -2   ] = 1 # )
		x[b,11,vc3+4  ] = 1

		s = f"({va[0]} {ops1} {vb[0]}) {ops3} ({va[1]} {ops2} {vb[1]}) = {vc3}"
		if do_print:
			print(s)
	# endfor b
	return x

def graycodePosEnc(ntok, nbits, rand_phase=False):
	'''
	Generate a graycode
	seems more principled than standard SPE?
	'''
	pos_enc = np.zeros((ntok,nbits*2), dtype=np.float32)
	indx = np.linspace(0, (ntok-1)*2*math.pi, ntok)
	if rand_phase:
		phase_offset = np.random.uniform() * 2 * math.pi
	else:
		phase_offset = 0
	for i in range(nbits):
		# gray code: [0][1] has a period of 4
		# [2][3] has a period of sqrt(4*8) = sqrt(32) = 4 sqrt(2)
		# [4][5] period of 8..
		# period = 4 * (math.sqrt(2.0))**i
		#   above is slower - does not help?
		period = 4 * 2.0**i
		if True:
			# sinusoidal, seems to work better?
			pos_enc[:, 2*i  ] = -np.cos(indx / period + phase_offset)
			pos_enc[:, 2*i+1] = np.sin(indx / period + phase_offset)
		else:
			# graycode! (thresholded)
			pos_enc[:, 2*i  ] = np.cos(indx / period + phase_offset) < 0
			pos_enc[:, 2*i+1] = np.sin(indx / period + phase_offset) < 0
	return pos_enc

def genData3(bs, do_print=False):
	'''
	Task 3: from a list of 8 integers,
	compute the op of two of them based on *pointers*
	rather than positional arguments.
	This ought to be easy.
	'''
	ntok = 16
	nbits = 4
	md = 64 - (5 + (nbits*2)*3) # 35

	x = np.zeros((bs, ntok, md + 5 + (nbits*2)*3), dtype=np.float32)
	y = np.zeros_like(x)
	nd = ntok - 2
	for b in range(bs):
		pos_enc = graycodePosEnc(ntok, nbits, rand_phase=True)
		d = np.random.randint(1, 8, size=(nd,))
		ai = randint(nd)
		bi = randint(nd)
		op = randint(4)
		c,ops = clipOp(d[ai], d[bi], op, md)
		# encode the list of nd integers
		for k in range(nd):
			x[b,k,d[k]+5] = 1
		x[b,nd,op] = 1
		x[b,nd+1,4 ] = 1 # result
		# positional encoding
		x[b,:,md+5:md+5+8] = pos_enc
		x[b,nd,md+5+8:md+5+16] = pos_enc[ai,:]
		x[b,nd,md+5+16:md+5+24] = pos_enc[bi,:]
		x[b,nd+1,md+5+8:md+5+16] = pos_enc[8,:] # point to op
		# y[b,nd+1, 5+d[ai]] = 1 # TEST - works super fast
		# y[b,nd+1, 5+d[bi]] = 1
		# y[b,nd+1,md+5+8:md+5+16] = pos_enc[ai,:] # copy pointer
		# y[b,nd+1,md+5+16:md+5+24] = pos_enc[bi,:]
		y[b,nd+1, 5+c] = 1
		if do_print:
			for i in range(nd):
				print(d[i], end=' ')
			print(f"\nd[{ai}] {ops} d[{bi}] = {d[ai]} {ops} {d[bi]} = {c}")

	return x,y

def plotData3():
	x,y = genData3(800, do_print=False) # Test
	bs = 3
	x,y = genData3(bs, do_print=True)
	fig,axs = plt.subplots(bs,2)
	for b in range(bs):
		im = axs[b, 0].imshow(np.squeeze(x[b,:,:]))
		plt.colorbar(im, ax=axs[b, 0])
		im = axs[b, 1].imshow(np.squeeze(y[b,:,:]))
		plt.colorbar(im, ax=axs[b, 1])
		axs[b,0].set_title('X')
		axs[b,1].set_title('Y')
	plt.show()

OPERATORS = ['+', '-', '*', '/']

class Expression:
	def __init__(self, value=None, operator=None, left=None, right=None):
		self.value = value
		self.op = operator
		self.left = left
		self.right = right
		self.lparen_loc = 0
		self.loc = 0 # doubles for either value or op
		self.rparen_loc = 0

	def __str__(self):
		if self.op is None:
			return str(self.value)
		operator = OPERATORS[self.op]
		return f"({self.left} {operator} {self.right})"

	def setLocRec(self, loc):
		if self.op is None:
			self.loc = loc
			return loc + 1
		else:
			self.value = 0 # clear the value if it's an op
			self.lparen_loc = loc
			loc += 1
			loc = self.left.setLocRec(loc)
			self.loc = loc # operator
			loc += 1
			loc = self.right.setLocRec(loc)
			self.rparen_loc = loc
			loc += 1
			return loc

	def getLoc(self):
		return self.loc

	def printLoc(self):
		if self.value is not None:
			return str(self.loc)
		return f"({self.left.printLoc()} {self.loc} {self.right.printLoc()})"

	def printParentLoc(self, parent):
		if self.op is None:
			return str(parent)
		return f"({self.left.printParentLoc(self.loc)} {parent} {self.right.printParentLoc(self.loc)})"

	def encode(self, md, x, b, pos_enc):
		# need to just encode the left and right children
		c = self.loc
		if self.op is None:
			x[b,c,self.value+5] = 1
			x[b,c,md+5:md+5+8] = pos_enc[c] # abs loc
			return c+1
		else:
			lc = self.lparen_loc
			rc = self.rparen_loc
			x[b,lc,0] = -1 # "("
			x[b,lc,5] = 1 # paren is zero
			x[b,lc,md+5:md+5+8] = pos_enc[lc] # abs loc
			self.left.encode(md, x, b, pos_enc)
			x[b,c,self.op] = 1
			if self.value is not None:
				x[b,c,self.value+5] = 1
			else:
				x[b,c,5] = 1 # default to zero
			x[b,c,md+5:md+5+8] = pos_enc[c] # abs loc
			x[b,c,md+5+8:md+5+16] = pos_enc[self.left.getLoc()]
			x[b,c,md+5+16:md+5+24] = pos_enc[self.right.getLoc()]
			self.right.encode(md, x, b, pos_enc)
			x[b,rc,1] = -1 # ")"
			x[b,rc,5] = 1 # paren is zero
			x[b,rc,md+5:md+5+8] = pos_enc[rc] # abs loc
			x[b,rc,md+5+8:md+5+16] = 0 # no parent
			return rc+1

	def evaluate(self, md:int):
		# recusively evaluate the expression
		if self.op is None:
			return self.value
		c,_ = clipOp(self.left.evaluate(md), self.right.evaluate(md), self.op, md)
		self.value = c # save for supervised learning
		return c

class ExpressionGenerator:
	"""Recursively generates random arithmetic expression trees."""

	def __init__(self, max_terms, modulo):
		self.max_terms = max(2, max_terms) # Need at least 2 terms for an op
		self.modulo = modulo
		# these cataland numbers start at 2.
		self.catalan = [2, 5, 14, 42, 132, 429, 1430, 4862]
		self.catalan[0] = 2 + 30 # increase the frequency of the
		self.catalan[1] = 5 + 20 # simple expr
		self.catalan[2] = 14 + 20 # our models r kiddos
		self.catalan_cumsum = np.cumsum(self.catalan)

	def generate(self):
		r = random.randrange(0, self.catalan_cumsum[self.max_terms-2])
		terms = np.sum(self.catalan_cumsum < r) + 2 # offset
		# print("terms:", terms)
		if terms > self.max_terms:
			pdb.set_trace()
		return self._generate_recursive(terms)

	def _generate_recursive(self, terms_count):
		"""The core recursive generation logic."""
		# Base case: if only one term is left, it must be a number.
		if terms_count <= 1:
			return Expression(value=random.randrange(self.modulo))

		op = random.randrange(4)

		# Split the remaining terms between left and right children.
		left_terms = random.randint(1, terms_count - 1)
		right_terms = terms_count - left_terms

		left_child = self._generate_recursive(left_terms)
		right_child = self._generate_recursive(right_terms)

		# Prevent division by the literal number 0.
		if op == '/' and str(right_child) == '0':
			while str(right_child) == '0':
				right_child = self._generate_recursive(right_terms) # Reroll

		return Expression(operator=op, left=left_child, right=right_child)

class ExpressionGeneratorDepth:
	"""Recursively generates random arithmetic expression trees."""

	def __init__(self, max_depth, modulo):
		self.max_depth = max(1, max_depth) # Need at least 2 terms for an op
		self.modulo = modulo
		# these cataland numbers start at 2.
		self.n_depth = [2, 6, 16, 25, 36]
		self.n_depth_cumsum = np.cumsum(self.n_depth)

	def generate(self):
		# unlike the original code, which tries to make all
		# expressions equally probable (via catalan numbers),
		# just generate a random depth, weighted by n_depth
		r = random.randrange(0, self.n_depth_cumsum[self.max_depth-1])
		depth = np.sum(self.n_depth_cumsum < r) + 1
		# print("depth:", depth)
		if depth > self.max_depth:
			pdb.set_trace()
		return self._generate_recursive(depth)

	def _generate_recursive(self, depth):
		"""The core recursive generation logic."""
		# Base case: if only one term is left, it must be a number.
		if depth <= 0:
			return Expression(value=1+random.randrange(self.modulo-1))

		op = random.randrange(4)

		r = random.randrange(0, self.n_depth_cumsum[depth-1])
		left_depth = np.sum(self.n_depth_cumsum < r) # no offset
		left_child = self._generate_recursive(left_depth)

		r = random.randrange(0, self.n_depth_cumsum[depth-1])
		right_depth = np.sum(self.n_depth_cumsum < r) # no offset
		right_child = self._generate_recursive(right_depth)

		return Expression(operator=op, left=left_child, right=right_child)

def genData4(bs, do_print=False):
	'''
	Task 4: from random arithmetic expressions,
	generate parse trees & evaluate them
	'''
	ntok = 32
	nbits = 4 # hardcoded in class expression
	md = 64 - (5 + (nbits*2)*3) # 35

	rng = np.random.default_rng()
	x = np.zeros((bs, ntok, md + 5 + 8*3), dtype=np.float32)
	y = np.zeros_like(x)
	exp_gen = ExpressionGeneratorDepth(3, 7) # NOTE!!!
	for b in range(bs):
		pos_enc = graycodePosEnc(ntok, nbits, rand_phase=True)
		val = 0
		while val == 0:
			tree = exp_gen.generate()
			val = tree.evaluate(md)
		tree.setLocRec(0) # also resets eval.
		if do_print:
			print("expr:", tree)
			print("res loc :", tree.printLoc())
			print("ploc:", tree.printParentLoc(ntok-1))
		# pos_enc_permute = rng.permutation(pos_enc, axis=0)
		# pos_enc_permute = np.copy(pos_enc)
		tree.encode(md, x, b, pos_enc)
		# encode the result
		result = tree.evaluate(md) # sets internal values of the ops
		if do_print:
			print("res: ", result)
		n = tree.encode(md, y, b, pos_enc)
		x[b, -1, 4] = 1
		x[b, n:, md+5:md+5+8] = pos_enc[n:]
		x[b, :, md+5+8:] = 0 # mask pointer
		x[b, n:-1, 5] = 1 # default zero
		y[b, -1, 4] = 1
		y[b, -1, result+5] = 1
		y[b, n:, md+5:md+5+8] = pos_enc[n:]
		y[b, n:-1, 5] = 1 # default zero
		y[b, -1, md+5+8:md+5+16] = pos_enc[tree.getLoc()]

	return x,y

def plotData4():
	x,y = genData4(800, do_print=False) # Test
	bs = 4
	x,y = genData4(bs, do_print=True)
	fig,axs = plt.subplots(bs,2)
	for b in range(bs):
		im = axs[b, 0].imshow(np.squeeze(x[b,:,:]))
		plt.colorbar(im, ax=axs[b, 0])
		im = axs[b, 1].imshow(np.squeeze(y[b,:,:]))
		plt.colorbar(im, ax=axs[b, 1])
		axs[b,0].set_title('X')
		axs[b,1].set_title('Y')
	plt.show()

def genData5(bs,md, do_print):
	'''
	Can a hypergraph transformer add and remove tokens?
	'''
	ntok = 128
	nbits = 6
	pos_enc = graycodePosEnc(ntok, nbits)
	x = np.zeros((bs, ntok, md + nbits*2), dtype=np.float32)
	y = np.zeros_like(x)

	x[:, :, md:] = pos_enc # static --
	y[:, :, md:] = pos_enc # but /could/ be permuted?

	for b in range(bs):
		s = np.random.randint(0, high=md-1, size=(ntok,), dtype=int)
		# where there is a 0, remove the token.
		# for 1, add one token before.
		sp = np.ones((ntok,), dtype=int) * (md-1) #default new token fill
		# make sure it can copy arbitrary vector content.
		noiz = np.random.uniform(low = 0, high=1, size=(ntok,8))
		noizp = np.zeros_like(noiz)
		j = 0
		l = 0
		for k in range(ntok):
			if s[k] == 0:
				# skip this token, don't copy, don't increment.
				l += 1 # noop
			elif s[k] == 1:
				if j < ntok:
					sp[j] = md-1 #new token tag
					j += 1
				if j < ntok:
					sp[j] = s[k]
					noizp[j] = noiz[k]
					j += 1
			else:
				if j < ntok:
					sp[j] = s[k]
					noizp[j] = noiz[k]
					j += 1
		if do_print:
			print("starting sequence:")
			print(s)
			print("after insert/delete:")
			print(sp)
		indx = np.arange(ntok, dtype=int)
		x[b, indx, s[indx]] = 1
		y[b, indx, sp[indx]] = 1
		x[b, :, 8:16] += noiz
		y[b, :, 8:16] += noizp
	return x, y

def plotData5():
	x,y = genData5(2, 24-8, do_print=True)
	print(x.shape)
	fig,axs = plt.subplots(2,2)
	for i in range(2):
		axs[0,i].imshow(np.squeeze(x[i,...]).T)
		axs[1,i].imshow(np.squeeze(y[i,...]).T)
	plt.show()

def genData6(bs, do_print):
	'''
	train the network to accurately multiply two one-digit
	base 16 numbers.  Yes, the computer can do trillions of these things per sec .. this is super inefficient.  But.
	'''
	md = 8 + 16 + 8 # one-hot indicators, digits, pointer, [posenc]
	ntok = 8
	nbits = 4

	pos_enc = graycodePosEnc(ntok, nbits, rand_phase=False)
	x = np.zeros((bs, ntok, md + nbits*2), dtype=np.float32)
	y = np.zeros_like(x)
	x[:, :, -nbits*2:] = pos_enc
	y[:, :, -nbits*2:] = pos_enc # this will be overwritten

	for b in range(bs):
		va = randint(16)
		vb = randint(16)
		op = randint(2)
		if op == 0:
			vc = va + vb
		else:
			vc = va*vb
		vc0 = vc % 16
		vc1 = vc // 16
		def encode(tok, val):
			x[b, tok, 0] = 1.0 # occupied!
			x[b, tok, val] += 1.0
		encode(0, va + 8)
		encode(1, 1+op*2) # * : +-*/=?() -> 12345678
		encode(2, vb + 8)
		encode(3, 5) # =
		encode(4, 6) # ? (result)
		# last two entries are left empty / free.

		def encodeY(tok, val):
			y[b, tok, 0] = 1.0 # occupied!
			y[b, tok, val] += 1.0
		encodeY(4, vc0 + 8)
		if vc1 > 0:
			# encode a pointer
			y[b, 4, 24:32] = y[b, 5, 32:40]
			encodeY(5, vc1 + 8)
			# leave the pointer field in tok 5 empty = Null
		# else:
			# noop - leave the pointer field in tok 4 empty.
		# this means we need to measure loss:
		# cross-entropy over the digit (both 4 & 5)
		# MSE over pointer field (both)
		if do_print:
			print(f"{va} * {vb} = {vc} = 0x{vc1:x}{vc0:x}")
	# TODO TODO: we need to vary the allocation & make sure the pointer op still works.
	# which is OK, since we're only allocating one token.
	return x, y

def plotData6():
	bs = 2
	x, y = genData6(bs, True)
	fig,axs = plt.subplots(bs,2)
	for b in range(bs):
		axs[b, 0].imshow(np.squeeze(x[b,:,:]).T)
		axs[b, 1].imshow(np.squeeze(y[b,:,:]).T)
		axs[b,0].set_title('X')
		axs[b,1].set_title('Y')
	plt.show()

class Encoder:
	'''
	helper class for
	'''
	def __init__(self, ntok, nbits, do_print):
		self.tok_ctr = 0
		self.vert_ctr = 0
		self.horiz_ctr = np.zeros(16, dtype=int)
		self.pos_enc = graycodePosEnc(ntok, nbits, rand_phase=False)
		self.nbits = nbits
		self.do_print = do_print

	def encode(self, z, b, val, pos_space):
		if type(val) == str:
			dic = [' ','+','-','*','/','=','?','(',')','sl1','sl2']
			# technically it's shift /right/ since this is little-endian..
			v = dic.index(val)
		else:
			v = val + 16 # number of operations
		z[b, self.tok_ctr, 0] = 1.0 # occupied!
		z[b, self.tok_ctr, v] = 1.0
		pos = self.horiz_ctr[pos_space]
		# 2d addressing scheme!
		nb = self.nbits
		z[b, self.tok_ctr, -nb*2:] = self.pos_enc[pos, :]
		z[b, self.tok_ctr, -nb*4:-nb*2] = self.pos_enc[pos_space, :]
		self.tok_ctr += 1
		self.horiz_ctr[pos_space] += 1
		if self.do_print:
			if type(val) == str:
				print(val, end=' ')
			else:
				print(f'{val:x}', end=' ')

	def encodeList(self, z, b, val_list):
		if self.do_print:
			print(f"{self.vert_ctr}: ", end='')
		for val in val_list:
			self.encode(z, b, val, self.vert_ctr)
		if self.do_print:
			print(" ") # newline
		self.vert_ctr += 1

def genData7(bs, do_print=False):
	'''
	train a network on one step tasks:
	- multiply two single-digit hex numbers
	- add two 1-4 digit hex numbers (requires 4+ layers for carry)
	- shift left 1 and 2 places.
	'''
	nop = 16 # _+-*/=?() -> 012345678
	ntok = 28
	nbits = 8

	md = nop + 16 + (nbits*2)*2 # one-hot indicators, digits, 2d posenc
	x = np.zeros((bs, ntok, md), dtype=np.float32)
	y = np.zeros_like(x)

	def encHex(lst, val, ndigits=0):
		va0 = val & 0xf
		va1 = (val >> 4) & 0xf
		va2 = (val >> 8) & 0xf
		va3 = (val >> 12) & 0xf
		va4 = (val >> 16) & 0xf
		va5 = (val >> 20) & 0xf
		lst.append(va0)
		if val >= 16 or ndigits >= 2:
			lst.append(va1)
		if val >= 256 or ndigits >= 3:
			lst.append(va2)
		if val >= 4096 or ndigits >= 4:
			lst.append(va3)
		if val >= 65536 or ndigits >= 5:
			lst.append(va4)
		if val >= 65536*16 or ndigits >= 6:
			lst.append(va5)

	for b in range(bs):
		# new encoder per batch for a random phase.
		enc = Encoder(ntok, nbits, do_print)
		x[b, :, -nbits*2:] = enc.pos_enc # "horizontal"
		x[b, :, -nbits*4:-nbits*2] = enc.pos_enc[0,:] # "vertical"
		# e.g. everything starts off as flat..
		task = b % 4
		task = 2
		if task == 0: # add, very easy for both
			va = randint(16)
			vb = randint(16)
			vc = va*vb
			vc0 = vc & 0xf
			vc1 = vc >> 4
			enc.encodeList(x, b, [va, '*', vb])
			enc.encodeList(y, b, [vc0, vc1])

		if task == 1: # multiply, very easy for both
			va = randint(16)
			vb = randint(16)
			vc = va*vb
			vc0 = vc & 0xf
			vc1 = vc >> 4
			enc.encodeList(x, b, [va, '*', vb])
			enc.encodeList(y, b, [vc0, vc1])

		if task == 2 and False:
			na = randint(4)+1
			nb = randint(4)+1
			# na = 4
			# nb = 4
			va = randint(16**na)
			vb = randint(16**nb)
			# encode the problem
			lst = []
			encHex(lst, va, ndigits=4) # ndigits makes it fixed
			lst.append('+')
			encHex(lst, vb, ndigits=4)
			enc.encodeList(x, b, lst)
			# and the solution
			vc = va + vb
			lst = []
			encHex(lst, vc)
			enc.encodeList(y, b, lst)

		# do task 2 in a different way, cascade fashion: spell out the carries
		if task == 2:
			na = randint(4)+1
			nb = randint(4)+1
			# na = 4
			# nb = 4
			va = randint(16**na)
			vb = randint(16**nb)
			# encode the problem
			lst = []
			encHex(lst, va)
			lst.append('+')
			encHex(lst, vb)
			enc.encodeList(x, b, lst)
			steps = max(na,nb) + 1
			step = randint(steps)
			# and the solution -- spell it out w/ carries
			for i in range(steps-1):
				vc = ((va >> (4*i)) & 0xf) + ((vb >> (4*i)) & 0xf)
				lst = []
				encHex(lst, vc << (4*i))
				if i == step:
					enc.encodeList(y, b, lst)
				elif i < step:
					enc.encodeList(x, b, lst)
			if step == max(na,nb):
				vc = va + vb
				lst = []
				encHex(lst, vc)
				enc.encodeList(y, b, lst)

		if task == 3: # also perfectly easy
			na = randint(3)+1
			shift = randint(2)+1
			va = randint(16**na)
			lst = []
			encHex(lst, va)
			if shift == 1:
				lst.append('sl1')
				vc = va*16
			if shift == 2:
				lst.append('sl2')
				vc = va*256
			enc.encodeList(x, b, lst)
			lst = []
			encHex(lst, vc)
			enc.encodeList(y, b, lst)

	return x, y

def plotData7():
	bs = 3
	x, y = genData7(bs, True)
	fig,axs = plt.subplots(bs,2)
	for b in range(bs):
		axs[b, 0].imshow(np.squeeze(x[b,:,:]))
		axs[b, 1].imshow(np.squeeze(y[b,:,:]))
		axs[b,0].set_title('X')
		axs[b,1].set_title('Y')
	plt.show()


def genData8(bs, do_print=False):
	'''
	Multiply two 2-digit hex numbers
	Use local allocation for intermediate variables
	'''
	nop = 16 # _+-*/=?() -> 012345678
	md = nop + 16 + 32 # one-hot indicators, digits, 2D posenc
	ntok = 80
	nbits = 8

	pos_enc = graycodePosEnc(ntok, nbits, rand_phase=True) # NOTE check random phase
	x = np.zeros((bs, ntok, md), dtype=np.float32)
	y = np.zeros_like(x)
	x[:, :, -nbits*2:] = pos_enc # "horizontal"
	x[:, :, -nbits*4:-nbits*2] = pos_enc[0,:] # "vertical"

	# TODO need a better way of addressing the 'local' tokens!!!

	for b in range(bs):
		# step = randint(10)+1 # what step are we supervising?
		step = 10 # FIXME
		va = randint(256)
		vb = randint(256)
		vc = va*vb
		# convert these all to hex.
		va0 = va & 0xf
		va1 = (va >> 4) & 0xf
		vb0 = vb & 0xf
		vb1 = (vb >> 4) & 0xf

		# first encode the problem.
		encodeList(x, ['(',va0,va1,'*',vb0,vb1,')','=','?'])
		# expand the arguments to one-digit operations.
		# could also pass pointers - no difference with one digit?
		# but pointers might allow the model to understand & change structure better?
		if step >= 1:
			# just a copy & reformat op
			encodeList(y, \
				['(',va0,'*',vb0,')','+','(',va0,'*',vb1,')','sl1'])
		if step >= 2:
			# reduce and sum it
			vc = va0 * vb0
			vc0 = vc & 0xf
			vc1 = (vc >> 4) & 0xf
			vd = va0 * vb1
			vd0 = vd & 0xf
			vd1 = (vd >> 4) & 0xf
			encodeList(y, \
				[vc0,vc1,'+',vd0,vd1,'sl1'])
		if step >= 3:
			encodeList(y, [vc0,vc1,'+',0,vd0,vd1])
		if step >= 4:
			# add with carry
			vc1 += vd0
			vc2 = vd1 + (vc1 >> 4)
			vc1 = vc1 & 0xf
			encodeList(y, [vc0, vc1, vc2])
		if step >= 5:
			# copy & reformat
			encodeList(y, \
				['(',va1,'*',vb0,')','sl1','+','(',va1,'*',vb1,')','sl2'])
		if step >= 6:
			# reduce and sum
			ve = va1 * vb0
			ve0 = ve & 0xf
			ve1 = (ve >> 4) & 0xf
			vf = va1 * vb1
			vf0 = vf & 0xf
			vf1 = (vf >> 4) & 0xf
			encodeList(y, \
				[ve0,ve1,'sl1','+',vf0,vf1,'sl2'])
		if step >= 7:
			encodeList(y, \
				[0,ve0,ve1,'+',0,0,vf0,vf1])
		if step >= 8:
			# add with carry
			ve1 += vf0
			ve2 = vf1 + (ve1 >> 4)
			encodeList(y, [0, ve0, ve1, ve2])
		if step >= 9:
			# copy & reformat
			encodeList(y, [vc0,vc1,vc2,'+',0,ve0,ve1,ve2])
		if step >= 10:
			# the answer!
			vc1 += ve0
			vc2 += ve1 + (vc1 >> 4)
			vc3 = ve2 + (vc2 >> 4)
			vc1 = vc1 & 0xf
			vc2 = vc2 & 0xf
			encodeList(y, [vc0,vc1,vc2,vc3])
		if do_print:
			print("num_tok", tok_ctr)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='generate compositional data for training graph/hypergraph transformers')
	parser.add_argument('-t', type=int, default=6, help='which test to run')
	args = parser.parse_args()
	if args.t == 3:
		plotData3()
	if args.t == 4:
		plotData4()
	if args.t == 6:
		plotData6()
	if args.t == 7:
		plotData7()
