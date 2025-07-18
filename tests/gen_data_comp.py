import numpy as np
import matplotlib.pyplot as plt
import random
import math
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

def genData3(bs, md):
	'''
	Task 3: from a list of 8 integers,
	compute the op of two of them based on *pointers*
	rather than positional arguments.
	This ought to be easy.
	'''
	pos_enc = np.zeros((10,8), dtype=np.float32)
	indx = np.linspace(0, 2*3.1415926, 10)
	for i in range(4):
		freq = 2**(i/3)
		pos_enc[:, 2*i  ] = np.sin(indx * freq)
		pos_enc[:, 2*i+1] = np.cos(indx * freq)

	x = np.zeros((bs, 10, md + 5 + 8*3), dtype=np.float32)
	for b in range(bs):
		d = np.random.randint(1, md, size=(8,))
		ai = randint(8)
		bi = randint(8)
		op = randint(4)
		c,_ = modOp(d[ai], d[bi], op, md)
		for k in range(8):
			x[b,k,d[k]+5] = 1
		x[b,8,op] = 1
		x[b,9,4 ] = 1 # result
		# positional encoding
		x[b,:,md+5:md+5+8] = pos_enc
		x[b,8,md+5+8:md+5+16] = pos_enc[ai,:]
		x[b,8,md+5+16:md+5+24] = pos_enc[bi,:]
		x[b,9,md+5+8:md+5+16] = pos_enc[8,:] # point to op

	return x

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
		if self.value is not None:
			return str(self.value)
		operator = OPERATORS[self.op]
		return f"({self.left} {operator} {self.right})"

	def setLocRec(self, loc):
		if self.value is not None:
			self.loc = loc
			return loc + 1
		else:
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
		if self.value is not None:
			return str(parent)
		return f"({self.left.printParentLoc(self.loc)} {parent} {self.right.printParentLoc(self.loc)})"

	def encode(self, md, x, b, pos_enc):
		# need to just encode the left and right children
		c = self.loc
		if self.value is not None:
			x[b,c,self.value+5] = 1
			x[b,c,md+5:md+5+8] = pos_enc[c] # abs loc
		else:
			lc = self.lparen_loc
			rc = self.rparen_loc
			x[b,lc,0] = -1 # "("
			x[b,lc,md+5:md+5+8] = pos_enc[lc] # abs loc
			self.left.encode(md, x, b, pos_enc)
			x[b,c,self.op] = 1
			x[b,c,md+5:md+5+8] = pos_enc[c] # abs loc
			x[b,c,md+5+8:md+5+16] = pos_enc[self.left.getLoc()]
			x[b,c,md+5+16:md+5+24] = pos_enc[self.right.getLoc()]
			self.right.encode(md, x, b, pos_enc)
			x[b,rc,1] = -1 # ")"
			x[b,rc,md+5:md+5+8] = pos_enc[rc] # abs loc
			x[b,rc,md+5+8:md+5+16] = 0 # no parent

	def evaluate(self, md:int):
		# recusively evaluate the expression
		if self.value is not None:
			return self.value % md
		c,_ = modOp(self.left.evaluate(md), self.right.evaluate(md), self.op, md)
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

def genData4(bs, md, do_print=False):
	'''
	Task 4: from random arithmetic expressions,
	generate parse trees
	'''
	ntok = 16
	pos_enc = np.zeros((ntok,8), dtype=np.float32)
	indx = np.linspace(0, 2*3.1415926, ntok)

	rng = np.random.default_rng()
	x = np.zeros((bs, ntok, md + 5 + 8*3), dtype=np.float32)
	exp_gen = ExpressionGenerator(4, md) # NOTE!!!
	for b in range(bs):
		tree = exp_gen.generate()
		tree.setLocRec(0)
		if do_print:
			print("expr:", tree)
			print("loc :", tree.printLoc())
			print("ploc:", tree.printParentLoc(ntok-1))
		# pos_enc_permute = rng.permutation(pos_enc, axis=0)
		# pos_enc_permute = np.copy(pos_enc)
		for i in range(4):
			freq = 2**(i/3)
			rand_phase = np.random.uniform(0, 3.1415926*2)
			pos_enc[:, 2*i  ] = np.sin(indx * freq + rand_phase)
			pos_enc[:, 2*i+1] = np.cos(indx * freq + rand_phase)
		tree.encode(md, x, b, pos_enc)
		# encode the result
		result = tree.evaluate(md)
		if do_print:
			print("res: ", result)
		x[b, -1, 4] = 1
		x[b, -1, result+5] = 1
		x[b, -1, md+5:md+5+8] = pos_enc[-1]
		x[b, -1, md+5+8:md+5+16] = pos_enc[tree.getLoc()]

	return x

def plotData4():
	x = genData4(800, 11, do_print=False) # Test
	x = genData4(8, 11, do_print=True)
	print(x.shape)
	fig,axs = plt.subplots(4,2)
	for i in range(8):
		j = i // 2
		k = i % 2
		axs[j,k].imshow(np.squeeze(x[i,...]))
	plt.show()

def graycodePosEnc(ntok, nbits):
	'''
	Generate a graycode
	seems more principled than standard SPE?
	'''
	pos_enc = np.zeros((ntok,nbits*2), dtype=np.float32)
	indx = np.linspace(0, (ntok-1)*2*math.pi, ntok)
	for i in range(nbits):
		# gray code: [0][1] has a period of 4
		# [2][3] has a period of sqrt(4*8) = sqrt(32) = 4 sqrt(2)
		# [4][5] period of 8..
		# period = 4 * (math.sqrt(2.0))**i
		# above is slower - does not help?
		period = 4 * 2.0**i
		phase = math.pi / period # indx is scaled by 2 pi
		if True:
			# sinusoidal, seems to work better?
			pos_enc[:, 2*i  ] = -np.cos(indx / period + phase)
			pos_enc[:, 2*i+1] = np.sin(indx / period + phase)
		else:
			# graycode! (thresholded)
			pos_enc[:, 2*i  ] = np.cos(indx / period + phase) < 0
			pos_enc[:, 2*i+1] = np.sin(indx / period + phase) < 0
	return pos_enc

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

	pos_enc = graycodePosEnc(ntok, nbits)
	x = np.zeros((bs, ntok, md + nbits*2), dtype=np.float32)
	y = np.zeros_like(x)
	x[:, :, -nbits*2:] = pos_enc
	y[:, :, -nbits*2:] = pos_enc # this will be overwritten

	for b in range(bs):
		va = randint(16)
		vb = randint(16)
		vc = va*vb
		vc0 = vc % 16
		vc1 = vc // 16
		def encode(tok, val):
			x[b, tok, 0] = 1.0 # occupied!
			x[b, tok, val] += 1.0
		encode(0, va + 8)
		encode(1, 3) # * : +-*/=? -> 123456
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

if __name__ == '__main__':
	# genData1(15, 19, True)
	# genData2(15, 19, True)

	# x = genData3(4, 19)
	# print(x.shape)
	# fig,axs = plt.subplots(2,2)
	# for i in range(4):
	# 	j = i // 2
	# 	k = i % 2
	# 	axs[j,k].imshow(np.squeeze(x[i,...]))
	# plt.show()

	plotData6()
