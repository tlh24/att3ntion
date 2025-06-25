import numpy as np
import matplotlib.pyplot as plt
import random
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
		# need to just encode the left and right childeren
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
	exp_gen = ExpressionGenerator(2, md) # NOTE!!!
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

def genData5(bs,md):
	'''
	Can a hypergraph transformer add and remove tokens?
	'''
	ntok = 16
	pos_enc = np.zeros((ntok,8), dtype=np.float32)
	indx = np.linspace(0, 2*3.1415926, ntok)
	rng = np.random.default_rng()
	x = np.zeros((bs, ntok, md + 5 + 8*3), dtype=np.float32)

	for b in range(bs):


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

	x = genData4(800, 11, do_print=False) # Test
	x = genData4(8, 11, do_print=True)
	print(x.shape)
	fig,axs = plt.subplots(4,2)
	for i in range(8):
		j = i // 2
		k = i % 2
		axs[j,k].imshow(np.squeeze(x[i,...]))
	plt.show()
