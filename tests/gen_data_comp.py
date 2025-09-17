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

	def count(self):
		if self.op is None:
			return 1 # ourself
		else:
			# 3 is for (,op,)
			return 3 + self.left.count() + self.right.count()

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
				x[b,c,self.value+5] = 1 # if op is evaluated
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
		# these 'catalan' numbers start at 2.
		self.n_depth = [2, 10, 25, 49, 81]
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
	exp_gen = ExpressionGeneratorDepth(4, 7) # NOTE!!!
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

def genData5(bs,md, do_print):
	'''
	Can a hypergraph transformer add and remove tokens?
	'''
	ntok = 32
	pos_enc = np.zeros((ntok,8), dtype=np.float32)
	indx = np.linspace(0, (ntok-1)*2*math.pi, ntok)
	for i in range(4):
		# gray code: [0][1] has a period of 4
		# [2][3] has a period of sqrt(4*8) = sqrt(32) = 4 sqrt(2)
		# [4][5] period of 8..
		# period = 4 * (math.sqrt(2.0))**i
		period = 4 * 2.0**i
		phase = math.pi / period # indx is scaled by 2 pi
		pos_enc[:, 2*i  ] = -np.cos(indx / period + phase)
		pos_enc[:, 2*i+1] = np.sin(indx / period + phase)
	x = np.zeros((bs, ntok, md + 8), dtype=np.float32)
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

	# x = genData4(800, 11, do_print=False) # Test
	# x = genData4(8, 11, do_print=True)
	# print(x.shape)
	# fig,axs = plt.subplots(4,2)
	# for i in range(8):
	# 	j = i // 2
	# 	k = i % 2
	# 	axs[j,k].imshow(np.squeeze(x[i,...]))
	# plt.show()

	x,y = genData5(2, 24-8, do_print=True)
	print(x.shape)
	fig,axs = plt.subplots(2,2)
	for i in range(2):
		axs[0,i].imshow(np.squeeze(x[i,...]))
		axs[1,i].imshow(np.squeeze(y[i,...]))
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
