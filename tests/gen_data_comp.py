import numpy as np
import matplotlib.pyplot as plt

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
		c = modOp(d[ai], d[bi], op, md)
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


if __name__ == '__main__':
	genData1(15, 19, True)
	genData2(15, 19, True)

	x = genData3(4, 19)
	print(x.shape)
	fig,axs = plt.subplots(2,2)
	for i in range(4):
		j = i // 2
		k = i % 2
		axs[j,k].imshow(np.squeeze(x[i,...]))
	plt.show()
