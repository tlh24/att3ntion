import numpy as np

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


if __name__ == '__main__':
	genData1(15, 19, True)
	genData2(15, 19, True)
