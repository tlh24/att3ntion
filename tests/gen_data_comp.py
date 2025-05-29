import numpy as np

# task 1: learn the rules of modulo arithmetic.

def genData(bs, md, do_print=False):
	'''
	Task 1: learn the rules of modulo arithmetic.
	'''

	def randint(k):
		return np.random.randint(k)

	assert(md < 32-4)
	x = np.zeros((bs, 4, 32), dtype=int)

	for b in range(bs):
		va = randint(md)
		vb = randint(md-1) + 1 # avoid divide by zero and noops
		op = np.random.randint(4)
		vc0 = (va + vb) % md
		vc1 = (va - vb) % md
		vc2 = (va * vb) % md
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


if __name__ == '__main__':
	genData(15, 19, True)
