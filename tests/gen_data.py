import numpy as np

def genData(bs, md, do_print=False): 
	'''
	Problem is a 3-term analogy: 
	if A `op B = C then D `op E = `F
	tokens indicated by ` need to be filled in.
	Values are integers, ops are the usual arithmetic operations. 
	values and ops are one-hot encoded in a 32-d vector.
	All operations are over the finite field of integers: mod 'md'
	'''
	def randint(k): 
		return np.random.randint(k)
		
	assert(md < 32-4)
	x = np.zeros((bs, 8, 32), dtype=int)

	n_ambig = 0
	n_sampled = 0
	expression_set = set()
	for b in range(bs): 
		unambig = False
		while not unambig: 
			n_sampled += 1
			va = randint(md)
			vb = randint(md-1) + 1 # avoid divide by zero and noops
			op = np.random.randint(4)
			vc0 = (va + vb) % md
			vc1 = (va - vb) % md
			vc2 = (va * vb) % md
			vc3 = (va // vb) % md
			# another way to do this is to sort the 4, calculate the differences, and compare that to zero - this requires 3 comparisons but potentially 4 swaps, so better to just do the 6 comparisons.
			if vc0 != vc1 and vc2 != vc3 and vc0 != vc2 and vc1 != vc3 and vc0 != vc3 and vc1 != vc2: 
				unambig = True
			else:
				n_ambig += 1
		# in contrast, D op E is always deterministic so don't need to check if the inference is unambiguous.  
		vd = randint(md)
		ve = randint(md-1) + 1
		match op: 
			case 0: 
				vc = vc0
				vf = (vd + ve) % md
			case 1:
				vc = vc1
				vf = (vd - ve) % md
			case 2: 
				vc = vc2
				vf = (vd * ve) % md
			case 3: 
				vc = vc3
				vf = (vd // ve) % md
		# encode. 
		x[b,0,va+4] = 1
		x[b,1,op  ] = 1 # must be masked
		x[b,2,vb+4] = 1
		x[b,3,vc+4] = 1
		
		x[b,4,vd+4] = 1
		x[b,5,op  ] = 1 # must be masked
		x[b,6,ve+4] = 1
		x[b,7,vf+4] = 1 # must be masked
		
		match op:
			case 0:
				ops = '+'
			case 1:
				ops = '-'
			case 2:
				ops = '*'
			case 3:
				ops = '/'
		s = f"if {va} op {vb} = {vc} then {vd} op {ve} = f  (op = {ops}, f = {vf})"
		expression_set.add(s)
		if do_print:
			print(s)
	# endfor b
	print(f"ambiguous: {n_ambig} / {n_sampled}, {100*n_ambig/n_sampled}% batch_size:{bs} unique:{len(expression_set)}")
	return x


if __name__ == '__main__':
	genData(1500000, 19, False)
	# unique seems to asymptope at 1512 with m = 7
	# ~319k m = 19
