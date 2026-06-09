import itertools
from functools import lru_cache
import random
import numpy as np
import matplotlib.pyplot as plt

OPS = ['+', '-', '*', '/']
# OPS = ['+']
COMMUTATIVE_OPS = {'+', '*'}

@lru_cache(maxsize=None)
def generate_ast(d, v):
	"""Generates optimal DAG of abstract syntax trees."""
	if d == 0:
		return [f"x_{i+1}" for i in range(v)] + ["C"]
	trees = []
	for op in OPS:
		is_comm = op in COMMUTATIVE_OPS
		for dl, dr in itertools.product(range(d), repeat=2):
			if max(dl, dr) != d - 1 or (is_comm and dl > dr):
				continue

			lefts, rights = generate_ast(dl, v), generate_ast(dr, v)
			pairs = itertools.combinations_with_replacement(lefts, 2) if (is_comm and dl == dr) \
					else itertools.product(lefts, rights)

			for L, R in pairs:
				if L == 'C' and R == 'C': continue # Fold constants
				if op in ('-', '/') and L == R: continue # Fold identities
				trees.append((op, L, R))

	return trees

def compile_ast(ast):
	"""
	Traverses the AST once. Returns:
	1. The readable string expression.
	2. The number of constants required.
	3. The number of variables used.
	4. A fast compiled lambda function: func(X, C, P)
	"""
	c_idx = 0
	used_vars = set() # Track unique variables

	def build(node, is_root=True):
		nonlocal c_idx
		if isinstance(node, str):
			if node.startswith('x_'):
				used_vars.add(node) # Log variable usage!
				return node, f"X[{int(node[2:]) - 1}]"
			idx, c_idx = c_idx, c_idx + 1
			return f"const_{idx+1}", f"C[{idx}]"

		op, L, R = node
		l_str, l_code = build(L, False)
		r_str, r_code = build(R, False)

		expr_str = f"{l_str} {r_str} {op}" if is_root else f"({l_str} {r_str} {op})" # use reverse polish notation

		if op == '+': code = f"({l_code} + {r_code}) % P"
		if op == '-': code = f"({l_code} - {r_code}) % P"
		if op == '*': code = f"({l_code} * {r_code}) % P"
		if op == '/': code = f"({l_code} * pow({r_code}, -1, P)) % P"

		return expr_str, code

	expr_str, code_str = build(ast)
	func = eval(f"lambda X, C, P: {code_str}")

	return expr_str, c_idx, len(used_vars), func

def evaluate_autoregressive(func, init_X, C_vals, L=10, P=113):
	"""
	Evaluates the compiled function autoregressively.
	init_X: e.g., [1, 2] -> x_2(t-2)=1, x_1(t-1)=2.
	"""
	# X represents the sliding window: [x_1, x_2, ...] -> [2, 1]
	X = list(reversed(init_X))
	seq = list(init_X)

	for _ in range(L):
		try:
			next_val = func(X, C_vals, P)
		except ValueError:
			return None # pow(0, -1, P) throws ValueError (Division by Zero)

		seq.append(next_val)
		# Shift the autoregressive window efficiently
		X = [next_val] + X[:-1]

	return seq

def gen_data(mode, max_d, V, P=113, L=1, data_size=1000, exact_v=True):
	"""
	mode 'grok': Formulas shared. Inputs (init_conditions & constants) split 60/40. (Grokking)
	mode 'formulas': Formulas split 60/40. Inputs sampled uniformly. (Formula generalization)
	max_d : maximum formula depth
	V :  number of variables
	L : length of autoregressive roll-out
	"""
	# 1. Compile all valid formulas from ASTs
	formulas = []
	for d in range(max_d + 1):
		for ast in generate_ast(d, V):
			expr_str, num_c, num_v, func = compile_ast(ast)
			# Strict enforcement of exactly V variables
			if exact_v and num_v != V:
				continue
			formulas.append((expr_str, num_c, func))

	# 2. Assign formulas to Train/Test (Mode B only)
	if mode != 'grok':
		random.shuffle(formulas)
		train_formulas = set(f[0] for f in formulas[:int(len(formulas) * 0.6)])

	train_data, test_data = [], []
	seen = set()
	patience = 0

	while len(train_data) + len(test_data) < data_size and patience < 5000:
		# Sample uniformly
		expr_str, num_c, func = random.choice(formulas)
		inputs = tuple(random.randrange(P) for _ in range(V + num_c))

		# Deduplication
		key = (expr_str, inputs)
		if key in seen:
			patience += 1
			continue

		seen.add(key)
		patience = 0

		# Evaluate
		init_X, C_vals = inputs[:V], inputs[V:]
		seq = evaluate_autoregressive(func, init_X, C_vals, L, P)

		if seq is None:
			continue # discard formulas that hit Division by Zero

		# Replace constants in the string safely (reverse order avoids const_10 -> const_1 bug)
		final_str = expr_str
		for i in reversed(range(num_c)):
			final_str = final_str.replace(f"const_{i+1}", str(C_vals[i]))

		record = (final_str, init_X, seq)

		# 3. Route to Train or Test based on the Mode rules
		if mode == 'grok':
			# Hash the inputs to get a stable 60/40 split across all formulas
			is_train = hash(inputs) % 100 < 60
		else:
			# Check which formula split this belongs to
			is_train = expr_str in train_formulas

		if is_train:
			train_data.append(record)
		else:
			test_data.append(record)

	return train_data, test_data

def to_numpy(train_data, test_data, P):
	"""Maps expressions and sequences to an int32 numpy array, padding the expression."""
	OP_MAP = {'+': 0, '-': 1, '*': 2, '/': 3, '(': 4, ')': 5}

	def tokenize(expr):
		# Safely separate parens so .split() works cleanly
		for r in '()': expr = expr.replace(r, f' {r} ')
		return [
			P + OP_MAP[t] if t in OP_MAP
			else P + len(OP_MAP) + int(t[2:]) if t.startswith('x_')
			else int(t)
			for t in expr.split()
		]
	# Pre-parse to compute the max expression length
	tr_parsed = [(tokenize(e), seq) for e, _, seq in train_data]
	te_parsed = [(tokenize(e), seq) for e, _, seq in test_data]

	max_e = max((len(e) for e, _ in tr_parsed + te_parsed), default=0)
	seq_l = len(tr_parsed[0][1]) # train and test must be the same len
	total_len = max_e + seq_l

	def build_array(parsed_data):
		# Pre-allocate contiguous array filled with -1
		arr = np.full((len(parsed_data), total_len), -1, dtype=np.int32)
		for i, (e_toks, seq) in enumerate(parsed_data):
			arr[i, :len(e_toks)] = e_toks  # Drop expression at the start
			arr[i, max_e:] = seq           # Drop sequence at the exact end
		return arr
	# return the maximum expression length & np train test arrays.
	return max_e, build_array(tr_parsed), build_array(te_parsed)

if __name__ == "__main__":
	V = 2 # Variables
	D = 2 # Depth

	init_conditions = [1, 2] # x_2(t-2) = 1, x_1(t-1) = 2
	c_vals = [3, 5, 7]       # Plentiful constants for 'C' mapping

	count = 0
	for d in range(D + 1):
		for ast in generate_ast(d, V):
			expr_str, num_c, num_vars, func = compile_ast(ast)

			# Slice only the constants this specific formula requires
			formula_constants = c_vals[:num_c]

			# Run the autoregression!
			seq = evaluate_autoregressive(func, init_conditions, formula_constants, L=5)

			# Only print formulas that survived (didn't divide by zero)
			if seq is not None:
				print(f"{expr_str:<30} | Consts: {formula_constants} | Seq: {seq}")
				count += 1

			if count >= 10: break
		if count >= 10: break

	print("Generating Mode A (Grokking Split)...")
	train_A, test_A = gen_data('grok', max_d=2, V=2, L=1, data_size=50)
	for row in train_A: print(f"Train: {row}")
	for row in test_A:  print(f"Test:  {row}")

	# check numpy conversion
	max_exprlen, train_np, test_np = to_numpy(train_A, test_A, 25)
	fig,axs = plt.subplots(2, 1, figsize=(12, 6))
	axs[0].imshow(train_np.T - 113)
	axs[0].set_title('Train')
	axs[1].imshow(test_np.T - 113)
	axs[1].set_title('Test')
	plt.show()

	print("\nGenerating Mode B (Formula Split)...")
	train_B, test_B = gen_data('B', max_d=1, V=2, L=2, data_size=5)
	for row in train_B: print(f"Train: {row}")
	for row in test_B:  print(f"Test:  {row}")

