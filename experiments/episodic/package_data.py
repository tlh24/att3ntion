import os, glob, re
import pandas as pd

def package_logs():
	files = glob.glob("data/losslog*.txt")
	if not files:
		print("No files found!")
		return

	file_info = []
	raw_names = set()

	# Step 1: Parse filenames to get the "raw" base name and replicate number
	for fname in files:
		base = os.path.basename(fname)

		# Extract replicate number (e.g., _r2 -> 2)
		r_match = re.search(r'_r(\d+)\.txt$', base)
		repl = int(r_match.group(1)) if r_match else 1

		# Strip the replicate and extension to get the raw experiment name
		raw = re.sub(r'_r\d+\.txt$', '', base).replace('.txt', '')

		file_info.append({'filename': fname, 'raw_name': raw, 'replicate': repl})
		raw_names.add(raw)

	# Step 2: Tokenize the raw names to find common prefixes/suffixes by words (not characters)
	token_lists = [name.split('_') for name in raw_names]

	prefix_tokens = []
	suffix_tokens = []

	if len(token_lists) > 1:
		min_len = min(len(t) for t in token_lists)

		# Find common prefix tokens
		for i in range(min_len):
			if all(t[i] == token_lists[0][i] for t in token_lists):
				prefix_tokens.append(token_lists[0][i])
			else:
				break

		# Find common suffix tokens (making sure we don't overlap with the prefix)
		for i in range(1, min_len - len(prefix_tokens) + 1):
			if all(t[-i] == token_lists[0][-i] for t in token_lists):
				suffix_tokens.insert(0, token_lists[0][-i])
			else:
				break

	# Step 3: Map the raw names to their new "delta" labels
	label_map = {}
	for raw in raw_names:
		tokens = raw.split('_')

		# Slice out the prefix and suffix
		start_idx = len(prefix_tokens)
		end_idx = len(tokens) - len(suffix_tokens)
		delta_tokens = tokens[start_idx:end_idx]

		# If everything cancels out (e.g., files were identical), fallback to raw name
		label_map[raw] = "_".join(delta_tokens) if delta_tokens else raw

	# Step 4: Read files and combine into one DataFrame
	all_dfs = []
	for info in file_info:
		fname = info['filename']
		try:
			df = pd.read_csv(fname, sep='\t', header=None, on_bad_lines='skip',
							 names=['iter', 'loss', 'top1', 'val_loss', 'val_top1'])

			# Apply the clean delta label
			df['label'] = label_map[info['raw_name']]
			df['replicate'] = info['replicate']
			all_dfs.append(df)

			print(f"Processed: {fname} -> Label: {df['label'].iloc[0]}")
		except Exception as e:
			print(f"Skipping {fname}: {e}")

	# Step 5: Save
	if all_dfs:
		combined_df = pd.concat(all_dfs, ignore_index=True)
		combined_df.to_csv("combined_logs.csv", index=False)
		print(f"\nSuccess! Stripped prefix: {'_'.join(prefix_tokens) + '_' if prefix_tokens else '(none)'}")
		print(f"Stripped suffix: {'_' + '_'.join(suffix_tokens) if suffix_tokens else '(none)'}")
		print(f"Saved {len(all_dfs)} files to combined_logs.csv")

if __name__ == "__main__":
	package_logs()
