import torch

if __name__ == '__main__':
	# let's estimate the peak flops here
	if torch.cuda.is_available():
		device = torch.device('cuda')
		torch.backends.cuda.matmul.allow_tf32 = True
	start_event = torch.cuda.Event(enable_timing=True)
	end_event = torch.cuda.Event(enable_timing=True)
	dtype = torch.bfloat16
	siz = 8192
	iters = int((8192 // siz)**3 * 5)
	a = torch.randn(siz, siz, dtype=dtype, device=device)
	b = torch.randn(siz, siz, dtype=dtype, device=device)
	c = torch.randn(siz, siz, dtype=dtype, device=device)
	d = torch.zeros(siz, siz, dtype=torch.float32, device=device)

	start_event.record()
	for i in range(iters):
		d = (a @ b)
		d += (b @ c)
	end_event.record()
	torch.cuda.synchronize()

	time = start_event.elapsed_time(end_event)
	flops = iters * siz**3 * 2.0
	print(f"matmul throughput: {(flops / 1e12) / (time / 1000)} TFlops")
