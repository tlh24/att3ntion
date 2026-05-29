# att3ntion

**Hypergraph attention: 3-way attention between token triplets, with O(N) memory scaling.**

## Quick Start
Clone this repository, then: 

```bash
# determine CUDA toolkit version:
nvcc --version | grep -oP 'release \K[\d.]+'

# Install CUDA-enabled Torch (either cu126 or cu130 depending on nvcc output)
pip install torch --extra-index-url https://download.pytorch.org/whl/cu130
pip install torch --extra-index-url https://download.pytorch.org/whl/cu126

# If something goes wrong, you can do:
pip install --force-reinstall torch --extra-index-url https://download.pytorch.org/whl/cu130
pip install --force-reinstall torch --extra-index-url https://download.pytorch.org/whl/cu126

# Install
cd att3ntion
pip install -r requirements.txt

# This will launch the CUDA compilation
# The --no-build-isolation flag is mandatory, and considered best practice
pip install -e . --no-build-isolation

# Run demo
python demo.py              # Basic usage
python demo.py --memory     # Memory scaling benchmark
python demo.py --train      # Train on arithmetic task
python demo.py --all        # Run everything
```

## UV based Quick Start

```bash
uv --version # must be >= 0.11

# to get or update uv, use:
curl -LsSf https://astral.sh/uv/install.sh | sh
hash -r

# clone
git clone git@github.com:tlh24/att3ntion
cd att3ntion

# determine CUDA toolkit version:
nvcc --version | grep -oP 'release \K[\d.]+'

# install dependencies (does not compile yet)
# choose cu130 or cu126 based on nvcc output
uv sync --extra cu130 --no-install-package att3ntion
uv sync --extra cu126 --no-install-package att3ntion

# launch the compilation
uv sync --extra cu130
uv sync --extra cu126

source .venv/bin/activate
```


## Basic Usage

```python
from att3ntion import HypergraphAttention

# Create layer (drop-in replacement for attention)
layer = HypergraphAttention(d_model=64, n_heads=4).cuda()

# Forward pass
x = torch.randn(batch_size, seq_len, d_model, device='cuda')
y = layer(x)  # Same shape as input
```

## Synopsis


- **3-way token interactions**: Models relationships between triplets $(Q, R, S)$ instead of pairwise relationships typical in standard self-attention. For more information & motivation, see **Theory**, below.
- **O(N) memory**: The naive implementation would require $O(N^3)$ memory for the attention tensor. This library uses Flash-style tiling and online streaming techniques to reduce memory complexity from $O(N^3)$ to $O(N)$, making 3-way attention feasible. However, the kernels are not yet fully optimized for _speed_ — this is a work in progress!

| Seq Length | att3ntion | Naive O(N³) | Savings |
|------------|-----------|-------------|---------|
| 64         | ~50 MB    | ~67 MB      | 1.3x    |
| 256        | ~200 MB   | ~4.3 GB     | 21x     |
| 1024       | ~800 MB   | ~275 GB     | 344x    |

Run `python demo.py --memory` to benchmark on your hardware.

- **Full autograd support**: Custom CUDA kernels with hand-written backward pass equivalent to torch.autograd. 

---


## Requirements

- Developed and tested on an **NVIDIA RTX 4080**. Should work on any CUDA-capable GPU with compute capability 7.0+ (Volta and newer).
- Python 3.10+
- PyTorch 2.0+ with CUDA support
- CUDA Toolkit 11.8+
- NVIDIA GPU (tested on RTX 4080)
- {You may need to downgrade GCC to work with NVCC}

---

## Theory

Normal attention measures the dot-product similarity between two projected versions the tokens, $Q,K$.  This is passed through a softmax to set the weighting of the $V$ associated with each $K$ - hence vanilla attention acts as a conditional 'get' operator, where information is fetched from key tokens to query tokens.  Explicitly: 

```math
\large
\displaylines{
Q = W_q X \qquad K = W_k X \qquad V = W_v X \\

A[b,h,i,j] = \sum_d Q[b,h,i,d] * K[b,h,j,d] \\

A_o[b,h,i,j] = \frac{e^{A[b,h,i,j]} }{ \sum_j e^{A[b,h,i,j]} } \\

Y[b,h,i,d] = \sum_j A_o[b,h,i,j] * V[b,h,j,d]
}
```
Above, $b$ is the batch dimension, $h$ head dim, $i$ ranges over the query token dim, $j$ over the key token dim, $A$ is raw DP, $A_o$ is attention post-softmax, and $O$ is the output.  Usually you either sum or concatenate to reduce $Y$ over the $h$ dimension. 

[L1 attention](https://github.com/tlh24/l1-attention) experimented with using the L1 norm to measure similarity between $Q$ and $K$. In the course of these experiments, it was realized that you can bidirectionalize attention, so that $Q$ gets info from $K$'s, and $Q$ sends info to $K$ as well - affording both 'gather' and 'scatter' operations. In preliminary experiments, this improves convergence on function-approximation test fixtures by ~ 2x, (depending on the problem type, sometimes bidirectional attention is required for convergence). 
(The 'gather' and 'scatter' $W_V$ can either be tied or independent -- more testing is needed on a wider variety of problems.)

Conditional scatter and gather operations are core elements of CS algorithms, and given enough memory and time[^1], they should be able to approximate any function.  Yet some seemingly fundamental operations are poorly expressed in these conditional message passing primitives.  For example, model inference: based on **structure between pairs** of tokens (detectable, presumably, in the latent space), you ought to modify _another_ token - not either of the pair.  Alternately, you cannot by fiat 'create' a linkage or edge between tokens with normal attention: attention is ephemeral and dependent on structure within the latent dimensions; it cannot be easily modified, only instantiated.  Creating linkages requires being able to 'write' to more than one token at the same time. 

An experimental solution to this problem is to allow for higher-order operations on the graph of relations between tokens; the first step of which is to increase the arity of the attention operation. 

Assume that you measure some similarity between three tokens, $Q,R,S$  ('K' is replaced by 'S' to avoid naming confusion - the corresponding natural indexes are then $i,j,k$).  Raw attention becomes:

```math
\large
\displaylines{
Q = W_q X \qquad R = W_r X \qquad S = W_s X \\

A[..,i,j,k] = \sum_d Q[..,i,d] * R[..,j,d] * S[..,k,d] \\
}
```

I.e. attention is one more term in the summation (this is no longer a dot-product!), and the resulting tensor is one higher dimension.  To propagate information within the tuples, we can do several operations:
1. Reduce along one dimension: e.g. for 'conventional' Q-R interactions, we sum over all S's), then apply softmax to the R dimension, selecting one $V_R$ to write to Q.  *This just reduces to conventional attention by way of softmax.*
2. Reduce and softmax along one dimension, but write twice: for Q-R and Q-S interactions, reduce over S and R, then softmax over R and S, selecting a pair $V_r,V_s$ for writing. *This reduces to conventional attention, with two writes per head.*
3. Softmax serially along two dimensions, write twice: for Q-R and Q-S interactions, softmax over S and R, then softmax over R and S, selecting a pair $V_r,V_s$ for writing. *More interesting due to the (decomposed) 2D softmax, which supports interaction terms.*
4. Reduce and softmax along two dimensions: for Q-R-S interactions, softmax over R-S dimensions, selecting one pair $V_r,V_s$ for writing to Q. *R can modulate which S is gathered, and S can modulate which R is gathered -  promising.*
5. Reduce and softmax along two dimensions: for Q-R-S interactions, softmax over R-S dimensions, select $V_q$ for writing to the pair R,S. *This is the scatter complement to above.*

Options 1-2 reduce to conventional attention; option 3 is more interesting, but options 4 and 5 support the 3-way interactions desired.  

Gather operations: 
```math
\large
\displaylines{
V_q = W_{vq} X \qquad V_r = W_{vr} X \qquad V_s = W_{vs} X \\

A_q[..,i,j,k] = \frac{ e^{A[..,i,j,k]} }{ \sum_{j,k} e^{A[..,i,j,k]} } \\

Y_q[..,i,d] = \sum_{j,k} A_q[..,i,j,k] ( V_r[..,j,d] \diamond V_s[..,k,d] )\\

A_r[..,i,j,k] = \frac{ e^{A[..,i,j,k]} }{ \sum_{i,k} e^{A[..,i,j,k]} } \\

Y_r[..,i,d] = \sum_{i,k} A_r[..,i,j,k] ( V_q[..,i,d] \diamond V_s[..,k,d] )\\

A_s[..,i,j,k] = \frac{ e^{A[..,i,j,k]} }{ \sum_{i,j} e^{A[..,i,j,k]} } \\

Y_s[..,i,d] = \sum_{i,j} A_s[..,i,j,k] ( V_q[..,i,d] \diamond V_r[..,j,d] )\\
}
```
Where $\large \diamond$ is either $\large +$ or $\large *$.  
Scatter operations: 
```math
\large
\displaylines{
V'_q = W'_{vq} X \qquad V'_r = W'_{vr} X \qquad V'_s = W'_{vs} X \\

Y'_r[..,j,d] = \sum_{i,k} A_q[..,i,j,k] * V'_q[..,i,d] 
				\diamond A_s[..,i,j,k] * V'_s[..,k,d] \\

Y'_s[..,k,d] = \sum_{i,j} A_q[..,i,j,k] * V'_q[..,i,d] 
				\diamond A_r[..,i,j,k] * V'_r[..,k,d] \\

Y'_q[..,i,d] = \sum_{j,k} A_r[..,i,j,k] * V'_r[..,j,d] 
				\diamond A_s[..,i,j,k] * V'_s[..,k,d] \\
}
```
As mentioned above, the scatter $V$ and $W$ tensors can be tied to the gather $V$ and $W$.  Finally: 
```math
\large
Y = Y_q + Y'_q + Y_r + Y'_r + Y_s + Y'_s 

```

The obvious problem with calculating directly via above is that you don't want to instantiate a $\large A[b,h,i,j,k]$ tensor -- if the number of tokens is large, this is a huge tensor!  
The above summations would suggest that a full $\large A[..]$ is required for calculating the various softmaxes -- but, given enough floating-point resolution (and stability), all of these operations and their inverses can be calculated in-place, without blowing up GPU memory. 



---

[^1]: Normal transformers can have limitless computation via autoregression, but are highly limited in internal memory by their fixed latent space.  Sure, you can have an expanding list of tokens in the (also limited) context window, but each head is  limited in the number of latent-space "named" global variables.  This is a deep problem that can be _partly_ addressed by hypergraph attention. 

---

## Project Structure

```
att3ntion/                          # repo root
├── att3ntion/                      # Python package
│   ├── __init__.py                 # public API: HypergraphAttention
│   ├── _autograd.py                # CUDA-backed layer + autograd bridge
│   └── _naive.py                   # naive O(N³) reference (testing only)
├── cpp/                            # pybind11 C++ bindings
│   ├── cuda_bindings.cpp/.h        # Python ↔ CUDA glue
│   └── torch_reference.cpp         # torch-based reference kernel
├── cuda/                           # hand-written CUDA kernels
│   ├── forward.cu                  # forward pass
│   ├── backward.cu                 # backward pass
│   └── common.cuh                  # shared constants & device utilities
├── tests/                          # benchmarks, equivalence tests, Makefile
├── demo.py                         # interactive demo
├── setup.py                        # build config (C extensions)
└── pyproject.toml                  # package metadata
```
