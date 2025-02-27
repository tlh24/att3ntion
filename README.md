# att3ntion
Hypergraph attention: attention between three tokens.  

Normal attention measures the dot-product similarity between two projected versions the tokens, $Q,K$.  This is passed through a softmax to set the weighting of the $V$ associated with each $K$ - hence vanilla attention acts as a conditional 'get' operator, where information is fetched from key tokens to query tokens.  Explicitly: 

```math
\large
\displaylines{
A[b,h,i,j] = \sum_d Q[b,h,i,d] * K[b,h,j,d] \\

A_o[b,h,i,j] = \frac{e^{A[b,h,i,j]} }{ \sum_j e^{A[b,h,i,j]} } \\

O[b,h,i,d] = \sum_j A_o[b,h,i,j] * V[b,h,j,d]
}
```
Above, $b$ is the batch dimension, $h$ head dim, $i$ ranges over the query token dim, $j$ over the key token dim, $A$ is raw DP, $A_o$ is attention post-softmax, and $O$ is the output. 

[L1 attention](https://github.com/tlh24/l1-attention) experimented with using the L1 norm to measure similarity between $Q$ and $K$. In the course of these experiments, we realized that you can bidirectionalize attention, so that $Q$ gets info from $K$'s, and $Q$ sends info to $K$ as well - affording both 'gather' and 'scatter' operations. In perliminary experiments, this improves convergence on function-approximation test fixtures by ~ 2x, (depending on the problem type). 

Conditional scatter and gather operations are core elements of CS algorithms, and given enough memory, they should be able to approximate any function.  Yet some seemingly fundamental operations are poorly expressed in these conditional message passing primitives.  For example, model inference: based on structure between pairs of tokens (detectable, presumably, in the latent space), you ought to modify another token - not either of the pair.  Alternately, you cannot by fiat 'create' a linkage or edge between tokens with normal attention: attention is ephemeral and dependent on structure within the latent dimensions, and cannot be facily modified, only instantiated.  

An obvious and very experimental solution to this problem is to allow for higher-order operations on the graph of relations between tokens; the first step of which is to increase the arity of the opeartion.  

Therefore assume that you measure some similarity between three tokens, $Q,R,S$  ('K' is replaced by 'S' to avoid indexing confusion - the corresponding natural indexes are then $i,j,k$).  Raw attention becomes:

```math
\large
A[..,i,j,k] = \sum_d Q[..,i,d] * R[..,j,d] * S[..,k,d] \\
```
I.e. attention is just one more term in the summation (this is no longer a dot-product!), and the resulting tensor is one higher dimension.  To get the pairwise interactions, we can do three things:
* Reduce along one dimension: e.g. for conventinal Q-R interactions, we sum over all possible R's), then apply softmax to the K dimension, selecting one $V_K$ to write to Q.)
* Reduce and softmax along one dimension, but write twice: for Q-R and Q-S interactions, reduce over S and R, then softmax over R and S, selecting a pair $V_r,V_s$ for writing
* Softmax serially along two dimensions, write twice: for Q-R and Q-S interactions, softmax over S and R, then softmax over R and S, selecting a pair $V_r,V_s$ for writing
* Reduce and softmax along two dimensions: for Q-R-S interactions, softmax over R-S dimensions, selecting one pair $V_r,V_s$ for writing to Q)
* Reduce and softmax along two dimensions; for Q-R-S interactions, softmax over R-S dimensons, select $V_q$ for writing to the pair R,S.
```math
\large
\displaylines{
A_{qs}[..,i,j] = \sum_{k} A[..,i,j,k] \\

A_{qr}[..,i,k] = \sum_{k} A[..,i,j,k] \\

A_{sr}[..,j,k] = \sum_{i} A[..,i,j,k] \\

A'_{qk}[..,i,j] = \frac{ e^{A_{qk}[..,i,j]} }{ \sum_j e^{A_{qk}[..,i,j]} } \\

A'_{kq}[..,i,j] = \frac{ e^{A_{qk}[..,i,j]} }{ \sum_i e^{A_{qk}[..,i,j]} } \\

A'_{qr}[..,i,k] = \frac{ e^{A_{qr}[..,i,k]} }{ \sum_k e^{A_{qr}[..,i,k]} } \\

A'_{rq}[..,i,k] = \frac{ e^{A_{qr}[..,i,k]} }{ \sum_i e^{A_{qr}[..,i,k]} } \\

A'_{kr}[..,j,k] = \frac{ e^{A_{kr}[..,j,k]} }{ \sum_j e^{A_{kr}[..,j,k]} } \\

A'_{rk}[..,j,k] = \frac{ e^{A_{kr}[..,j,k]} }{ \sum_k e^{A_{kr}[..,j,k]} } \\

O_q[..,i,d] = \sum_j A'_{qk}[..,i,j] * V_k[..,j,d] + \sum_k A'_{qr}[..,i,k] * V_r[..,k,d] \\

O_k[..,j,d] = \sum_i A'_{kq}[..,i,j] * V_q[..,i,d] + \sum_k A'_{kr}[..,j,k] * V_r[..,k,d] \\

O_r[..,k,d] = \sum_i A'_{rq}[..,i,k] * V_q[..,i,d] + \sum_j A'_{rk}[..,j,k] * V_k[..,j,d] \\
}
```
