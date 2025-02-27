# att3ntion
Hypergraph attention: attention between three tokens.  

Normal attention measures the dot-product similarity between two projected versions the tokens, $Q,K$.  This is passed through a softmax to set the weighting of the $V$ associated with each $K$ - hence vanilla attention acts as a conditional 'get' operator, where information is fetched from key tokens to query tokens.  Explicitly: 

```math
\large
\displaylines{
Q = W_Q X ; K = W_K X ; V = W_V X \\

A[b,h,i,j] = \sum_d Q[b,h,i,d] * K[b,h,j,d] \\

A_o[b,h,i,j] = \frac{e^{A[b,h,i,j]} }{ \sum_j e^{A[b,h,i,j]} } \\

Y[b,h,i,d] = \sum_j A_o[b,h,i,j] * V[b,h,j,d]
}
```
Above, $b$ is the batch dimension, $h$ head dim, $i$ ranges over the query token dim, $j$ over the key token dim, $A$ is raw DP, $A_o$ is attention post-softmax, and $O$ is the output.  Usually you either sum or concatenate to reduce $Y$ over the $h$ dimension. 

[L1 attention](https://github.com/tlh24/l1-attention) experimented with using the L1 norm to measure similarity between $Q$ and $K$. In the course of these experiments, we realized that you can bidirectionalize attention, so that $Q$ gets info from $K$'s, and $Q$ sends info to $K$ as well - affording both 'gather' and 'scatter' operations. In preliminary experiments, this improves convergence on function-approximation test fixtures by ~ 2x, (depending on the problem type). 
The 'gather' and 'scatter' $W_V$ can either be tied or independent -- more testing is needed on a wider variety of problems here. 

Conditional scatter and gather operations are core elements of CS algorithms, and given enough memory, they should be able to approximate any function.  Yet some seemingly fundamental operations are poorly expressed in these conditional message passing primitives.  For example, model inference: based on structure between pairs of tokens (detectable, presumably, in the latent space), you ought to modify another token - not either of the pair.  Alternately, you cannot by fiat 'create' a linkage or edge between tokens with normal attention: attention is ephemeral and dependent on structure within the latent dimensions, and cannot be facily modified, only instantiated.  

An obvious and very experimental solution to this problem is to allow for higher-order operations on the graph of relations between tokens; the first step of which is to increase the arity of the operation.  

Therefore assume that you measure some similarity between three tokens, $Q,R,S$  ('K' is replaced by 'S' to avoid naming confusion - the corresponding natural indexes are then $i,j,k$).  Raw attention becomes:

```math
\large
\displaylines{
Q = W_Q X ; R = W_R X ; S = W_S X \\

A[..,i,j,k] = \sum_d Q[..,i,d] * R[..,j,d] * S[..,k,d] \\
}
```

I.e. attention is one more term in the summation (this is no longer a dot-product!), and the resulting tensor is one higher dimension.  To propagate information within the tuples, we can do many things:
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
V_q = W_vq X ; V_r = W_vr X ; V_s = W_vs X \\

A_q[..,i,j,k] = \frac{ e^{A[..,i,j,k]} }{ \sum_{j,k} e^{A[..,i,j,k]} } \\

Y_q[..,i,d] = \sum_{j,k} A_q[..,i,j,k] ( V_r[..,j,d] \diamond V_s[;;,k,d] )\\

A_r[..,i,j,k] = \frac{ e^{A[..,i,j,k]} }{ \sum_{i,k} e^{A[..,i,j,k]} } \\

Y_r[..,i,d] = \sum_{i,k} A_r[..,i,j,k] ( V_q[..,i,d] \diamond V_s[;;,k,d] )\\

A_s[..,i,j,k] = \frac{ e^{A[..,i,j,k]} }{ \sum_{i,j} e^{A[..,i,j,k]} } \\

Y_s[..,i,d] = \sum_{i,j} A_s[..,i,j,k] ( V_q[..,i,d] \diamond V_r[;;,j,d] )\\
}
```
Where $\large \diamond$ is either $\large +$ or $\large *$.  
Scatter operations: 
```math
\large
\displaylines{
V'_q = W'_vq X ; V'_r = W'_vr X ; V'_s = W'_vs X \\

Y'_r[..,j,d] = \sum_{i,k} A_q[..,i,j,k] * V'_q[..,i,d] 
				+ A_s[..,i,j,k] * V'_s[..,k,d] \\

Y'_s[..,k,d] = \sum_{i,j} A_q[..,i,j,k] * V'_q[..,i,d] 
				+ A_r[..,i,j,k] * V'_r[..,k,d] \\

Y'_q[..,i,d] = \sum_{j,k} A_r[..,i,j,k] * V'_r[..,j,d] 
				+ A_s[..,i,j,k] * V'_s[..,k,d] \\
}
```
As mentioned above, $\large W'_{??}$ and $\large V'_{??}$ can be tied to W_{??}$ and $\large V_{??}$.  

Finally: 
```math
\large
Y = Y_q + Y'_q + Y_r + Y'_r + Y_s + Y'_s 

```

The obvious problem with calculating directly via above is that you don't want to instantiate a $\large A[b,h,i,j,k]$ tensor -- if the number of tokens is large, this is a huge tensor!  
The above summations would suggest that a full $\large A[]$ is required for calculating the various softmaxes -- but, given enough floating-point resolution (and stability), all of these operations and their inverses can be calculated in-place, without blowing up GPU memory. 
