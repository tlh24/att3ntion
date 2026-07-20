#!/usr/bin/bash

# parallel -j 16 -u 'python train.py --batch-size 16 --epochs 30 --log-name testy --nsamp {1} --repl {2} --device cuda:$(( ({%} - 1) % 8 ))' ::: 0 1 2 4 8 16 32 ::: 1 2 3 4

parallel -j 16 -u 'python train.py --batch-size 16 --epochs 30 --log-name addmul2 --heads 8 --hidden 512 --repl {1} --attn {2} --device cuda:$(( ({%} - 1) % 4 ))' ::: 1 2 3 4 5 6 7 8 ::: graph hypergraph
# {%} - 1 converts the slot (1-indexed) to the GPU number, so that no two runs are simultaneously executing on the same GPU.


# Note:
# 'add' is for 3 layer HG, 4 layer graph.
# 'addl2' is for 2 layer hg, 4 layer g, looped twice
# 'addl1' is for 3 layer hg, 6 layer g, not looped.

# for experiments below, HG has 3 layers, G has 6, not looped.
# addmul: 6 heads, 384 hidden dim
# addmul2: 8 heads, 512 hidden
# addmul3: 12 heads, 768 hidden
# addsubmul: 12 heads, 768 hidden
