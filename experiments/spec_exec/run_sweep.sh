#!/usr/bin/bash

# parallel -j 16 -u 'python train.py --batch-size 16 --epochs 30 --log-name testy --nsamp {1} --repl {2} --device cuda:$(( ({%} - 1) % 8 ))' ::: 0 1 2 4 8 16 32 ::: 1 2 3 4

# parallel -j 16 -u 'python train.py --batch-size 16 --epochs 30 --log-name addmul_hednorm --heads 6 --hidden 384 --repl {1} --attn {2} --device cuda:$(( ({%} - 1) % 4 ))' ::: 1 2 3 4 5 6 7 8 ::: graph hypergraph
# {%} - 1 converts the slot (1-indexed) to the GPU number, so that no two runs are simultaneously executing on the same GPU.


parallel -j 16 -u 'python {1}.py --batch-size 16 --epochs 30 --log-name addmul_{1} --heads 6 --hidden 384 --repl {2} --attn {3} --device cuda:$(( ({%} - 1) % 8 ))' ::: train train_reconfig ::: 5 6 7 8 ::: graph hypergraph

# Note:
# 'add' is for 3 layer HG, 4 layer graph.
# 'addl2' is for 2 layer hg, 4 layer g, looped twice
# 'addl1' is for 3 layer hg, 6 layer g, not looped.
