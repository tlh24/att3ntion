#!/usr/bin/bash

parallel -j 16 -u 'python train.py --batch-size 16 --epochs 30 --log-name testy --nsamp {1} --repl {2} --device cuda:$(( ({%} - 1) % 8 ))' ::: 0 1 2 4 8 16 32 ::: 1 2 3 4
# {%} - 1 converts the slot (1-indexed) to the GPU number, so that no two runs are simultaneously executing on the same GPU.
