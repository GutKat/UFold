import random

import numpy as np
from ufold import utils
from Network import U_Net
from torch.optim import Adam
import torch
from torch import nn
from ufold import random_generator, postprocess

utils.seed_torch(42)
random.seed(42)

class random_input():
    def __init__(self, length):
        seq, ss, energy = random_generator.generate_random_seq_and_ss([length])[0]
        self.seq = seq
        self.ss = ss



N_seqs = 1
n_seq = 16*5

def ct2struct(ct):
    stack = list()
    struct = list()
    for i in range(len(ct)):
        if ct[i] == '(':
            stack.append(i)
        if ct[i] == ')':
            left = stack.pop()
            struct.append([left, i])
    return struct

import os

def create_bbseq_file(sample, path):
    filename = os.path.split(path)[1]
    bbseqs = []

    pairs = ct2struct(sample.ss)
    paired = [0] * len(sample.ss)
    for pair in pairs:
        le = pair[0]
        ri = pair[1]
        paired[le] = ri + 1
        paired[ri] = le + 1
    bbseq = {"seq": sample.seq, "pair": paired}
    bbseqs.append(bbseq)

    with open(path, "w") as f:
        seq = bbseq["seq"]
        pairs = bbseq["pair"]
        n = len(seq)
        for index in range(n):
            line = [str(index+1), str(seq[index]), str(pairs[index])]
            line = " ".join(line)
            f.write(line)
            if index != n-1:
                f.write("\n")
    return None

for i in range(100):
    test = random_input(n_seq)
    create_bbseq_file(test, f"test_files/test{i}.txt")



