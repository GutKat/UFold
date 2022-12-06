import random
import numpy as np
from ufold import utils
from ufold import random_generator, postprocess
import os



class random_input():
    def __init__(self, length):
        seq, ss, energy = random_generator.generate_random_seq_and_ss([length])[0]
        self.seq = seq
        self.ss = ss


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


def main(seed= 42):
    utils.seed_torch(seed)
    random.seed(seed)
    N_seqs = 10000
    n_seq = 16*10

    folder_path = f"data/test_files/N{N_seqs}_n{n_seq}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for i in range(N_seqs):
        test = random_input(n_seq)
        create_bbseq_file(test, f"{folder_path}/test{i}.txt")
    print(f"finish creating {N_seqs} random sequences with length of {n_seq}")

if __name__ == '__main__':
    main()


